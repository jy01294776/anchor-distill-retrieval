from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anchor_distill.data import BankingExample
from anchor_distill.schemas import TeacherRecord


@dataclass(frozen=True)
class TrainingManifest:
    run_id: str
    mode: str
    base_model: str
    seed: int
    epochs: int
    learning_rate: float
    examples: int
    accepted_teacher_records: int
    label_budget: int
    hybrid_lambda: float | None
    output_path: str
    model_sha256: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def model_directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = (
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name != "training_manifest.json"
    )
    for file in sorted(files):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path: Path, manifest: TrainingManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _device_name(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OrdinalCalibrationHead:
    """Lazy wrapper to keep torch optional for non-ML commands."""

    @staticmethod
    def build() -> Any:
        import torch

        class _Head(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.raw_scale = torch.nn.Parameter(torch.tensor(1.0))
                self.raw_threshold_deltas = torch.nn.Parameter(torch.zeros(4))

            def forward(self, similarities: Any) -> Any:
                import torch.nn.functional as functional

                scale = functional.softplus(self.raw_scale) + 1e-4
                deltas = functional.softplus(self.raw_threshold_deltas) + 1e-4
                thresholds = torch.cumsum(deltas, dim=0)
                thresholds = thresholds - thresholds.mean()
                cumulative = torch.sigmoid(
                    scale * (similarities.unsqueeze(-1) - thresholds)
                )
                probabilities = torch.cat(
                    [
                        1 - cumulative[..., :1],
                        cumulative[..., :-1] - cumulative[..., 1:],
                        cumulative[..., -1:],
                    ],
                    dim=-1,
                )
                return probabilities.clamp_min(1e-8)

        return _Head()


def soft_kl_loss(student_probabilities: Any, teacher_probabilities: Any) -> Any:
    import torch.nn.functional as functional

    return functional.kl_div(
        student_probabilities.log(),
        teacher_probabilities,
        reduction="batchmean",
    )


def teacher_expected_relevance(record: TeacherRecord) -> float:
    return sum(
        rating * probability
        for rating, probability in enumerate(
            record.label_probabilities,
            start=1,
        )
    )


class RetrievalTrainer:
    def __init__(
        self,
        *,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        seed: int = 20260722,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        set_seed(seed)
        self.torch = torch
        self.device = _device_name(device)
        self.model = SentenceTransformer(base_model, device=self.device)
        self.ordinal_head = OrdinalCalibrationHead.build().to(self.device)
        self.base_model = base_model
        self.seed = seed

    def _encode(self, texts: Sequence[str]) -> Any:
        features = self.model.preprocess(list(texts))
        features = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in features.items()
        }
        return self.model(features)["sentence_embedding"]

    def _gold_loss(
        self,
        batch: Sequence[BankingExample],
        anchor_ids: Sequence[str],
        anchor_texts: Sequence[str],
    ) -> Any:
        import torch.nn.functional as functional

        query_embeddings = functional.normalize(
            self._encode([example.text for example in batch]), dim=-1
        )
        anchor_embeddings = functional.normalize(self._encode(anchor_texts), dim=-1)
        logits = query_embeddings @ anchor_embeddings.T / 0.05
        anchor_lookup = {anchor_id: index for index, anchor_id in enumerate(anchor_ids)}
        targets = self.torch.tensor(
            [anchor_lookup[example.category] for example in batch],
            device=self.device,
        )
        return functional.cross_entropy(logits, targets)

    def _teacher_loss(
        self,
        records: Sequence[TeacherRecord],
        query_text: dict[str, str],
        anchor_text: dict[str, str],
        *,
        hard: bool,
    ) -> Any:
        import torch.nn.functional as functional

        accepted = [record for record in records if record.accepted]
        query_embeddings = functional.normalize(
            self._encode([query_text[record.query_id] for record in accepted]), dim=-1
        )
        anchor_embeddings = functional.normalize(
            self._encode([anchor_text[record.anchor_id] for record in accepted]), dim=-1
        )
        similarities = (query_embeddings * anchor_embeddings).sum(dim=-1)
        student_probabilities = self.ordinal_head(similarities)
        if hard:
            targets = self.torch.tensor(
                [record.selected_rating - 1 for record in accepted],
                device=self.device,
            )
            return functional.nll_loss(student_probabilities.log(), targets)
        teacher = self.torch.tensor(
            [record.label_probabilities for record in accepted],
            dtype=student_probabilities.dtype,
            device=self.device,
        )
        return soft_kl_loss(student_probabilities, teacher)

    def _listwise_teacher_loss(
        self,
        groups: Sequence[Sequence[TeacherRecord]],
        query_text: dict[str, str],
        anchor_text: dict[str, str],
        *,
        hard: bool,
        teacher_temperature: float = 1.0,
        student_temperature: float = 0.05,
    ) -> Any:
        import torch.nn.functional as functional

        if not groups:
            raise ValueError("listwise teacher groups are required")
        accepted_groups = [
            [record for record in group if record.accepted] for group in groups
        ]
        candidate_counts = {len(group) for group in accepted_groups}
        if 0 in candidate_counts or len(candidate_counts) != 1:
            raise ValueError(
                "listwise batches require equal non-zero candidates per query"
            )
        candidates_per_query = candidate_counts.pop()
        query_ids = [group[0].query_id for group in accepted_groups]
        query_embeddings = functional.normalize(
            self._encode([query_text[query_id] for query_id in query_ids]),
            dim=-1,
        )
        flat_records = [record for group in accepted_groups for record in group]
        anchor_embeddings = functional.normalize(
            self._encode([anchor_text[record.anchor_id] for record in flat_records]),
            dim=-1,
        ).reshape(len(accepted_groups), candidates_per_query, -1)
        similarities = self.torch.einsum(
            "bd,bkd->bk",
            query_embeddings,
            anchor_embeddings,
        )
        student_logits = similarities / student_temperature
        teacher_scores = self.torch.tensor(
            [
                [teacher_expected_relevance(record) for record in group]
                for group in accepted_groups
            ],
            dtype=student_logits.dtype,
            device=self.device,
        )
        if hard:
            return functional.cross_entropy(
                student_logits,
                teacher_scores.argmax(dim=1),
            )
        teacher_probabilities = functional.softmax(
            teacher_scores / teacher_temperature,
            dim=1,
        )
        return functional.kl_div(
            functional.log_softmax(student_logits, dim=1),
            teacher_probabilities,
            reduction="batchmean",
        )

    def train(
        self,
        *,
        mode: str,
        examples: Sequence[BankingExample],
        anchors: dict[str, str],
        teacher_records: Sequence[TeacherRecord] = (),
        query_text: dict[str, str] | None = None,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        hybrid_lambda: float = 0.5,
        output_dir: Path,
        label_budget: int = -1,
    ) -> TrainingManifest:
        valid_modes = {
            "gold",
            "hard",
            "soft",
            "hybrid",
            "listwise_hard",
            "listwise_soft",
            "listwise_hybrid",
        }
        if mode not in valid_modes:
            raise ValueError(f"unsupported training mode: {mode}")
        teacher_modes = valid_modes - {"gold"}
        if mode in teacher_modes and not teacher_records:
            raise ValueError("teacher records are required for distillation")
        if not 0 <= hybrid_lambda <= 1:
            raise ValueError("hybrid_lambda must be in [0, 1]")

        parameters = list(self.model.parameters()) + list(
            self.ordinal_head.parameters()
        )
        optimizer = self.torch.optim.AdamW(parameters, lr=learning_rate)
        anchor_ids = sorted(anchors)
        anchor_texts = [anchors[anchor_id] for anchor_id in anchor_ids]
        queries = query_text or {
            example.example_id: example.text for example in examples
        }
        rng = random.Random(self.seed)
        accepted_records = [record for record in teacher_records if record.accepted]
        if mode in teacher_modes and not accepted_records:
            raise ValueError("no accepted teacher records are available")
        grouped_records: dict[str, list[TeacherRecord]] = {}
        for record in accepted_records:
            grouped_records.setdefault(record.query_id, []).append(record)
        teacher_groups = [
            sorted(group, key=lambda record: record.anchor_id)
            for _, group in sorted(grouped_records.items())
        ]
        listwise = mode.startswith("listwise_")
        hybrid = mode in {"hybrid", "listwise_hybrid"}
        hard = mode in {"hard", "listwise_hard"}

        for _epoch in range(epochs):
            gold_order = list(examples)
            teacher_order = list(accepted_records)
            teacher_group_order = list(teacher_groups)
            rng.shuffle(gold_order)
            rng.shuffle(teacher_order)
            rng.shuffle(teacher_group_order)
            if mode == "gold":
                steps = max(1, (len(gold_order) + batch_size - 1) // batch_size)
            elif listwise and not hybrid:
                steps = max(
                    1,
                    (len(teacher_group_order) + batch_size - 1) // batch_size,
                )
            elif mode in {"hard", "soft"}:
                steps = max(1, (len(teacher_order) + batch_size - 1) // batch_size)
            else:
                teacher_length = (
                    len(teacher_group_order) if listwise else len(teacher_order)
                )
                steps = max(
                    1,
                    (len(gold_order) + batch_size - 1) // batch_size,
                    (teacher_length + batch_size - 1) // batch_size,
                )
            for step in range(steps):
                gold_offset = (step * batch_size) % max(1, len(gold_order))
                teacher_offset = (step * batch_size) % max(1, len(teacher_order))
                teacher_group_offset = (step * batch_size) % max(
                    1,
                    len(teacher_group_order),
                )
                batch = gold_order[gold_offset : gold_offset + batch_size]
                teacher_batch = teacher_order[
                    teacher_offset : teacher_offset + batch_size
                ]
                teacher_group_batch = teacher_group_order[
                    teacher_group_offset : teacher_group_offset + batch_size
                ]
                optimizer.zero_grad()
                if mode == "gold":
                    loss = self._gold_loss(batch, anchor_ids, anchor_texts)
                else:
                    if listwise:
                        teacher_loss = self._listwise_teacher_loss(
                            teacher_group_batch,
                            queries,
                            anchors,
                            hard=hard,
                        )
                    else:
                        teacher_loss = self._teacher_loss(
                            teacher_batch,
                            queries,
                            anchors,
                            hard=hard,
                        )
                    if hybrid:
                        gold_loss = self._gold_loss(batch, anchor_ids, anchor_texts)
                        loss = (
                            hybrid_lambda * teacher_loss
                            + (1 - hybrid_lambda) * gold_loss
                        )
                    else:
                        loss = teacher_loss
                loss.backward()
                optimizer.step()

        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))
        self.torch.save(
            self.ordinal_head.state_dict(),
            output_dir / "ordinal_head.pt",
        )
        digest = model_directory_sha256(output_dir)
        recorded_hybrid_lambda = hybrid_lambda if hybrid else None
        run_material = {
            "mode": mode,
            "base_model": self.base_model,
            "seed": self.seed,
            "label_budget": label_budget,
            "example_ids": sorted(example.example_id for example in examples),
            "teacher_pairs": sorted(
                (record.query_id, record.anchor_id)
                for record in teacher_records
                if record.accepted
            ),
            "hybrid_lambda": recorded_hybrid_lambda,
        }
        run_id = hashlib.sha256(
            json.dumps(run_material, sort_keys=True).encode()
        ).hexdigest()[:16]
        manifest = TrainingManifest(
            run_id=run_id,
            mode=mode,
            base_model=self.base_model,
            seed=self.seed,
            epochs=epochs,
            learning_rate=learning_rate,
            examples=len(examples),
            accepted_teacher_records=sum(record.accepted for record in teacher_records),
            label_budget=label_budget,
            hybrid_lambda=recorded_hybrid_lambda,
            output_path=str(output_dir),
            model_sha256=digest,
        )
        save_manifest(output_dir / "training_manifest.json", manifest)
        return manifest
