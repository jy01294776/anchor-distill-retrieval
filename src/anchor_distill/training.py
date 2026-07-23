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
    files = (candidate for candidate in path.rglob("*") if candidate.is_file())
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
        if mode not in {"gold", "hard", "soft", "hybrid"}:
            raise ValueError(f"unsupported training mode: {mode}")
        if mode in {"hard", "soft", "hybrid"} and not teacher_records:
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
        if mode in {"hard", "soft", "hybrid"} and not accepted_records:
            raise ValueError("no accepted teacher records are available")

        for _epoch in range(epochs):
            gold_order = list(examples)
            teacher_order = list(accepted_records)
            rng.shuffle(gold_order)
            rng.shuffle(teacher_order)
            if mode == "gold":
                steps = max(1, (len(gold_order) + batch_size - 1) // batch_size)
            elif mode in {"hard", "soft"}:
                steps = max(1, (len(teacher_order) + batch_size - 1) // batch_size)
            else:
                steps = max(
                    1,
                    (len(gold_order) + batch_size - 1) // batch_size,
                    (len(teacher_order) + batch_size - 1) // batch_size,
                )
            for step in range(steps):
                gold_offset = (step * batch_size) % max(1, len(gold_order))
                teacher_offset = (step * batch_size) % max(1, len(teacher_order))
                batch = gold_order[gold_offset : gold_offset + batch_size]
                teacher_batch = teacher_order[
                    teacher_offset : teacher_offset + batch_size
                ]
                optimizer.zero_grad()
                if mode == "gold":
                    loss = self._gold_loss(batch, anchor_ids, anchor_texts)
                else:
                    teacher_loss = self._teacher_loss(
                        teacher_batch,
                        queries,
                        anchors,
                        hard=mode == "hard",
                    )
                    if mode == "hybrid":
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
            "hybrid_lambda": hybrid_lambda if mode == "hybrid" else None,
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
            hybrid_lambda=hybrid_lambda if mode == "hybrid" else None,
            output_path=str(output_dir),
            model_sha256=digest,
        )
        save_manifest(output_dir / "training_manifest.json", manifest)
        return manifest
