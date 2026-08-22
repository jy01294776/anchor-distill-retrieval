from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anchor_distill.config import Settings
from anchor_distill.constants import JobType
from anchor_distill.data import (
    anchors_from_categories,
    load_banking77,
    stratified_few_shot,
)
from anchor_distill.evaluation import evaluate_retriever, save_result
from anchor_distill.jobs.models import Job
from anchor_distill.jobs.repository import JobRepository
from anchor_distill.jobs.service import JobHandler
from anchor_distill.retrieval import build_index, save_index_manifest
from anchor_distill.schemas import TeacherRecord
from anchor_distill.teacher import TeacherClient


def _atomic_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""))
    os.replace(temporary, path)


def _load_records(path: Path) -> list[TeacherRecord]:
    if not path.exists():
        return []
    return [
        TeacherRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _relative_path(settings: Settings, value: str, default: tuple[str, ...]) -> Path:
    parts = tuple(Path(value).parts) if value else default
    if Path(value).is_absolute():
        raise ValueError("job payload paths must be project-relative")
    return settings.path(*parts)


def build_handlers(settings: Settings) -> dict[JobType, JobHandler]:
    raw_dir = settings.path("data", "raw", "banking77")

    def teacher_label(job: Job, repository: JobRepository) -> dict[str, Any]:
        pairs = job.payload.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("teacher_label payload requires a non-empty pairs list")
        output = _relative_path(
            settings,
            str(job.payload.get("output", "")),
            ("data", "interim", f"teacher_{job.id}.jsonl"),
        )
        records = _load_records(output)
        keyed = {(record.query_id, record.anchor_id): record for record in records}
        client = TeacherClient(settings)
        completed = int((job.checkpoint or {}).get("completed_pairs", 0))
        for index, pair in enumerate(pairs[completed:], start=completed):
            if not isinstance(pair, dict):
                raise ValueError("each pair must be an object")
            key = (str(pair["query_id"]), str(pair["anchor_id"]))
            if key not in keyed:
                keyed[key] = client.score_pair(
                    query_id=key[0],
                    query=str(pair["query"]),
                    anchor_id=key[1],
                    anchor=str(pair["anchor"]),
                )
                _atomic_jsonl(
                    output,
                    [
                        record.model_dump_json()
                        for record in sorted(
                            keyed.values(),
                            key=lambda value: (value.query_id, value.anchor_id),
                        )
                    ],
                )
            repository.save_checkpoint(
                job.id,
                {
                    "completed_pairs": index + 1,
                    "output": str(output.relative_to(settings.project_root)),
                },
            )
        return {
            "completed_pairs": len(pairs),
            "output": str(output.relative_to(settings.project_root)),
        }

    def build_teacher_dataset(
        job: Job,
        repository: JobRepository,
    ) -> dict[str, Any]:
        del repository
        source = _relative_path(
            settings,
            str(job.payload.get("input", "")),
            ("data", "interim", "teacher_pilot.jsonl"),
        )
        output = _relative_path(
            settings,
            str(job.payload.get("output", "")),
            ("data", "processed", f"teacher_dataset_{job.id}.jsonl"),
        )
        records = _load_records(source)
        accepted = [record for record in records if record.accepted]
        _atomic_jsonl(output, [record.model_dump_json() for record in accepted])
        return {
            "input_records": len(records),
            "accepted_records": len(accepted),
            "output": str(output.relative_to(settings.project_root)),
        }

    def train_student(job: Job, repository: JobRepository) -> dict[str, Any]:
        del repository
        from anchor_distill.training import RetrievalTrainer

        examples = load_banking77(raw_dir)
        anchors = anchors_from_categories(example.category for example in examples)
        mode = str(job.payload.get("mode", "gold"))
        shots = int(job.payload.get("shots", 1))
        selected = stratified_few_shot(examples, shots, seed=settings.seed)
        teacher_records: list[TeacherRecord] = []
        if mode != "gold":
            teacher_path = _relative_path(
                settings,
                str(job.payload.get("teacher_records", "")),
                ("data", "processed", "teacher_dataset.jsonl"),
            )
            teacher_records = _load_records(teacher_path)
        query_text = {example.example_id: example.text for example in examples}
        output = _relative_path(
            settings,
            str(job.payload.get("output", "")),
            ("models", f"{mode}_{shots}shot_{job.id}"),
        )
        trainer = RetrievalTrainer(
            base_model=str(
                job.payload.get(
                    "base_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            ),
            seed=settings.seed,
        )
        manifest = trainer.train(
            mode=mode,
            examples=selected,
            anchors=anchors,
            teacher_records=teacher_records,
            query_text=query_text,
            epochs=int(job.payload.get("epochs", 2)),
            batch_size=int(job.payload.get("batch_size", 32)),
            hybrid_lambda=float(job.payload.get("hybrid_lambda", 0.5)),
            output_dir=output,
            label_budget=len(selected),
        )
        return {
            "run_id": manifest.run_id,
            "model_sha256": manifest.model_sha256,
            "output": str(output.relative_to(settings.project_root)),
        }

    def evaluate_model(job: Job, repository: JobRepository) -> dict[str, Any]:
        del repository
        from sentence_transformers import SentenceTransformer

        from anchor_distill.training import _device_name

        examples = load_banking77(raw_dir)
        anchors = anchors_from_categories(example.category for example in examples)
        model_name = str(
            job.payload.get(
                "model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        model = SentenceTransformer(model_name, device=_device_name())
        result = evaluate_retriever(
            model,
            examples,
            anchors,
            model_version=model_name,
            split="test",
        )
        output = _relative_path(
            settings,
            str(job.payload.get("output", "")),
            ("artifacts", "benchmarks", f"evaluation_{job.id}.json"),
        )
        save_result(output, result)
        return {
            "recall_at_1": result.recall_at_1,
            "mrr": result.mrr,
            "output": str(output.relative_to(settings.project_root)),
        }

    def build_retrieval_index(
        job: Job,
        repository: JobRepository,
    ) -> dict[str, Any]:
        del repository
        from sentence_transformers import SentenceTransformer

        from anchor_distill.training import _device_name

        examples = load_banking77(raw_dir)
        anchors = anchors_from_categories(example.category for example in examples)
        model_name = str(
            job.payload.get(
                "model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        model = SentenceTransformer(model_name, device=_device_name())
        index = build_index(
            model,
            anchors,
            model_version=model_name,
            examples=examples,
        )
        output = _relative_path(
            settings,
            str(job.payload.get("output", "")),
            ("models", f"index_{job.id}.npz"),
        )
        index.save(output)
        manifest = output.with_suffix(".manifest.json")
        save_index_manifest(manifest, index)
        return {
            "anchors": len(index.anchor_ids),
            "evidence_records": len(index.evidence_ids),
            "output": str(output.relative_to(settings.project_root)),
        }

    return {
        JobType.TEACHER_LABEL: teacher_label,
        JobType.BUILD_TEACHER_DATASET: build_teacher_dataset,
        JobType.TRAIN_STUDENT: train_student,
        JobType.EVALUATE_MODEL: evaluate_model,
        JobType.BUILD_INDEX: build_retrieval_index,
    }
