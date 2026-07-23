from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from anchor_distill.data import (
    anchors_from_categories,
    load_banking77,
    stratified_few_shot,
)
from anchor_distill.evaluation import calibrate_teacher, save_result
from anchor_distill.schemas import TeacherRecord


@dataclass(frozen=True)
class TeacherPilotResult:
    records: int
    queries: int
    candidates_per_query: int
    accepted_records: int
    output_path: str
    calibration_path: str


class TeacherScorer(Protocol):
    def score_pair(
        self,
        *,
        query_id: str,
        query: str,
        anchor_id: str,
        anchor: str,
    ) -> TeacherRecord: ...


def _negative_anchors(
    query_id: str,
    gold_anchor: str,
    anchor_ids: list[str],
    count: int,
) -> list[str]:
    candidates = [anchor_id for anchor_id in anchor_ids if anchor_id != gold_anchor]
    return sorted(
        candidates,
        key=lambda anchor_id: hashlib.sha256(
            f"{query_id}:{anchor_id}".encode()
        ).hexdigest(),
    )[:count]


def _write_records(path: Path, records: list[TeacherRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n"
    )
    os.replace(temporary, path)


def run_teacher_pilot(
    *,
    raw_dir: Path,
    output_path: Path,
    calibration_path: Path,
    query_count: int,
    negative_count: int,
    seed: int,
    client: TeacherScorer,
) -> TeacherPilotResult:
    if query_count < 1 or negative_count < 1:
        raise ValueError("query_count and negative_count must be positive")
    examples = load_banking77(raw_dir)
    anchors = anchors_from_categories(example.category for example in examples)
    selected = stratified_few_shot(examples, 1, seed=seed)[:query_count]
    records: list[TeacherRecord] = []
    anchor_ids = sorted(anchors)
    for example in selected:
        candidates = [example.category]
        candidates.extend(
            _negative_anchors(
                example.example_id,
                example.category,
                anchor_ids,
                negative_count,
            )
        )
        for anchor_id in candidates:
            records.append(
                client.score_pair(
                    query_id=example.example_id,
                    query=example.text,
                    anchor_id=anchor_id,
                    anchor=anchors[anchor_id],
                )
            )
    _write_records(output_path, records)
    calibration = calibrate_teacher(
        records,
        {example.example_id: example.category for example in selected},
    )
    save_result(calibration_path, calibration)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "pilot": asdict(
                    TeacherPilotResult(
                        records=len(records),
                        queries=len(selected),
                        candidates_per_query=negative_count + 1,
                        accepted_records=sum(record.accepted for record in records),
                        output_path=str(output_path),
                        calibration_path=str(calibration_path),
                    )
                ),
                "model_resolved": sorted({record.model_resolved for record in records}),
                "prompt_hash": sorted({record.prompt_hash for record in records}),
                "rubric_id": sorted({record.rubric_id for record in records}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TeacherPilotResult(
        records=len(records),
        queries=len(selected),
        candidates_per_query=negative_count + 1,
        accepted_records=sum(record.accepted for record in records),
        output_path=str(output_path),
        calibration_path=str(calibration_path),
    )
