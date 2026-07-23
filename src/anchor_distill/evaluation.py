from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anchor_distill.data import BankingExample
from anchor_distill.metrics import (
    expected_calibration_error,
    ranking_metrics,
    softmax,
)
from anchor_distill.schemas import TeacherRecord


@dataclass(frozen=True)
class EvaluationResult:
    model_version: str
    split: str
    examples: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    ndcg_at_10: float
    macro_f1: float
    ece: float


@dataclass(frozen=True)
class JudgeCalibration:
    rubric_id: str
    records: int
    accepted_rate: float
    exact_match_accuracy: float
    brier_score: float
    ece: float
    mean_recognized_mass: float


def evaluate_retriever(
    model: Any,
    examples: Sequence[BankingExample],
    anchors: dict[str, str],
    *,
    model_version: str,
    split: str = "test",
) -> EvaluationResult:
    selected = [example for example in examples if example.split == split]
    if not selected:
        raise ValueError(f"no examples found for split {split}")
    anchor_ids = sorted(anchors)
    query_embeddings = np.asarray(
        model.encode(
            [example.text for example in selected],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    anchor_embeddings = np.asarray(
        model.encode(
            [anchors[anchor_id] for anchor_id in anchor_ids],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    scores = query_embeddings @ anchor_embeddings.T
    gold_indices = np.asarray(
        [anchor_ids.index(example.category) for example in selected]
    )
    probabilities = softmax(scores, axis=1)
    metrics = ranking_metrics(scores, gold_indices)
    return EvaluationResult(
        model_version=model_version,
        split=split,
        examples=len(selected),
        recall_at_1=metrics["recall_at_1"],
        recall_at_5=metrics["recall_at_5"],
        mrr=metrics["mrr"],
        ndcg_at_10=metrics["ndcg_at_10"],
        macro_f1=metrics["macro_f1"],
        ece=expected_calibration_error(probabilities, gold_indices),
    )


def calibrate_teacher(
    records: Sequence[TeacherRecord],
    gold_query_labels: dict[str, str],
) -> JudgeCalibration:
    if not records:
        raise ValueError("teacher records are required")
    accepted = [record for record in records if record.accepted]
    if not accepted:
        raise ValueError("no accepted teacher records")
    gold = np.asarray(
        [
            float(gold_query_labels[record.query_id] == record.anchor_id)
            for record in accepted
        ]
    )
    probability = np.asarray(
        [
            record.label_probabilities[3] + record.label_probabilities[4]
            for record in accepted
        ]
    )
    prediction = np.asarray([record.selected_rating >= 4 for record in accepted])
    binary_probabilities = np.column_stack([1 - probability, probability])
    return JudgeCalibration(
        rubric_id=accepted[0].rubric_id,
        records=len(records),
        accepted_rate=len(accepted) / len(records),
        exact_match_accuracy=float(np.mean(prediction == gold)),
        brier_score=float(np.mean((probability - gold) ** 2)),
        ece=expected_calibration_error(
            binary_probabilities,
            gold.astype(int),
        ),
        mean_recognized_mass=float(
            np.mean([record.recognized_mass for record in accepted])
        ),
    )


def save_result(
    path: Path,
    result: EvaluationResult | JudgeCalibration,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
