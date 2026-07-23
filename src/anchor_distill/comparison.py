from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anchor_distill.data import BankingExample
from anchor_distill.metrics import paired_bootstrap_difference


@dataclass(frozen=True)
class ModelComparison:
    baseline_model: str
    candidate_model: str
    split: str
    examples: int
    recall_at_1: dict[str, float]
    reciprocal_rank: dict[str, float]


def _outcomes(
    model: Any,
    examples: Sequence[BankingExample],
    anchors: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    anchor_ids = sorted(anchors)
    query_embeddings = np.asarray(
        model.encode(
            [example.text for example in examples],
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
    order = np.argsort(-scores, axis=1, kind="stable")
    targets = np.asarray([anchor_ids.index(example.category) for example in examples])
    ranks = np.argmax(order == targets[:, None], axis=1) + 1
    return (ranks == 1).astype(float), 1.0 / ranks


def compare_models(
    baseline: Any,
    candidate: Any,
    examples: Sequence[BankingExample],
    anchors: dict[str, str],
    *,
    baseline_name: str,
    candidate_name: str,
    split: str = "test",
    replicates: int = 1000,
    seed: int = 20260722,
) -> ModelComparison:
    selected = [example for example in examples if example.split == split]
    baseline_correct, baseline_rr = _outcomes(baseline, selected, anchors)
    candidate_correct, candidate_rr = _outcomes(candidate, selected, anchors)
    return ModelComparison(
        baseline_model=baseline_name,
        candidate_model=candidate_name,
        split=split,
        examples=len(selected),
        recall_at_1=paired_bootstrap_difference(
            candidate_correct,
            baseline_correct,
            replicates=replicates,
            seed=seed,
        ),
        reciprocal_rank=paired_bootstrap_difference(
            candidate_rr,
            baseline_rr,
            replicates=replicates,
            seed=seed + 1,
        ),
    )


def save_comparison(path: Path, comparison: ModelComparison) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(comparison), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)
