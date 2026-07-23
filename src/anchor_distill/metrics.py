from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np


def _validate_ranking_inputs(
    scores: np.ndarray, target_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    if targets.shape != (scores.shape[0],):
        raise ValueError("target_indices must have one value per row")
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("target index outside score columns")
    return scores, targets


def ranking_metrics(
    scores: np.ndarray,
    target_indices: np.ndarray,
    *,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    scores, targets = _validate_ranking_inputs(scores, target_indices)
    order = np.argsort(-scores, axis=1, kind="stable")
    target_positions = np.argmax(order == targets[:, None], axis=1) + 1
    metrics = {f"recall_at_{k}": float(np.mean(target_positions <= k)) for k in ks}
    metrics["mrr"] = float(np.mean(1.0 / target_positions))
    cutoff = min(10, scores.shape[1])
    gains = np.where(target_positions <= cutoff, 1.0 / np.log2(target_positions + 1), 0)
    metrics["ndcg_at_10"] = float(np.mean(gains))
    predictions = order[:, 0]
    metrics["macro_f1"] = macro_f1(targets, predictions, scores.shape[1])
    return metrics


def macro_f1(
    targets: np.ndarray, predictions: np.ndarray, number_of_classes: int
) -> float:
    values: list[float] = []
    for label in range(number_of_classes):
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            values.append(2 * true_positive / denominator)
    return float(np.mean(values)) if values else 0.0


def expected_calibration_error(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == targets).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > left) & (confidence <= right)
        if np.any(in_bin):
            error += float(np.mean(in_bin)) * abs(
                float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
            )
    return error


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return cast(
        np.ndarray,
        exponentiated / exponentiated.sum(axis=axis, keepdims=True),
    )


def paired_bootstrap_difference(
    metric_values_a: np.ndarray,
    metric_values_b: np.ndarray,
    *,
    replicates: int = 1000,
    seed: int = 20260722,
) -> dict[str, float]:
    a = np.asarray(metric_values_a, dtype=float)
    b = np.asarray(metric_values_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired vectors must be one-dimensional and equal length")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(a), size=len(a))
        differences[index] = float(np.mean(a[sample] - b[sample]))
    return {
        "difference": float(np.mean(a - b)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_positive": float(np.mean(differences > 0)),
    }


def normalized_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    if not len(values):
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(5))
