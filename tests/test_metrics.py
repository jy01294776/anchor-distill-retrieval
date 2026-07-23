from __future__ import annotations

import numpy as np

from anchor_distill.metrics import (
    expected_calibration_error,
    ranking_metrics,
)


def test_ranking_metrics_on_known_example() -> None:
    scores = np.asarray([[3, 2, 1], [3, 2, 1], [3, 2, 1]])
    targets = np.asarray([0, 1, 2])
    metrics = ranking_metrics(scores, targets, ks=(1, 2))
    assert metrics["recall_at_1"] == 1 / 3
    assert metrics["recall_at_2"] == 2 / 3
    assert metrics["mrr"] == (1 + 0.5 + 1 / 3) / 3
    assert 0 < metrics["ndcg_at_10"] < 1


def test_ece_is_zero_for_matching_bins() -> None:
    probabilities = np.asarray([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]])
    targets = np.asarray([0, 1, 1, 0])
    assert expected_calibration_error(probabilities, targets, bins=2) == 0.25
