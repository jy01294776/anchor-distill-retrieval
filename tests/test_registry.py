from __future__ import annotations

from anchor_distill.registry import evaluate_promotion
from anchor_distill.schemas import PromotionEvidence


def evidence(**overrides: object) -> PromotionEvidence:
    values: dict[str, object] = {
        "model_uri": "models:/candidate/1",
        "candidate_metrics": {"mrr": 0.82},
        "champion_metrics": {"mrr": 0.80},
        "latency_p95_ms": 40,
        "maximum_latency_p95_ms": 50,
        "calibration_ece": 0.04,
        "maximum_calibration_ece": 0.05,
        "lineage_complete": True,
        "security_checks_passed": True,
    }
    values.update(overrides)
    return PromotionEvidence(**values)


def test_promotion_requires_all_gates() -> None:
    assert evaluate_promotion(evidence()).approved
    rejected = evaluate_promotion(
        evidence(latency_p95_ms=60, security_checks_passed=False)
    )
    assert not rejected.approved
    assert "latency_p95_exceeds_gate" in rejected.reasons
    assert "security_checks_failed" in rejected.reasons
