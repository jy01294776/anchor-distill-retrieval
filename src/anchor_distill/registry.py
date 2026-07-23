from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anchor_distill.schemas import PromotionEvidence


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: tuple[str, ...]
    decided_at: str


def evaluate_promotion(evidence: PromotionEvidence) -> PromotionDecision:
    reasons: list[str] = []
    if evidence.latency_p95_ms > evidence.maximum_latency_p95_ms:
        reasons.append("latency_p95_exceeds_gate")
    if evidence.calibration_ece > evidence.maximum_calibration_ece:
        reasons.append("calibration_ece_exceeds_gate")
    if not evidence.lineage_complete:
        reasons.append("lineage_incomplete")
    if not evidence.security_checks_passed:
        reasons.append("security_checks_failed")
    if evidence.champion_metrics is not None:
        for metric, champion_value in evidence.champion_metrics.items():
            candidate = evidence.candidate_metrics.get(metric)
            if candidate is None:
                reasons.append(f"missing_candidate_metric:{metric}")
            elif candidate < champion_value:
                reasons.append(f"candidate_regressed:{metric}")
    return PromotionDecision(
        approved=not reasons,
        reasons=tuple(reasons),
        decided_at=datetime.now(UTC).isoformat(),
    )


def save_promotion_decision(
    destination: Path,
    evidence: PromotionEvidence,
    decision: PromotionDecision,
) -> None:
    payload: dict[str, Any] = {
        "evidence": evidence.model_dump(mode="json"),
        "decision": asdict(decision),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def log_mlflow_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)
