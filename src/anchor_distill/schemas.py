from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anchor_distill.constants import JobStatus, JobType


class JobCreate(BaseModel):
    job_type: JobType
    payload: dict[str, Any] = Field(default_factory=dict)


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_type: JobType
    status: JobStatus
    idempotency_key: str
    request_hash: str
    payload: dict[str, Any]
    attempt: int
    checkpoint: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class JobEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    from_status: JobStatus | None
    to_status: JobStatus
    message: str | None
    created_at: datetime


class TokenAlternative(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenPosition(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TokenAlternative]


class TeacherRecord(BaseModel):
    query_id: str
    anchor_id: str
    rubric_id: str
    prompt_hash: str
    model_requested: str
    model_resolved: str
    system_fingerprint: str | None = None
    selected_rating: int = Field(ge=1, le=5)
    label_probabilities: tuple[float, float, float, float, float]
    recognized_mass: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0)
    accepted: bool
    refusal: bool = False
    request_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    cached_prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)

    @field_validator("label_probabilities")
    @classmethod
    def validate_distribution(
        cls, values: tuple[float, float, float, float, float]
    ) -> tuple[float, float, float, float, float]:
        if any(value < 0 for value in values):
            raise ValueError("probabilities must be nonnegative")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        return values


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalHit(BaseModel):
    anchor_id: str
    label: str
    score: float
    evidence: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    query: str
    model_version: str
    hits: list[RetrievalHit]
    needs_review: bool
    confidence_margin: float


class PromotionEvidence(BaseModel):
    model_uri: str
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None = None
    latency_p95_ms: float
    maximum_latency_p95_ms: float
    calibration_ece: float
    maximum_calibration_ece: float
    lineage_complete: bool
    security_checks_passed: bool
