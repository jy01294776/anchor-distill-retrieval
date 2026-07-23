from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

job_context: ContextVar[str | None] = ContextVar("job_id", default=None)

JOB_TRANSITIONS = Counter(
    "anchor_distill_job_transitions_total",
    "Job state transitions.",
    ("job_type", "from_status", "to_status"),
)
JOB_DURATION = Histogram(
    "anchor_distill_job_duration_seconds",
    "Wall-clock duration of job execution.",
    ("job_type", "outcome"),
)
ACTIVE_JOBS = Gauge(
    "anchor_distill_active_jobs",
    "Currently running jobs.",
    ("job_type",),
)
RETRY_TOTAL = Counter(
    "anchor_distill_retries_total",
    "Retryable failures observed by operation and status.",
    ("operation", "reason"),
)
TEACHER_REQUESTS = Counter(
    "anchor_distill_teacher_requests_total",
    "Teacher request outcomes.",
    ("model", "outcome"),
)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_job(job_id: str) -> None:
    job_context.set(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)


def clear_job() -> None:
    job_context.set(None)
    structlog.contextvars.clear_contextvars()


@contextmanager
def measure_job(job_type: str) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    state: dict[str, Any] = {"outcome": "failed"}
    ACTIVE_JOBS.labels(job_type).inc()
    try:
        yield state
    finally:
        ACTIVE_JOBS.labels(job_type).dec()
        JOB_DURATION.labels(job_type, str(state["outcome"])).observe(
            time.perf_counter() - started
        )
