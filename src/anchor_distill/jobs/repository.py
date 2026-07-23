from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from anchor_distill.constants import ALLOWED_TRANSITIONS, JobStatus, JobType
from anchor_distill.jobs.models import Base, Job, JobEvent
from anchor_distill.observability import JOB_TRANSITIONS


class IdempotencyConflict(ValueError):
    pass


class InvalidTransition(ValueError):
    pass


def canonical_request_hash(job_type: JobType, payload: dict[str, Any]) -> str:
    material = {
        "job_type": job_type.value,
        "payload": payload,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class JobRepository:
    def __init__(self, database_url: str) -> None:
        connect_args = (
            {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        )
        self.engine: Engine = create_engine(database_url, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.sessions() as session:
            yield session

    def create_or_get(
        self,
        *,
        job_type: JobType,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[Job, bool]:
        request_hash = canonical_request_hash(job_type, payload)
        with self.session() as session:
            existing = session.scalar(
                select(Job).where(Job.idempotency_key == idempotency_key)
            )
            if existing:
                if existing.request_hash != request_hash:
                    raise IdempotencyConflict(
                        "idempotency key was already used for a different request"
                    )
                return existing, False
            job = Job(
                id=str(uuid.uuid4()),
                job_type=job_type,
                status=JobStatus.QUEUED,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                payload=payload,
            )
            session.add(job)
            session.add(
                JobEvent(
                    job_id=job.id,
                    from_status=None,
                    to_status=JobStatus.QUEUED,
                    message="created",
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.scalar(
                    select(Job).where(Job.idempotency_key == idempotency_key)
                )
                if existing and existing.request_hash == request_hash:
                    return existing, False
                raise IdempotencyConflict(
                    "concurrent idempotency-key conflict"
                ) from None
            return job, True

    def get(self, job_id: str) -> Job | None:
        with self.session() as session:
            return session.get(Job, job_id)

    def events(self, job_id: str) -> list[JobEvent]:
        with self.session() as session:
            return list(
                session.scalars(
                    select(JobEvent)
                    .where(JobEvent.job_id == job_id)
                    .order_by(JobEvent.id)
                )
            )

    def transition(
        self,
        job_id: str,
        target: JobStatus,
        *,
        message: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> Job:
        with self.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            previous = JobStatus(job.status)
            if target not in ALLOWED_TRANSITIONS[previous]:
                raise InvalidTransition(f"{previous} -> {target} is not allowed")
            job.status = target
            if target == JobStatus.RUNNING:
                job.attempt += 1
            job.error_code = error_code
            job.error_message = error_message
            session.add(
                JobEvent(
                    job_id=job.id,
                    from_status=previous,
                    to_status=target,
                    message=message,
                )
            )
            session.commit()
            JOB_TRANSITIONS.labels(
                JobType(job.job_type).value, previous.value, target.value
            ).inc()
            return job

    def save_checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> Job:
        with self.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            if JobStatus(job.status) != JobStatus.RUNNING:
                raise InvalidTransition("checkpoints require a running job")
            job.checkpoint = checkpoint
            session.commit()
            return job
