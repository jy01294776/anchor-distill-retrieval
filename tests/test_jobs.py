from __future__ import annotations

from pathlib import Path

import pytest

from anchor_distill.constants import JobStatus, JobType
from anchor_distill.faults import FaultInjector, WorkerCrash
from anchor_distill.jobs.models import Job
from anchor_distill.jobs.repository import (
    IdempotencyConflict,
    InvalidTransition,
    JobRepository,
)
from anchor_distill.jobs.service import (
    checkpointed_steps,
    execute_job,
    recover_lost_job,
)


@pytest.fixture
def repository(tmp_path: Path) -> JobRepository:
    repo = JobRepository(f"sqlite:///{tmp_path / 'jobs.db'}")
    repo.create_schema()
    return repo


def test_idempotent_create_returns_same_job(repository: JobRepository) -> None:
    first, first_created = repository.create_or_get(
        job_type=JobType.EVALUATE_MODEL,
        payload={"model": "candidate"},
        idempotency_key="evaluation-001",
    )
    second, second_created = repository.create_or_get(
        job_type=JobType.EVALUATE_MODEL,
        payload={"model": "candidate"},
        idempotency_key="evaluation-001",
    )
    assert first_created
    assert not second_created
    assert first.id == second.id


def test_idempotency_conflict_is_detected(repository: JobRepository) -> None:
    repository.create_or_get(
        job_type=JobType.BUILD_INDEX,
        payload={"model": "a"},
        idempotency_key="build-index-001",
    )
    with pytest.raises(IdempotencyConflict):
        repository.create_or_get(
            job_type=JobType.BUILD_INDEX,
            payload={"model": "b"},
            idempotency_key="build-index-001",
        )


def test_checkpoint_survives_failure_and_resume(repository: JobRepository) -> None:
    job, _ = repository.create_or_get(
        job_type=JobType.TRAIN_STUDENT,
        payload={},
        idempotency_key="train-resume-001",
    )
    repository.transition(job.id, JobStatus.RUNNING)
    repository.save_checkpoint(job.id, {"completed_steps": 2})
    repository.transition(job.id, JobStatus.FAILED, error_code="worker_crash")
    repository.transition(job.id, JobStatus.QUEUED)
    resumed = repository.transition(job.id, JobStatus.RUNNING)
    assert resumed.attempt == 2
    assert resumed.checkpoint == {"completed_steps": 2}


def test_invalid_transition_fails(repository: JobRepository) -> None:
    job, _ = repository.create_or_get(
        job_type=JobType.BUILD_INDEX,
        payload={},
        idempotency_key="invalid-transition-001",
    )
    with pytest.raises(InvalidTransition):
        repository.transition(job.id, JobStatus.SUCCEEDED)


def test_worker_crash_requeues_and_resumes_from_checkpoint(
    repository: JobRepository,
) -> None:
    job, _ = repository.create_or_get(
        job_type=JobType.BUILD_INDEX,
        payload={},
        idempotency_key="worker-crash-resume-001",
    )
    calls: list[str] = []
    injector = FaultInjector(["worker_crash"])

    def crashing_handler(current: Job, repo: JobRepository) -> dict[str, int]:
        del current
        return checkpointed_steps(
            job,
            repo,
            [
                lambda: calls.append("step-1"),
                injector.trigger,
                lambda: calls.append("step-3"),
            ],
        )

    with pytest.raises(WorkerCrash):
        execute_job(
            repository,
            job.id,
            {JobType.BUILD_INDEX: crashing_handler},
        )
    crashed = repository.get(job.id)
    assert crashed is not None
    assert crashed.checkpoint == {"completed_steps": 1}
    recover_lost_job(repository, job.id)

    def resumed_handler(current: Job, repo: JobRepository) -> dict[str, int]:
        del current
        resumed = repository.get(job.id)
        assert resumed is not None
        return checkpointed_steps(
            resumed,
            repo,
            [
                lambda: calls.append("step-1-repeated"),
                lambda: calls.append("step-2-retried"),
                lambda: calls.append("step-3"),
            ],
        )

    completed = execute_job(
        repository,
        job.id,
        {JobType.BUILD_INDEX: resumed_handler},
    )
    assert completed.status == JobStatus.SUCCEEDED
    assert calls == ["step-1", "step-2-retried", "step-3"]
