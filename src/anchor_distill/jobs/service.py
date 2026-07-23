from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from anchor_distill.constants import JobStatus, JobType
from anchor_distill.faults import RetryableFailure
from anchor_distill.jobs.models import Job
from anchor_distill.jobs.repository import JobRepository
from anchor_distill.observability import bind_job, clear_job, measure_job

JobHandler = Callable[[Job, JobRepository], dict[str, Any]]


def checkpointed_steps(
    job: Job,
    repository: JobRepository,
    steps: list[Callable[[], None]],
) -> dict[str, Any]:
    completed = int((job.checkpoint or {}).get("completed_steps", 0))
    for index, step in enumerate(steps[completed:], start=completed):
        step()
        repository.save_checkpoint(job.id, {"completed_steps": index + 1})
    return {"completed_steps": len(steps)}


def execute_job(
    repository: JobRepository,
    job_id: str,
    handlers: dict[JobType, JobHandler],
) -> Job:
    logger = structlog.get_logger()
    job = repository.get(job_id)
    if job is None:
        raise KeyError(job_id)
    if JobStatus(job.status) != JobStatus.QUEUED:
        return job
    bind_job(job_id)
    running = repository.transition(job_id, JobStatus.RUNNING, message="worker_start")
    with measure_job(JobType(running.job_type).value) as measurement:
        try:
            handler = handlers[JobType(running.job_type)]
            checkpoint = handler(running, repository)
            repository.save_checkpoint(job_id, checkpoint)
            completed = repository.transition(
                job_id, JobStatus.SUCCEEDED, message="artifact_commit_complete"
            )
            measurement["outcome"] = "succeeded"
            logger.info("job_succeeded", job_type=completed.job_type)
            return completed
        except RetryableFailure as exc:
            failed = repository.transition(
                job_id,
                JobStatus.FAILED,
                message="retryable_failure",
                error_code=exc.reason,
                error_message=str(exc),
            )
            measurement["outcome"] = "retryable_failure"
            logger.warning("job_failed", reason=exc.reason)
            return failed
        except Exception as exc:
            failed = repository.transition(
                job_id,
                JobStatus.FAILED,
                message="unhandled_failure",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            measurement["outcome"] = "failed"
            logger.exception("job_failed")
            return failed
        finally:
            clear_job()


def cancel_job(repository: JobRepository, job_id: str) -> Job:
    job = repository.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return repository.transition(job_id, JobStatus.CANCELLED, message="user_cancel")


def resume_job(repository: JobRepository, job_id: str) -> Job:
    job = repository.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return repository.transition(job_id, JobStatus.QUEUED, message="user_resume")


def recover_lost_job(repository: JobRepository, job_id: str) -> Job:
    job = repository.get(job_id)
    if job is None:
        raise KeyError(job_id)
    if JobStatus(job.status) != JobStatus.RUNNING:
        raise ValueError("only a running job can be recovered as worker-lost")
    repository.transition(
        job_id,
        JobStatus.FAILED,
        message="worker_lost",
        error_code="worker_crash",
        error_message="worker process exited before terminal state commit",
    )
    return repository.transition(
        job_id,
        JobStatus.QUEUED,
        message="automatic_worker_lost_requeue",
    )
