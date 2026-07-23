from __future__ import annotations

from celery import Celery

from anchor_distill.config import Settings
from anchor_distill.jobs.handlers import build_handlers
from anchor_distill.jobs.repository import JobRepository
from anchor_distill.jobs.service import execute_job

settings = Settings()
celery_app = Celery(
    "anchor_distill",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def run_job(self: object, job_id: str) -> dict[str, str]:
    repository = JobRepository(settings.database_url)
    repository.create_schema()
    result = execute_job(repository, job_id, handlers=build_handlers(settings))
    return {"id": result.id, "status": str(result.status)}
