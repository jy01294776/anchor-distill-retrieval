from __future__ import annotations

import os
import uuid

import pytest

from anchor_distill.config import Settings
from anchor_distill.constants import JobStatus, JobType
from anchor_distill.jobs.repository import JobRepository


@pytest.mark.systems_e2e
def test_postgres_redis_celery_job_round_trip() -> None:
    if os.environ.get("RUN_SYSTEM_E2E") != "1":
        pytest.skip("set RUN_SYSTEM_E2E=1 with live system dependencies")

    from anchor_distill.jobs.tasks import run_job

    settings = Settings()
    source = settings.path("data", "interim", "systems_e2e_input.jsonl")
    output = settings.path("data", "processed", "systems_e2e_output.jsonl")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("")

    repository = JobRepository(settings.database_url)
    repository.create_schema()
    job, created = repository.create_or_get(
        job_type=JobType.BUILD_TEACHER_DATASET,
        payload={
            "input": str(source.relative_to(settings.project_root)),
            "output": str(output.relative_to(settings.project_root)),
        },
        idempotency_key=f"systems-e2e-{uuid.uuid4()}",
    )
    assert created

    result = run_job.delay(job.id).get(timeout=60)

    completed = repository.get(job.id)
    assert result == {"id": job.id, "status": "succeeded"}
    assert completed is not None
    assert completed.status == JobStatus.SUCCEEDED
    assert completed.checkpoint == {
        "input_records": 0,
        "accepted_records": 0,
        "output": "data/processed/systems_e2e_output.jsonl",
    }
    assert [event.to_status for event in repository.events(job.id)] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
    ]
