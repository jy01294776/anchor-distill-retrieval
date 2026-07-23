from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from anchor_distill.api import create_app
from anchor_distill.config import Settings
from anchor_distill.jobs.repository import JobRepository


def test_job_api_idempotency_and_events(tmp_path: Path) -> None:
    repository = JobRepository(f"sqlite:///{tmp_path / 'jobs.db'}")
    app = create_app(
        settings=Settings(project_root=tmp_path),
        repository=repository,
        enqueue=lambda _job_id: None,
    )
    client = TestClient(app)
    payload = {"job_type": "evaluate_model", "payload": {"model": "candidate"}}
    headers = {"Idempotency-Key": "evaluation-api-001"}
    first = client.post("/v1/jobs", json=payload, headers=headers)
    second = client.post("/v1/jobs", json=payload, headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    events = client.get(f"/v1/jobs/{first.json()['id']}/events")
    assert events.status_code == 200
    assert events.json()[0]["to_status"] == "queued"


def test_idempotency_key_reuse_with_different_request_is_409(
    tmp_path: Path,
) -> None:
    repository = JobRepository(f"sqlite:///{tmp_path / 'jobs.db'}")
    client = TestClient(
        create_app(
            settings=Settings(project_root=tmp_path),
            repository=repository,
            enqueue=lambda _job_id: None,
        )
    )
    headers = {"Idempotency-Key": "same-key-different-request"}
    assert (
        client.post(
            "/v1/jobs",
            json={"job_type": "build_index", "payload": {"model": "a"}},
            headers=headers,
        ).status_code
        == 202
    )
    conflict = client.post(
        "/v1/jobs",
        json={"job_type": "build_index", "payload": {"model": "b"}},
        headers=headers,
    )
    assert conflict.status_code == 409
