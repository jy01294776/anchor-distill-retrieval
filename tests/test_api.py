from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from anchor_distill.api import create_app
from anchor_distill.config import Settings
from anchor_distill.jobs.repository import JobRepository
from anchor_distill.retrieval import NumpyIndex


class FakeEncoder:
    def encode(self, _texts: list[str], **_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


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


def test_retrieve_returns_training_evidence_and_review_explanation(
    tmp_path: Path,
) -> None:
    repository = JobRepository(f"sqlite:///{tmp_path / 'jobs.db'}")
    index = NumpyIndex(
        anchor_ids=["card_arrival", "cash_withdrawal"],
        labels=["card arrival", "cash withdrawal"],
        embeddings=np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32),
        model_version="test-model",
        evidence_ids=["train-00001", "train-00002"],
        evidence_texts=["where is my card", "cash did not arrive"],
        evidence_anchor_ids=["card_arrival", "cash_withdrawal"],
        evidence_embeddings=np.asarray(
            [[1.0, 0.0], [0.99, 0.01]],
            dtype=np.float32,
        ),
    )
    client = TestClient(
        create_app(
            settings=Settings(project_root=tmp_path),
            repository=repository,
            enqueue=lambda _job_id: None,
            encoder=FakeEncoder(),
            index=index,
        )
    )

    response = client.post(
        "/v1/retrieve",
        json={"query": "where is it", "top_k": 2, "evidence_per_hit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["needs_review"]
    assert payload["evidence_split"] == "train"
    assert payload["hits"][0]["evidence"][0] == {
        "citation_id": "BANKING77:train-00001",
        "source": "BANKING77 train",
        "example_id": "train-00001",
        "anchor_id": "card_arrival",
        "text": "where is my card",
        "score": 1.0,
    }
    assert "[BANKING77:train-00001]" in payload["explanation"]
    assert "Review required" in payload["explanation"]
