from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from anchor_distill.config import Settings
from anchor_distill.jobs.repository import (
    IdempotencyConflict,
    InvalidTransition,
    JobRepository,
)
from anchor_distill.jobs.service import cancel_job, resume_job
from anchor_distill.retrieval import NumpyIndex, retrieve_with_evidence
from anchor_distill.schemas import (
    JobCreate,
    JobEventRead,
    JobRead,
    RetrieveRequest,
    RetrieveResponse,
)


def create_app(
    *,
    settings: Settings | None = None,
    repository: JobRepository | None = None,
    enqueue: Callable[[str], None] | None = None,
    encoder: Any | None = None,
    index: NumpyIndex | None = None,
) -> FastAPI:
    active = settings or Settings()
    jobs = repository or JobRepository(active.database_url)
    jobs.create_schema()
    if enqueue is None:

        def enqueue_default(job_id: str) -> None:
            from anchor_distill.jobs.tasks import run_job

            run_job.delay(job_id)

        enqueue_job: Callable[[str], None] = enqueue_default
    else:
        enqueue_job = enqueue

    app = FastAPI(title="Anchor Distill Retrieval", version="0.1.0")
    app.state.settings = active
    app.state.jobs = jobs
    app.state.encoder = encoder
    app.state.index = index

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/jobs",
        response_model=JobRead,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_job(
        request: JobCreate,
        idempotency_key: Annotated[str, Header(min_length=8, max_length=255)],
    ) -> Any:
        try:
            job, created = jobs.create_or_get(
                job_type=request.job_type,
                payload=request.payload,
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if created:
            enqueue_job(job.id)
        return job

    @app.get("/v1/jobs/{job_id}", response_model=JobRead)
    def get_job(job_id: str) -> Any:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobRead)
    def cancel(job_id: str) -> Any:
        try:
            return cancel_job(jobs, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except InvalidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/resume", response_model=JobRead)
    def resume(job_id: str) -> Any:
        try:
            job = resume_job(jobs, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except InvalidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        enqueue_job(job.id)
        return job

    @app.get("/v1/jobs/{job_id}/events", response_model=list[JobEventRead])
    def events(job_id: str) -> Any:
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return jobs.events(job_id)

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        active_encoder = app.state.encoder
        active_index = app.state.index
        if active_encoder is None or active_index is None:
            raise HTTPException(status_code=503, detail="retrieval model unavailable")
        embedding = np.asarray(
            active_encoder.encode(
                [request.query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )[0]
        result = retrieve_with_evidence(
            active_index,
            embedding,
            query=request.query,
            top_k=request.top_k,
            evidence_per_hit=request.evidence_per_hit,
        )
        return RetrieveResponse.model_validate(result)

    return app


app = create_app()
