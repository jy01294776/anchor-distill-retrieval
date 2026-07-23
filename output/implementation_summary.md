# Implementation summary

The clean-room repository now contains:

- public BANKING77 ingestion with counts, licensing, and SHA-256 provenance;
- Structured Outputs plus token-logprob recovery and calibration gates;
- gold, hard-target, soft-target KL, and hybrid student-training paths;
- retrieval evaluation and paired bootstrap comparison;
- an asynchronous job API with persistent state, idempotency, cancel/resume,
  checkpoints, and event history;
- injected 429, 5xx, timeout, and worker-crash failure modes;
- JSON logs, Prometheus metrics, MLflow promotion gates, and local performance
  benchmarking;
- FastAPI, PostgreSQL, Redis/Celery, Docker Compose, GitHub Actions, wheel and
  image build definitions, SBOM generation, and security scanning;
- clean-room, secret, path, and thesis-term scanning.

The doctoral repository remains frozen and unmodified. The public project has
no dependency on its data, source, outputs, constructs, or model artifacts.
