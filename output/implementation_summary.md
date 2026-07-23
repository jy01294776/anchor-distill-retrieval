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

Validated experiment evidence now includes:

- 1/4/16-shot and full-supervision MiniLM baselines;
- a 1,920-pair, cost-capped public teacher dataset with separate training and
  calibration queries;
- ordinal and listwise hard, soft, and hybrid distillation runs, including
  retained negative results;
- paired bootstrap comparisons and an explicit multiple-test-look limitation;
- a Sentence-T5-XL reference and local parameter/throughput/RSS comparison;
- a quality/performance/cost frontier with observed and projected teacher cost;
- a 10,003-record train-only evidence index and low-confidence retrieval demo
  with typed public-example citations and deterministic explanations;
- 26 passing local tests, strict mypy, Ruff, a safe 190-file isolation scan,
  and successful wheel/sdist builds.

The public repository is
`https://github.com/jy01294776/anchor-distill-retrieval`; draft PR #1 contains
the completed experiment matrix. GitHub Actions verifies tests, packaging, a
live PostgreSQL/Redis/Celery job round trip, container construction, SBOM
generation, and the fixable HIGH/CRITICAL vulnerability gate.

The doctoral repository remains frozen and unmodified. The public project has
no dependency on its data, source, outputs, constructs, or model artifacts.
