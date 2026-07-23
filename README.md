# Anchor Distill Retrieval

An evidence-first reference implementation for cost-aware soft-target
distillation in fine-grained semantic retrieval.

> Motivated by doctoral research on scalable text measurement and LLM
> evaluation. This clean-room implementation contains no research data,
> proprietary constructs, manuscript results, or submission code.

## Why this exists

An LLM can grade ambiguous query–anchor pairs, but it is too expensive and
operationally fragile to sit on every retrieval request. This project tests
whether token-level teacher uncertainty can be distilled into a compact
sentence encoder under limited human-label budgets.

The project asks:

1. Do soft teacher targets outperform the same teacher's hard labels?
2. How many gold labels are needed to approach full-supervision quality?
3. What quality, calibration, latency, memory, and cost trade-offs result?
4. Does the labeling/training system recover correctly from partial failure?

No result is pre-claimed. Promotion and resume claims are generated only from
validated experiment artifacts.

## Public benchmark

The benchmark uses BANKING77, a CC BY 4.0 dataset with 77 fine-grained intents.
Official test examples are never used for teacher labeling or training.
Because several candidate systems have now been evaluated on the same test
split, comparisons are reported as exploratory rather than as a single-look
confirmatory result.

Compared systems:

- MiniLM zero-shot
- 1/4/16-shot gold fine-tuning
- hard-label teacher distillation
- soft-target KL distillation
- gold + soft-target hybrid
- full-supervised reference
- Sentence-T5-XL zero-shot reference

## System

Long-running operations are asynchronous jobs:

```text
teacher_label
build_teacher_dataset
train_student
evaluate_model
build_index
```

```text
POST /v1/jobs
GET  /v1/jobs/{id}
POST /v1/jobs/{id}/cancel
POST /v1/jobs/{id}/resume
GET  /v1/jobs/{id}/events
POST /v1/retrieve
```

FastAPI serves the API, PostgreSQL stores job state and idempotency keys,
Redis/Celery executes at-least-once tasks, and idempotent checkpoints prevent
duplicate artifact commits. MLflow records model lineage and promotion gates.
GitHub Actions verifies a live PostgreSQL/Redis/Celery round trip, builds the
wheel and container image, generates an SBOM, and gates the image on fixable
HIGH/CRITICAL Trivy findings.

## Quick start

```bash
uv sync --extra ml --extra systems --group dev
uv run anchor-distill isolation check
uv run anchor-distill data prepare
uv run pytest
docker compose up --build
```

Run the synthetic compatibility probe before any teacher labeling:

```bash
uv run anchor-distill teacher probe
```

The probe checks Structured Outputs, token log-probabilities, five-label mass
coverage, refusal handling, and model snapshot metadata. It never sends
benchmark or research data.

## Validated public results

All rows below use the 3,080-example BANKING77 test split.

| System | Human labels | Teacher pairs | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| MiniLM zero-shot | 0 | 0 | 0.6237 | 0.8481 | 0.7251 |
| MiniLM gold 1-shot | 77 | 0 | 0.6594 | 0.8740 | 0.7573 |
| MiniLM gold 4-shot | 308 | 0 | 0.7075 | 0.9214 | 0.8001 |
| MiniLM gold 16-shot | 1,232 | 0 | 0.8269 | 0.9662 | 0.8885 |
| MiniLM gold full | 10,003 | 0 | 0.9192 | 0.9903 | 0.9504 |
| MiniLM listwise hard KD | 0 | 1,280 | 0.6519 | 0.8705 | 0.7509 |
| MiniLM listwise soft KD | 0 | 1,280 | 0.6318 | 0.8438 | 0.7272 |
| MiniLM listwise hybrid | 77 | 1,280 | 0.6623 | 0.8724 | 0.7582 |
| Sentence-T5-XL zero-shot | 0 | 0 | 0.6516 | 0.8633 | 0.7497 |

The listwise hard student improved Recall@1 over zero-shot MiniLM by 0.0282
(paired 95% bootstrap interval [0.0195, 0.0380]). Soft-target KD was 0.0201
below hard-target KD ([-0.0302, -0.0110]); the experiment therefore does not
support a claim that soft targets improve retrieval in this setting. Adding
the teacher signal to the 77-label run changed Recall@1 by only 0.0029
([-0.0032, 0.0091]).

The public teacher run produced 1,920 accepted query-anchor judgments over 192
training/calibration queries. It cost $0.0473 at the recorded token mix, with
pair accuracy 0.8891, Brier score 0.0925, and ECE 0.0842 on the 64-query
calibration partition.

The 22.7M-parameter listwise-hard student and 1.24B-parameter Sentence-T5-XL
had nearly identical Recall@1 point estimates (difference 0.0003; interval
[-0.0146, 0.0146]). On the same local Apple-silicon/MPS benchmark, the student
used 54.7x fewer parameters, measured 24.0x higher throughput, and used 4.6x
less peak RSS. This interval is not a formal equivalence test, and local
throughput is not a service SLA.

See `artifacts/benchmarks/` and
`artifacts/reports/quality_cost_frontier.md` for machine-readable evidence and
interpretation limits.

## Evidence policy

- `artifacts/` contains validated reports and benchmarks.
- `context/claim_policy.md` maps every allowed resume claim to evidence.
- Failed or null experiments remain visible.
- CI evidence distinguishes fixed vulnerabilities from upstream-unfixed CVEs;
  no individual CVE is allowlisted.
- `Productized` and `Deployed` are prohibited until a real deployment is
  verified.
- Resume files under `artifacts/resume/` are local-only and git-ignored because
  they contain personal contact information.

## License and attribution

Project code is Apache-2.0. BANKING77 remains under CC BY 4.0; see
`THIRD_PARTY_NOTICES.md`.
