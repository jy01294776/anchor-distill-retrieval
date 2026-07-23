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
Official test examples are never used for teacher labeling, training, prompt
selection, or model promotion.

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

The current evidence package is intentionally small but real:

- MiniLM zero-shot on 3,080 held-out BANKING77 examples: Recall@1 0.6237,
  Recall@5 0.8481, and MRR 0.7251.
- MiniLM fine-tuned on one public example per intent (77 labels): Recall@1
  0.6594, Recall@5 0.8740, and MRR 0.7573.
- Paired 2,000-replicate bootstrap: Recall@1 difference +0.0357 with a 95%
  interval of [0.0276, 0.0438].
- The public teacher compatibility probe selected `gpt-4o-mini`; a 10-pair
  pilot passed the token-mass gate for all records. This pilot does not support
  a general teacher-quality or distillation-effect claim.
- Local Apple-silicon/MPS batch benchmark (256 texts, five repeats): 2,591
  texts/second and 157.5 ms p95. This is a local benchmark, not a service SLA.

See `artifacts/benchmarks/` and `artifacts/reports/`. The full 1/4/16-shot,
hard/soft/hybrid distillation, full-supervision, and large-encoder experiment
matrix remains pending.

## Evidence policy

- `artifacts/` contains validated reports and benchmarks.
- `context/claim_policy.md` maps every allowed resume claim to evidence.
- Failed or null experiments remain visible.
- `Productized` and `Deployed` are prohibited until a real deployment is
  verified.
- Resume files under `artifacts/resume/` are local-only and git-ignored because
  they contain personal contact information.

## License and attribution

Project code is Apache-2.0. BANKING77 remains under CC BY 4.0; see
`THIRD_PARTY_NOTICES.md`.
