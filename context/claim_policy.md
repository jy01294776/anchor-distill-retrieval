# Claim-to-evidence policy

| Claim | Required evidence |
|---|---|
| Fine-tuned a compact sentence encoder | completed training manifest and model hash |
| Performed soft-target knowledge distillation | accepted token-mass audit plus KL training manifest |
| Improved retrieval quality | held-out test metric and bootstrap interval |
| Reduced latency or cost | reproducible benchmark with hardware/model metadata |
| Built a recoverable async system | fault-injection integration tests |
| Packaged the system | successful wheel and container builds |
| Deployed the system | production URL, deployment record, and monitoring evidence |

No improvement, scalability, or production claim may be inferred from code
presence alone.

Current allowed public-project claims:

- Fine-tuned and versioned compact encoders under 1-, 4-, 16-shot and
  full-supervision public-label budgets.
- Recovered token-level teacher probability distributions and performed real
  hard-target, soft-target KL, and hybrid distillation.
- Listwise hard-target distillation improved Recall@1 over zero-shot MiniLM by
  2.82 percentage points, with exploratory paired 95% bootstrap interval
  [1.95, 3.80] percentage points.
- Soft-target KD underperformed hard-target KD by 2.01 Recall@1 percentage
  points in this experiment; the negative result is retained.
- Ran a budget-capped 1,920-pair teacher pipeline with aggregate token, dollar,
  latency, agreement, Brier, and ECE evidence.
- Measured a 54.7x parameter reduction, 24.0x local throughput increase, and
  4.6x peak-RSS reduction versus Sentence-T5-XL at a nearly equal Recall@1
  point estimate. The confidence interval is not an equivalence test.
- Built and locally tested recoverable job orchestration, including injected
  worker-crash checkpoint recovery.
- Built a reproducible wheel and source distribution.
- Verified a live PostgreSQL/Redis/Celery job round trip in GitHub Actions.
- Built the container in GitHub Actions, generated an SBOM, upgraded fixable
  HIGH dependencies, and passed the Trivy HIGH/CRITICAL gate while ignoring
  only findings with no upstream fix.

Current prohibited claims:

- Soft distillation improves quality or label efficiency.
- The hybrid run significantly improves over the 77-label gold baseline.
- The compact student is formally equivalent to Sentence-T5-XL.
- The system reduces production cost.
- The system is deployed or productized.

Interpretation requirements:

- Test comparisons are labeled exploratory because multiple systems were
  evaluated on the same public test split.
- MPS training is seeded and versioned but is not claimed to be bitwise
  deterministic across repeated runs.
- Local batch throughput is not a production-service SLA.
