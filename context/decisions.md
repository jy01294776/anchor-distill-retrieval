# Decisions

## 2026-07-22

- The project is a clean-room public engineering extension motivated by
  doctoral work; it has no research-data or source-code dependency.
- BANKING77 is the sole v1 benchmark.
- The official test split is evaluation-only.
- Existing research judge outputs are not training data.
- Soft distillation requires recovered token-level probability mass; a hard
  score alone is not described as soft distillation.
- Job execution is at-least-once with idempotent checkpoints.
- `Built` and `Packaged` are permitted after validation; `Deployed` and
  `Productized` require a verified real deployment.
- A synthetic compatibility probe rejected `gpt-5-mini` because this account
  cannot request its token log-probabilities. `gpt-4o-mini` passed Structured
  Outputs plus top-logprob recovery and resolved to `gpt-4o-mini-2024-07-18`;
  it is therefore the v1 public teacher. This decision concerns only public,
  synthetic/BANKING77 work and does not alter the doctoral pipeline.
- The first evidence-producing training run uses one official BANKING77
  training example per intent. The official test split remains evaluation-only.
- A 10-pair teacher pilot is treated as an API/data-contract gate, not as
  evidence that soft distillation works.
- Local resume DOCX/PDF files are git-ignored because they contain contact
  information. Only claim policies and aggregate public experiment evidence are
  eligible for a public repository.
- The container and PostgreSQL/Redis integration remain unverified because no
  Docker CLI is installed. CI definitions do not count as a successful build.
