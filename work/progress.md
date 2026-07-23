# Progress

## 2026-07-22

- Created and verified the clean-room project structure and isolation contract.
- Downloaded official BANKING77 files: 10,003 train, 3,080 test, 77 intents;
  recorded source URLs, license, and SHA-256 hashes.
- Verified an isolated Python 3.12.12 / torch 2.13.0 arm64 environment with MPS
  built and available.
- Ran the synthetic teacher compatibility gate. `gpt-5-mini` rejected
  logprobs; `gpt-4o-mini-2024-07-18` passed Structured Outputs and five-label
  token-mass recovery.
- Ran a public 10-pair teacher pilot: 10/10 records accepted, exact-match 0.90,
  Brier 0.0995, and ECE 0.0998. This is compatibility evidence only.
- Established the MiniLM zero-shot baseline on the untouched test split.
- Fine-tuned MiniLM on 77 public labels and evaluated it on the untouched test
  split. Recall@1 increased from 0.6237 to 0.6594; the paired 2,000-replicate
  bootstrap interval for the +0.0357 difference was [0.0276, 0.0438].
- Benchmarked local MPS batch encoding at 2,591 texts/second with 157.5 ms p95
  for batches of 256 over five repeats.
- Implemented five concrete Celery job handlers, persistent job state,
  idempotency, cancellation/resume, checkpoint recovery, event history,
  fault injection, metrics/logging, MLflow gates, and secure Docker context.
- Validation passes: 17 tests, Ruff, and strict mypy across 25 source files.
- Generated and visually verified Applied Scientist and MLE resume variants in
  DOCX and PDF; the original resume was not modified.

## Still pending

- Run 4-shot, 16-shot, full-supervision, hard-target, soft-target KL, hybrid,
  and large-encoder comparisons.
- Run PostgreSQL/Redis/Celery end-to-end integration and build the container;
  Docker CLI is unavailable on this machine.
- Complete cost accounting and a larger teacher-calibration gold set before
  making distillation-effect or cost-savings claims.
- Publish to GitHub after authentication is renewed.

## 2026-07-23

- Created and pushed the public GitHub repository:
  `https://github.com/jy01294776/anchor-distill-retrieval`.
- Completed 4-shot, 16-shot, and full-supervision MiniLM baselines plus the
  Sentence-T5-XL zero-shot reference.
- Completed a budget-capped public teacher run: 1,920/1,920 accepted pairs,
  269,422 prompt tokens, 11,520 completion tokens, and $0.0473 observed cost.
- Completed ordinal and listwise hard, soft, and hybrid distillation. The
  ordinal pairwise objective failed; listwise hard KD improved Recall@1 by
  0.0282 over zero-shot MiniLM. Soft KD was 0.0201 below hard KD, and the
  hybrid-vs-1-shot interval crossed zero.
- Generated the quality/performance/cost frontier and measured the 22.7M
  student against the 1.24B Sentence-T5-XL reference.
- Added an actual PostgreSQL/Redis/Celery job round-trip test to GitHub Actions
  and changed container scanning to inspect the locally loaded image.
- Removed three machine-specific absolute paths from the public teacher
  summary and added a regression test for relative artifact paths.
- Local validation passes: Ruff format/check, strict mypy over 26 source files,
  28 tests passed with one live-systems test skipped, isolation scan safe over
  192 files, and wheel/sdist build succeeded.

## Remote validation

- Pushed `agent/complete-experiment-matrix` and opened draft PR #1.
- GitHub Actions passed strict tests, package build, the live
  PostgreSQL/Redis/Celery system round trip, container build, SBOM generation,
  and the Trivy HIGH/CRITICAL gate.
- The security gate first failed on real findings. Fixable Python packages and
  the base runtime were upgraded; only upstream-unfixed findings are ignored.
- Updated and rendered both local resume variants after the remote gates
  passed.
- Added train-only evidence embeddings and typed citations to `/v1/retrieve`,
  plus runnable `retrieval build` and `retrieval query` CLI commands.
- Built the real 10,003-record BANKING77 training evidence index. The ambiguous
  `cash card payment` demo produced margin 0.0023, routed to review, and cited
  the top competing public training examples.
