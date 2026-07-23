# AGENTS.md

## Role

Build a factual, recruiter-facing ML/LLM engineering project. Distinguish
measured evidence from planned work and never turn an unverified result into a
resume claim.

## Isolation contract

1. This repository is a clean-room implementation.
2. Never read, import, copy, symlink, or reference the doctoral research
   workspace.
3. All runtime paths must resolve inside this repository.
4. Only public, license-compatible data may enter `data/`.
5. Do not add research data, proprietary constructs, manuscript results,
   submission code, judge outputs, or research-trained model weights.
6. Secrets belong only in ignored local environment files.
7. Run `anchor-distill isolation check` before tests, releases, or publication.

## File organization

- Production code: `src/`
- Tests and fixtures: `tests/`
- Plans and decisions: `context/`
- Progress and logs: `work/`
- Public-data stages: `data/raw`, `data/interim`, `data/processed`
- Models: `models/`
- Reports, benchmarks, and resume evidence: `artifacts/`
- Short handoff summaries: `output/`

Do not add new top-level folders without explicit approval.

## Workflow

- Inspect before editing.
- Use public dataset checksums and versioned experiment manifests.
- Treat the official test split as final evaluation only.
- Keep API, worker, training, and evaluation operations idempotent.
- Call an execution exactly-once only if that property is actually proven.
  This project uses at-least-once task execution with idempotent artifact
  commits.
- Use `Built` or `Packaged` until a real deployment exists.
- Update `work/progress.md` and `context/decisions.md` before ending a task.

## Validation

- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy src`
- `uv run anchor-distill isolation check`
- `docker compose config`
