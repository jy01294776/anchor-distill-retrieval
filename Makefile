.PHONY: test lint typecheck isolation api worker

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

isolation:
	uv run anchor-distill isolation check

api:
	uv run uvicorn anchor_distill.api:app --reload

worker:
	uv run celery -A anchor_distill.jobs.tasks.celery_app worker --loglevel=INFO
