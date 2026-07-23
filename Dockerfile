FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[systems]"

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src ./src
USER appuser
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "anchor_distill.api:app", "--host", "0.0.0.0", "--port", "8000"]
