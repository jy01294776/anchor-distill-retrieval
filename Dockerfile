FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade \
      "wheel>=0.46.2" \
      "jaraco.context>=6.1.0" \
    && python -m pip install --no-cache-dir --prefix=/install ".[systems]"

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade \
      "wheel>=0.46.2" \
      "jaraco.context>=6.1.0" \
    && useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src ./src
USER appuser
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "anchor_distill.api:app", "--host", "0.0.0.0", "--port", "8000"]
