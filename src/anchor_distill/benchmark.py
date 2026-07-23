from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class BenchmarkResult:
    model_version: str
    samples: int
    repeats: int
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_items_per_second: float
    python_traced_peak_memory_mb: float
    process_peak_rss_mb: float
    embedding_dimension: int
    python: str
    platform: str


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(quantile * len(ordered))) - 1)
    return ordered[index]


def benchmark_encoder(
    model: Encoder,
    texts: list[str],
    *,
    model_version: str,
    repeats: int = 5,
) -> BenchmarkResult:
    if not texts or repeats < 1:
        raise ValueError("texts and positive repeats are required")
    model.encode(texts[: min(2, len(texts))], show_progress_bar=False)
    timings: list[float] = []
    total_items = 0
    dimension = 0
    tracemalloc.start()
    for _ in range(repeats):
        started = time.perf_counter()
        embeddings = np.asarray(
            model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        )
        timings.append(time.perf_counter() - started)
        total_items += len(texts)
        dimension = int(embeddings.shape[1])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_megabytes = (
        maximum_rss / (1024 * 1024)
        if platform.system() == "Darwin"
        else maximum_rss / 1024
    )
    per_request_ms = [value * 1000 for value in timings]
    return BenchmarkResult(
        model_version=model_version,
        samples=len(texts),
        repeats=repeats,
        latency_p50_ms=statistics.median(per_request_ms),
        latency_p95_ms=percentile(per_request_ms, 0.95),
        throughput_items_per_second=total_items / sum(timings),
        python_traced_peak_memory_mb=peak / (1024 * 1024),
        process_peak_rss_mb=rss_megabytes,
        embedding_dimension=dimension,
        python=platform.python_version(),
        platform=platform.platform(),
    )


def save_benchmark(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
