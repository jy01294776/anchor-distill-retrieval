from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from anchor_distill.observability import RETRY_TOTAL

T = TypeVar("T")


class RetryableFailure(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class WorkerCrash(BaseException):
    """Deliberately bypass normal exception handling in a crash test."""


@dataclass
class FaultInjector:
    schedule: list[str]

    def trigger(self) -> None:
        if not self.schedule:
            return
        fault = self.schedule.pop(0)
        if fault == "429":
            raise RetryableFailure("429", "injected rate limit")
        if fault == "5xx":
            raise RetryableFailure("5xx", "injected server failure")
        if fault == "timeout":
            raise RetryableFailure("timeout", "injected timeout")
        if fault == "worker_crash":
            raise WorkerCrash("injected worker crash")
        if fault != "none":
            raise ValueError(f"unknown injected fault: {fault}")


def run_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.25,
    sleep: Callable[[float], None],
    jitter: Callable[[], float] = random.random,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except RetryableFailure as exc:
            RETRY_TOTAL.labels(operation_name, exc.reason).inc()
            if attempt == max_attempts:
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            sleep(delay * (0.5 + jitter()))
    raise AssertionError("retry loop exhausted without returning or raising")
