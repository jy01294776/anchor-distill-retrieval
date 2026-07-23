from __future__ import annotations

from anchor_distill.faults import FaultInjector, run_with_retry


def test_retry_recovers_after_injected_transient_failures() -> None:
    injector = FaultInjector(["429", "5xx", "timeout", "none"])
    delays: list[float] = []

    def operation() -> str:
        injector.trigger()
        return "committed-once"

    result = run_with_retry(
        operation,
        operation_name="test",
        max_attempts=4,
        sleep=delays.append,
        jitter=lambda: 0,
    )
    assert result == "committed-once"
    assert delays == [0.125, 0.25, 0.5]
