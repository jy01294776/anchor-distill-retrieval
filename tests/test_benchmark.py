from __future__ import annotations

import numpy as np

from anchor_distill.benchmark import benchmark_encoder


class FakeParameter:
    def numel(self) -> int:
        return 10

    def element_size(self) -> int:
        return 4


class FakeEncoder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.ones((len(texts), 3))

    def parameters(self) -> list[FakeParameter]:
        return [FakeParameter(), FakeParameter()]


def test_benchmark_records_parameter_evidence() -> None:
    result = benchmark_encoder(
        FakeEncoder(),
        ["a", "b"],
        model_version="fake",
        repeats=1,
    )
    assert result.parameter_count == 20
    assert result.parameter_memory_mb == 80 / (1024 * 1024)
