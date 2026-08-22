from __future__ import annotations

import json
from pathlib import Path

from anchor_distill.frontier import build_quality_cost_frontier


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_frontier_calculates_efficiency_and_teacher_cost(tmp_path: Path) -> None:
    benchmarks = tmp_path / "benchmarks"
    metric = {"recall_at_1": 0.5, "recall_at_5": 0.8, "mrr": 0.6}
    mini_perf = {
        "parameter_count": 10,
        "throughput_items_per_second": 100.0,
        "process_peak_rss_mb": 20.0,
    }
    large_perf = {
        "parameter_count": 100,
        "throughput_items_per_second": 10.0,
        "process_peak_rss_mb": 80.0,
    }
    _write(benchmarks / "listwise_hard_teacher_minilm.json", metric)
    _write(benchmarks / "zero_shot_sentence_t5_xl.json", metric)
    _write(benchmarks / "encoder_listwise_hard_minilm.json", mini_perf)
    _write(benchmarks / "encoder_sentence_t5_xl.json", large_perf)
    _write(
        benchmarks / "listwise_hard_vs_sentence_t5_xl.json",
        {"recall_at_1": {"ci_low": -0.01, "ci_high": 0.01}},
    )
    teacher_summary = tmp_path / "teacher.json"
    _write(
        teacher_summary,
        {
            "recorded_pairs": 20,
            "estimated_cost_usd": 0.02,
            "plan": {
                "train_queries": 1,
                "calibration_queries": 1,
                "candidates_per_query": 10,
            },
            "pricing": {"source": "official", "captured_on": "2026-07-23"},
        },
    )
    result = build_quality_cost_frontier(
        benchmark_dir=benchmarks,
        teacher_summary_path=teacher_summary,
        json_output_path=tmp_path / "frontier.json",
        markdown_output_path=tmp_path / "frontier.md",
    )
    efficiency = result["student_vs_large_encoder"]
    cost = result["teacher_cost"]
    assert efficiency["parameter_reduction_factor"] == 10
    assert efficiency["throughput_factor"] == 10
    assert cost["projected_usd_per_million_pairs"] == 1000
