from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class FrontierRow:
    system: str
    training_signal: str
    human_labels: int
    teacher_queries: int
    teacher_pairs: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    parameter_count: int
    throughput_items_per_second: float
    process_peak_rss_mb: float
    performance_source: str


EXPERIMENTS = (
    (
        "MiniLM zero-shot",
        "zero-shot",
        0,
        0,
        0,
        "zero_shot_minilm.json",
        "encoder_minilm.json",
    ),
    (
        "MiniLM gold 1-shot",
        "gold",
        77,
        0,
        0,
        "gold_1shot_minilm.json",
        "encoder_minilm.json",
    ),
    (
        "MiniLM gold 4-shot",
        "gold",
        308,
        0,
        0,
        "gold_4shot_minilm.json",
        "encoder_minilm.json",
    ),
    (
        "MiniLM gold 16-shot",
        "gold",
        1232,
        0,
        0,
        "gold_16shot_minilm.json",
        "encoder_minilm.json",
    ),
    (
        "MiniLM gold full",
        "gold",
        10003,
        0,
        0,
        "gold_full_minilm.json",
        "encoder_minilm.json",
    ),
    (
        "MiniLM listwise hard KD",
        "teacher hard",
        0,
        128,
        1280,
        "listwise_hard_teacher_minilm.json",
        "encoder_listwise_hard_minilm.json",
    ),
    (
        "MiniLM listwise soft KD",
        "teacher soft",
        0,
        128,
        1280,
        "listwise_soft_teacher_minilm.json",
        "encoder_listwise_hard_minilm.json",
    ),
    (
        "MiniLM listwise hybrid",
        "gold + teacher soft",
        77,
        128,
        1280,
        "listwise_hybrid_1shot_teacher_minilm.json",
        "encoder_listwise_hard_minilm.json",
    ),
    (
        "Sentence-T5-XL zero-shot",
        "zero-shot",
        0,
        0,
        0,
        "zero_shot_sentence_t5_xl.json",
        "encoder_sentence_t5_xl.json",
    ),
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def build_quality_cost_frontier(
    *,
    benchmark_dir: Path,
    teacher_summary_path: Path,
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, Any]:
    rows: list[FrontierRow] = []
    for (
        system,
        training_signal,
        human_labels,
        teacher_queries,
        teacher_pairs,
        metric_name,
        performance_name,
    ) in EXPERIMENTS:
        metric_path = benchmark_dir / metric_name
        performance_path = benchmark_dir / performance_name
        if not metric_path.exists() or not performance_path.exists():
            continue
        metric = _load(metric_path)
        performance = _load(performance_path)
        rows.append(
            FrontierRow(
                system=system,
                training_signal=training_signal,
                human_labels=human_labels,
                teacher_queries=teacher_queries,
                teacher_pairs=teacher_pairs,
                recall_at_1=float(metric["recall_at_1"]),
                recall_at_5=float(metric["recall_at_5"]),
                mrr=float(metric["mrr"]),
                parameter_count=int(performance["parameter_count"]),
                throughput_items_per_second=float(
                    performance["throughput_items_per_second"]
                ),
                process_peak_rss_mb=float(performance["process_peak_rss_mb"]),
                performance_source=performance_name,
            )
        )
    by_name = {row.system: row for row in rows}
    student = by_name["MiniLM listwise hard KD"]
    reference = by_name["Sentence-T5-XL zero-shot"]
    teacher = _load(teacher_summary_path)
    recorded_pairs = int(teacher["recorded_pairs"])
    teacher_queries = int(teacher["plan"]["train_queries"]) + int(
        teacher["plan"]["calibration_queries"]
    )
    teacher_cost = float(teacher["estimated_cost_usd"])
    comparison = _load(benchmark_dir / "listwise_hard_vs_sentence_t5_xl.json")
    payload: dict[str, Any] = {
        "rows": [asdict(row) for row in rows],
        "teacher_cost": {
            "observed_pairs": recorded_pairs,
            "observed_queries": teacher_queries,
            "observed_cost_usd": teacher_cost,
            "projected_usd_per_million_pairs": (
                teacher_cost / recorded_pairs * 1_000_000
            ),
            "projected_usd_per_million_queries_at_current_candidate_count": (
                teacher_cost / teacher_queries * 1_000_000
            ),
            "candidates_per_query": int(teacher["plan"]["candidates_per_query"]),
            "pricing_source": teacher["pricing"]["source"],
            "pricing_captured_on": teacher["pricing"]["captured_on"],
        },
        "student_vs_large_encoder": {
            "recall_at_1_difference": (student.recall_at_1 - reference.recall_at_1),
            "recall_at_1_ci_low": comparison["recall_at_1"]["ci_low"],
            "recall_at_1_ci_high": comparison["recall_at_1"]["ci_high"],
            "parameter_reduction_factor": (
                reference.parameter_count / student.parameter_count
            ),
            "throughput_factor": (
                student.throughput_items_per_second
                / reference.throughput_items_per_second
            ),
            "peak_rss_reduction_factor": (
                reference.process_peak_rss_mb / student.process_peak_rss_mb
            ),
            "student_seconds_per_million_texts": (
                1_000_000 / student.throughput_items_per_second
            ),
            "reference_seconds_per_million_texts": (
                1_000_000 / reference.throughput_items_per_second
            ),
        },
        "limitations": [
            (
                "Local MPS throughput is a machine-specific batch benchmark, "
                "not a service SLA."
            ),
            (
                "The test-set comparisons are exploratory because multiple "
                "candidate systems were evaluated."
            ),
            "The observed confidence interval does not establish formal equivalence.",
            (
                "Dollar projections use observed teacher token mix and "
                "captured API prices."
            ),
        ],
    }
    _atomic_text(
        json_output_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    table_rows = [
        (
            f"| {row.system} | {row.training_signal} | {row.human_labels} | "
            f"{row.recall_at_1:.4f} | {row.mrr:.4f} | "
            f"{row.parameter_count / 1_000_000:.1f}M | "
            f"{row.throughput_items_per_second:.1f} | "
            f"{row.process_peak_rss_mb:.0f} |"
        )
        for row in rows
    ]
    efficiency = payload["student_vs_large_encoder"]
    cost = payload["teacher_cost"]
    projected_query_cost = cost[
        "projected_usd_per_million_queries_at_current_candidate_count"
    ]
    markdown = "\n".join(
        [
            "# Quality, Performance, and Cost Frontier",
            "",
            (
                "| System | Signal | Human labels | Recall@1 | MRR | Params | "
                "texts/s | Peak RSS MB |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## Student versus large encoder",
            "",
            (
                f"The listwise-hard MiniLM point estimate differs from "
                f"Sentence-T5-XL by {efficiency['recall_at_1_difference']:+.4f} "
                f"Recall@1; the paired 95% interval is "
                f"[{efficiency['recall_at_1_ci_low']:+.4f}, "
                f"{efficiency['recall_at_1_ci_high']:+.4f}]."
            ),
            (
                f"It uses {efficiency['parameter_reduction_factor']:.1f}x fewer "
                f"parameters, measured {efficiency['throughput_factor']:.1f}x "
                f"higher local throughput, and used "
                f"{efficiency['peak_rss_reduction_factor']:.1f}x less peak RSS."
            ),
            "",
            "## Teacher cost",
            "",
            (
                f"The observed teacher run cost ${cost['observed_cost_usd']:.5f} "
                f"for {cost['observed_pairs']:,} pairs. At the same token mix, "
                f"that projects to "
                f"${cost['projected_usd_per_million_pairs']:.2f} per million "
                f"pairs or "
                f"${projected_query_cost:.2f} "
                f"per million queries with {cost['candidates_per_query']} candidates."
            ),
            "",
            "## Interpretation limits",
            "",
            *[f"- {value}" for value in payload["limitations"]],
            "",
        ]
    )
    _atomic_text(markdown_output_path, markdown)
    return payload
