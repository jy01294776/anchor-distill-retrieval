from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from anchor_distill.data import (
    BankingExample,
    anchors_from_categories,
    load_banking77,
    stratified_few_shot,
)
from anchor_distill.evaluation import calibrate_teacher, save_result
from anchor_distill.schemas import TeacherRecord
from anchor_distill.teacher import (
    RUBRIC,
    TEACHER_CACHED_INPUT_USD_PER_MILLION,
    TEACHER_INPUT_USD_PER_MILLION,
    TEACHER_MAX_COMPLETION_TOKENS,
    TEACHER_OUTPUT_USD_PER_MILLION,
    TEACHER_PRICING_CAPTURED_ON,
    TEACHER_PRICING_SOURCE,
    estimate_teacher_cost,
)


@dataclass(frozen=True)
class TeacherPilotResult:
    records: int
    queries: int
    candidates_per_query: int
    accepted_records: int
    output_path: str
    calibration_path: str


@dataclass(frozen=True)
class TeacherPair:
    partition: str
    query_id: str
    query: str
    anchor_id: str
    anchor: str


@dataclass(frozen=True)
class TeacherDatasetPlan:
    train_queries: int
    calibration_queries: int
    candidates_per_query: int
    planned_pairs: int
    estimated_upper_bound_usd: float
    calibration_candidate_recall: float
    selector_model: str
    gold_labels_used_for_candidate_selection: bool


@dataclass(frozen=True)
class TeacherDatasetResult:
    plan: TeacherDatasetPlan
    recorded_pairs: int
    accepted_pairs: int
    train_pairs: int
    calibration_pairs: int
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    pair_exact_match_accuracy: float
    pair_brier_score: float
    pair_ece: float
    query_top1_accuracy: float
    mean_entropy: float
    train_output_path: str
    calibration_output_path: str
    summary_path: str


class TeacherScorer(Protocol):
    def score_pair(
        self,
        *,
        query_id: str,
        query: str,
        anchor_id: str,
        anchor: str,
    ) -> TeacherRecord: ...


def _negative_anchors(
    query_id: str,
    gold_anchor: str,
    anchor_ids: list[str],
    count: int,
) -> list[str]:
    candidates = [anchor_id for anchor_id in anchor_ids if anchor_id != gold_anchor]
    return sorted(
        candidates,
        key=lambda anchor_id: hashlib.sha256(
            f"{query_id}:{anchor_id}".encode()
        ).hexdigest(),
    )[:count]


def _write_records(path: Path, records: list[TeacherRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n"
    )
    os.replace(temporary, path)


def _read_records(path: Path) -> list[TeacherRecord]:
    if not path.exists():
        return []
    return [
        TeacherRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _relative_output_paths(*paths: Path) -> tuple[str, ...]:
    """Return portable paths without exposing the machine-specific project root."""
    resolved = [path.resolve() for path in paths]
    shared_root = Path(os.path.commonpath([str(path) for path in resolved]))
    if shared_root in resolved:
        shared_root = shared_root.parent
    return tuple(path.relative_to(shared_root).as_posix() for path in resolved)


def _hash_order(
    examples: list[BankingExample],
    *,
    seed: int,
) -> list[BankingExample]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{seed}:{example.example_id}".encode()
        ).hexdigest(),
    )


def _upper_bound_tokens(query: str, anchor: str) -> int:
    characters = len(RUBRIC) + len(query) + len(anchor) + 240
    return math.ceil(characters / 3)


def build_teacher_dataset_plan(
    *,
    examples: list[BankingExample],
    anchors: dict[str, str],
    selector: Any,
    selector_model: str,
    train_query_count: int,
    calibration_query_count: int,
    candidates_per_query: int,
    seed: int,
) -> tuple[TeacherDatasetPlan, list[TeacherPair], dict[str, str]]:
    if train_query_count < 1 or calibration_query_count < 1:
        raise ValueError("teacher train and calibration query counts must be positive")
    if not 1 <= candidates_per_query <= len(anchors):
        raise ValueError("candidates_per_query is outside the anchor bank")
    ordered = _hash_order(
        [example for example in examples if example.split == "train"],
        seed=seed,
    )
    requested = train_query_count + calibration_query_count
    if len(ordered) < requested:
        raise ValueError("not enough public training examples for teacher partitions")
    train_examples = ordered[:train_query_count]
    calibration_examples = ordered[train_query_count:requested]
    selected = train_examples + calibration_examples
    anchor_ids = sorted(anchors)
    query_embeddings = np.asarray(
        selector.encode(
            [example.text for example in selected],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    anchor_embeddings = np.asarray(
        selector.encode(
            [anchors[anchor_id] for anchor_id in anchor_ids],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    scores = query_embeddings @ anchor_embeddings.T
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :candidates_per_query]
    pairs: list[TeacherPair] = []
    gold_labels: dict[str, str] = {}
    calibration_covered = 0
    for index, example in enumerate(selected):
        partition = "train" if index < train_query_count else "calibration"
        candidates = [anchor_ids[value] for value in ranked[index]]
        if partition == "calibration" and example.category in candidates:
            calibration_covered += 1
        gold_labels[example.example_id] = example.category
        pairs.extend(
            TeacherPair(
                partition=partition,
                query_id=example.example_id,
                query=example.text,
                anchor_id=anchor_id,
                anchor=anchors[anchor_id],
            )
            for anchor_id in candidates
        )
    upper_bound = sum(
        estimate_teacher_cost(
            prompt_tokens=_upper_bound_tokens(pair.query, pair.anchor),
            cached_prompt_tokens=0,
            completion_tokens=TEACHER_MAX_COMPLETION_TOKENS,
        )
        for pair in pairs
    )
    plan = TeacherDatasetPlan(
        train_queries=train_query_count,
        calibration_queries=calibration_query_count,
        candidates_per_query=candidates_per_query,
        planned_pairs=len(pairs),
        estimated_upper_bound_usd=upper_bound,
        calibration_candidate_recall=calibration_covered / calibration_query_count,
        selector_model=selector_model,
        gold_labels_used_for_candidate_selection=False,
    )
    return plan, pairs, gold_labels


def _score_with_retry(
    client: TeacherScorer,
    pair: TeacherPair,
    *,
    attempts: int = 5,
) -> TeacherRecord:
    transient = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    for attempt in range(attempts):
        try:
            return client.score_pair(
                query_id=pair.query_id,
                query=pair.query,
                anchor_id=pair.anchor_id,
                anchor=pair.anchor,
            )
        except transient:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError("unreachable retry state")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _query_top1_accuracy(
    records: list[TeacherRecord],
    gold_labels: dict[str, str],
) -> float:
    grouped: dict[str, list[TeacherRecord]] = {}
    for record in records:
        if record.accepted:
            grouped.setdefault(record.query_id, []).append(record)
    correct = 0
    for query_id in gold_labels:
        candidates = grouped.get(query_id, [])
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda record: sum(
                rating * probability
                for rating, probability in enumerate(
                    record.label_probabilities,
                    start=1,
                )
            ),
        )
        correct += best.anchor_id == gold_labels[query_id]
    return correct / len(gold_labels) if gold_labels else 0.0


def run_teacher_dataset(
    *,
    raw_dir: Path,
    train_output_path: Path,
    calibration_output_path: Path,
    summary_path: Path,
    selector: Any,
    selector_model: str,
    train_query_count: int,
    calibration_query_count: int,
    candidates_per_query: int,
    max_cost_usd: float,
    workers: int,
    seed: int,
    client: TeacherScorer,
) -> TeacherDatasetResult:
    if max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    examples = load_banking77(raw_dir)
    anchors = anchors_from_categories(example.category for example in examples)
    plan, pairs, gold_labels = build_teacher_dataset_plan(
        examples=examples,
        anchors=anchors,
        selector=selector,
        selector_model=selector_model,
        train_query_count=train_query_count,
        calibration_query_count=calibration_query_count,
        candidates_per_query=candidates_per_query,
        seed=seed,
    )
    if plan.estimated_upper_bound_usd > max_cost_usd:
        raise ValueError(
            "teacher plan exceeds the explicit cost cap: "
            f"{plan.estimated_upper_bound_usd:.6f} > {max_cost_usd:.6f}"
        )

    train_records = _read_records(train_output_path)
    calibration_records = _read_records(calibration_output_path)
    keyed = {
        (record.query_id, record.anchor_id): record
        for record in train_records + calibration_records
    }
    pair_lookup = {(pair.query_id, pair.anchor_id): pair for pair in pairs}
    remaining = [pair for pair in pairs if (pair.query_id, pair.anchor_id) not in keyed]
    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_pairs = {
            executor.submit(_score_with_retry, client, pair): pair for pair in remaining
        }
        try:
            for future in as_completed(future_pairs):
                pair = future_pairs[future]
                record = future.result()
                keyed[(pair.query_id, pair.anchor_id)] = record
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= 20:
                    _write_records(
                        train_output_path,
                        sorted(
                            (
                                value
                                for key, value in keyed.items()
                                if pair_lookup[key].partition == "train"
                            ),
                            key=lambda value: (value.query_id, value.anchor_id),
                        ),
                    )
                    _write_records(
                        calibration_output_path,
                        sorted(
                            (
                                value
                                for key, value in keyed.items()
                                if pair_lookup[key].partition == "calibration"
                            ),
                            key=lambda value: (value.query_id, value.anchor_id),
                        ),
                    )
                    completed_since_checkpoint = 0
        finally:
            train_records = sorted(
                (
                    value
                    for key, value in keyed.items()
                    if key in pair_lookup and pair_lookup[key].partition == "train"
                ),
                key=lambda value: (value.query_id, value.anchor_id),
            )
            calibration_records = sorted(
                (
                    value
                    for key, value in keyed.items()
                    if key in pair_lookup
                    and pair_lookup[key].partition == "calibration"
                ),
                key=lambda value: (value.query_id, value.anchor_id),
            )
            _write_records(train_output_path, train_records)
            _write_records(calibration_output_path, calibration_records)

    calibration_gold = {
        query_id: label
        for query_id, label in gold_labels.items()
        if any(record.query_id == query_id for record in calibration_records)
    }
    pair_calibration = calibrate_teacher(calibration_records, calibration_gold)
    all_records = train_records + calibration_records
    latencies = [record.latency_ms for record in all_records]
    relative_train, relative_calibration, relative_summary = _relative_output_paths(
        train_output_path,
        calibration_output_path,
        summary_path,
    )
    result = TeacherDatasetResult(
        plan=plan,
        recorded_pairs=len(all_records),
        accepted_pairs=sum(record.accepted for record in all_records),
        train_pairs=len(train_records),
        calibration_pairs=len(calibration_records),
        prompt_tokens=sum(record.prompt_tokens for record in all_records),
        cached_prompt_tokens=sum(record.cached_prompt_tokens for record in all_records),
        completion_tokens=sum(record.completion_tokens for record in all_records),
        estimated_cost_usd=sum(record.estimated_cost_usd for record in all_records),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        pair_exact_match_accuracy=pair_calibration.exact_match_accuracy,
        pair_brier_score=pair_calibration.brier_score,
        pair_ece=pair_calibration.ece,
        query_top1_accuracy=_query_top1_accuracy(
            calibration_records,
            calibration_gold,
        ),
        mean_entropy=float(np.mean([record.entropy for record in all_records])),
        train_output_path=relative_train,
        calibration_output_path=relative_calibration,
        summary_path=relative_summary,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                **asdict(result),
                "pricing": {
                    "input_usd_per_million": TEACHER_INPUT_USD_PER_MILLION,
                    "cached_input_usd_per_million": (
                        TEACHER_CACHED_INPUT_USD_PER_MILLION
                    ),
                    "output_usd_per_million": TEACHER_OUTPUT_USD_PER_MILLION,
                    "source": TEACHER_PRICING_SOURCE,
                    "captured_on": TEACHER_PRICING_CAPTURED_ON,
                    "max_cost_usd": max_cost_usd,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, summary_path)
    return result


def run_teacher_pilot(
    *,
    raw_dir: Path,
    output_path: Path,
    calibration_path: Path,
    query_count: int,
    negative_count: int,
    seed: int,
    client: TeacherScorer,
) -> TeacherPilotResult:
    if query_count < 1 or negative_count < 1:
        raise ValueError("query_count and negative_count must be positive")
    examples = load_banking77(raw_dir)
    anchors = anchors_from_categories(example.category for example in examples)
    selected = stratified_few_shot(examples, 1, seed=seed)[:query_count]
    records: list[TeacherRecord] = []
    anchor_ids = sorted(anchors)
    for example in selected:
        candidates = [example.category]
        candidates.extend(
            _negative_anchors(
                example.example_id,
                example.category,
                anchor_ids,
                negative_count,
            )
        )
        for anchor_id in candidates:
            records.append(
                client.score_pair(
                    query_id=example.example_id,
                    query=example.text,
                    anchor_id=anchor_id,
                    anchor=anchors[anchor_id],
                )
            )
    _write_records(output_path, records)
    calibration = calibrate_teacher(
        records,
        {example.example_id: example.category for example in selected},
    )
    save_result(calibration_path, calibration)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "pilot": asdict(
                    TeacherPilotResult(
                        records=len(records),
                        queries=len(selected),
                        candidates_per_query=negative_count + 1,
                        accepted_records=sum(record.accepted for record in records),
                        output_path=str(output_path),
                        calibration_path=str(calibration_path),
                    )
                ),
                "model_resolved": sorted({record.model_resolved for record in records}),
                "prompt_hash": sorted({record.prompt_hash for record in records}),
                "rubric_id": sorted({record.rubric_id for record in records}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TeacherPilotResult(
        records=len(records),
        queries=len(selected),
        candidates_per_query=negative_count + 1,
        accepted_records=sum(record.accepted for record in records),
        output_path=str(output_path),
        calibration_path=str(calibration_path),
    )
