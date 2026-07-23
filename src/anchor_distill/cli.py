from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from anchor_distill.config import Settings
from anchor_distill.data import prepare_banking77
from anchor_distill.isolation import scan_project
from anchor_distill.teacher import TeacherClient

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
teacher_app = typer.Typer(no_args_is_help=True)
isolation_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(teacher_app, name="teacher")
app.add_typer(isolation_app, name="isolation")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(train_app, name="train")
app.add_typer(report_app, name="report")


class OutputFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


@data_app.command("prepare")
def data_prepare() -> None:
    manifest = prepare_banking77()
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@teacher_app.command("probe")
def teacher_probe() -> None:
    result = TeacherClient().probe()
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if not result.passed:
        raise typer.Exit(code=1)


@teacher_app.command("pilot")
def teacher_pilot(
    queries: Annotated[int, typer.Option(min=1, max=20)] = 2,
    negatives: Annotated[int, typer.Option(min=1, max=10)] = 4,
) -> None:
    from anchor_distill.teacher_pipeline import run_teacher_pilot

    settings = Settings()
    result = run_teacher_pilot(
        raw_dir=settings.path("data", "raw", "banking77"),
        output_path=settings.path(
            "data",
            "interim",
            "teacher_pilot.jsonl",
        ),
        calibration_path=settings.path(
            "artifacts",
            "reports",
            "teacher_pilot_calibration.json",
        ),
        query_count=queries,
        negative_count=negatives,
        seed=settings.seed,
        client=TeacherClient(settings),
    )
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))


@teacher_app.command("dataset")
def teacher_dataset(
    train_queries: Annotated[int, typer.Option(min=1, max=1000)] = 128,
    calibration_queries: Annotated[int, typer.Option(min=1, max=500)] = 64,
    candidates: Annotated[int, typer.Option(min=2, max=20)] = 10,
    max_cost_usd: Annotated[float, typer.Option(min=0.01, max=1.0)] = 0.25,
    workers: Annotated[int, typer.Option(min=1, max=32)] = 8,
    dry_run: Annotated[
        bool,
        typer.Option(help="Print the deterministic plan without API calls."),
    ] = False,
) -> None:
    from sentence_transformers import SentenceTransformer

    from anchor_distill.data import anchors_from_categories, load_banking77
    from anchor_distill.teacher_pipeline import (
        build_teacher_dataset_plan,
        run_teacher_dataset,
    )
    from anchor_distill.training import _device_name

    settings = Settings()
    selector_model = "sentence-transformers/all-MiniLM-L6-v2"
    selector = SentenceTransformer(selector_model, device=_device_name())
    if dry_run:
        examples = load_banking77(settings.path("data", "raw", "banking77"))
        anchors = anchors_from_categories(example.category for example in examples)
        plan, _, _ = build_teacher_dataset_plan(
            examples=examples,
            anchors=anchors,
            selector=selector,
            selector_model=selector_model,
            train_query_count=train_queries,
            calibration_query_count=calibration_queries,
            candidates_per_query=candidates,
            seed=settings.seed,
        )
        typer.echo(json.dumps(plan.__dict__, indent=2, sort_keys=True))
        if plan.estimated_upper_bound_usd > max_cost_usd:
            raise typer.Exit(code=1)
        return
    result = run_teacher_dataset(
        raw_dir=settings.path("data", "raw", "banking77"),
        train_output_path=settings.path(
            "data",
            "interim",
            "teacher_train.jsonl",
        ),
        calibration_output_path=settings.path(
            "data",
            "interim",
            "teacher_calibration.jsonl",
        ),
        summary_path=settings.path(
            "artifacts",
            "reports",
            "teacher_dataset_summary.json",
        ),
        selector=selector,
        selector_model=selector_model,
        train_query_count=train_queries,
        calibration_query_count=calibration_queries,
        candidates_per_query=candidates,
        max_cost_usd=max_cost_usd,
        workers=workers,
        seed=settings.seed,
        client=TeacherClient(settings),
    )
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True, default=str))


@isolation_app.command("check")
def isolation_check(
    root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = Settings()
    report = scan_project(
        root or settings.project_root,
        settings.forbidden_root_paths(),
    )
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    if not report.safe:
        raise typer.Exit(code=1)


@evaluate_app.command("zero-shot")
def evaluate_zero_shot(
    model_name: Annotated[
        str,
        typer.Option(help="SentenceTransformer model identifier."),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    output_name: Annotated[
        str,
        typer.Option(help="JSON filename under artifacts/benchmarks."),
    ] = "zero_shot_minilm.json",
) -> None:
    from sentence_transformers import SentenceTransformer

    from anchor_distill.data import (
        anchors_from_categories,
        load_banking77,
    )
    from anchor_distill.evaluation import evaluate_retriever, save_result
    from anchor_distill.training import _device_name

    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    anchors = anchors_from_categories(example.category for example in examples)
    device = _device_name()
    model = SentenceTransformer(model_name, device=device)
    result = evaluate_retriever(
        model,
        examples,
        anchors,
        model_version=model_name,
        split="test",
    )
    destination = settings.path("artifacts", "benchmarks", output_name)
    save_result(destination, result)
    typer.echo(
        json.dumps(
            {
                **result.__dict__,
                "device": device,
                "artifact": str(destination.relative_to(settings.project_root)),
            },
            indent=2,
            sort_keys=True,
        )
    )


@evaluate_app.command("compare")
def evaluate_compare(
    candidate_model: Annotated[str, typer.Option()] = ("models/gold_1shot_minilm"),
    baseline_model: Annotated[str, typer.Option()] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    output_name: Annotated[
        str | None,
        typer.Option(help="JSON filename under artifacts/benchmarks."),
    ] = None,
    bootstrap_replicates: Annotated[
        int,
        typer.Option(min=100, max=10000),
    ] = 1000,
) -> None:
    from sentence_transformers import SentenceTransformer

    from anchor_distill.comparison import compare_models, save_comparison
    from anchor_distill.data import (
        anchors_from_categories,
        load_banking77,
    )
    from anchor_distill.training import _device_name

    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    anchors = anchors_from_categories(example.category for example in examples)
    device = _device_name()
    baseline = SentenceTransformer(baseline_model, device=device)
    candidate = SentenceTransformer(candidate_model, device=device)
    comparison = compare_models(
        baseline,
        candidate,
        examples,
        anchors,
        baseline_name=baseline_model,
        candidate_name=candidate_model,
        replicates=bootstrap_replicates,
        seed=settings.seed,
    )
    artifact_name = output_name or (
        f"{Path(candidate_model).name}_vs_{Path(baseline_model).name}.json"
    )
    destination = settings.path("artifacts", "benchmarks", artifact_name)
    save_comparison(destination, comparison)
    typer.echo(
        json.dumps(
            {
                **comparison.__dict__,
                "device": device,
                "artifact": str(destination.relative_to(settings.project_root)),
            },
            indent=2,
            sort_keys=True,
        )
    )


@benchmark_app.command("encoder")
def benchmark_encoder_command(
    model_name: Annotated[
        str,
        typer.Option(help="SentenceTransformer model identifier."),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    output_name: Annotated[
        str | None,
        typer.Option(help="JSON filename under artifacts/benchmarks."),
    ] = None,
    samples: Annotated[int, typer.Option(min=1, max=1000)] = 256,
    repeats: Annotated[int, typer.Option(min=1, max=20)] = 5,
) -> None:
    from sentence_transformers import SentenceTransformer

    from anchor_distill.benchmark import (
        benchmark_encoder,
        save_benchmark,
    )
    from anchor_distill.data import load_banking77
    from anchor_distill.training import _device_name

    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    texts = [example.text for example in examples if example.split == "test"][:samples]
    device = _device_name()
    model = SentenceTransformer(model_name, device=device)
    result = benchmark_encoder(
        model,
        texts,
        model_version=model_name,
        repeats=repeats,
    )
    artifact_name = output_name or (
        f"encoder_{Path(model_name).name.replace('-', '_')}.json"
    )
    destination = settings.path("artifacts", "benchmarks", artifact_name)
    save_benchmark(destination, result)
    typer.echo(
        json.dumps(
            {
                **result.__dict__,
                "device": device,
                "artifact": str(destination.relative_to(settings.project_root)),
            },
            indent=2,
            sort_keys=True,
        )
    )


@report_app.command("frontier")
def report_frontier() -> None:
    from anchor_distill.frontier import build_quality_cost_frontier

    settings = Settings()
    result = build_quality_cost_frontier(
        benchmark_dir=settings.path("artifacts", "benchmarks"),
        teacher_summary_path=settings.path(
            "artifacts",
            "reports",
            "teacher_dataset_summary.json",
        ),
        json_output_path=settings.path(
            "artifacts",
            "reports",
            "quality_cost_frontier.json",
        ),
        markdown_output_path=settings.path(
            "artifacts",
            "reports",
            "quality_cost_frontier.md",
        ),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@train_app.command("gold")
def train_gold(
    shots: Annotated[int, typer.Option(min=1, max=16)] = 1,
    full: Annotated[
        bool,
        typer.Option(help="Use every official training example."),
    ] = False,
    epochs: Annotated[int, typer.Option(min=1, max=10)] = 2,
    base_model: Annotated[str, typer.Option()] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
) -> None:
    from anchor_distill.data import (
        anchors_from_categories,
        load_banking77,
        select_training_examples,
    )
    from anchor_distill.training import RetrievalTrainer

    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    training_examples = select_training_examples(
        examples,
        shots_per_category=None if full else shots,
        seed=settings.seed,
    )
    anchors = anchors_from_categories(example.category for example in examples)
    budget_name = "full" if full else f"{shots}shot"
    destination = settings.path(
        "models",
        f"gold_{budget_name}_minilm",
    )
    trainer = RetrievalTrainer(
        base_model=base_model,
        seed=settings.seed,
    )
    manifest = trainer.train(
        mode="gold",
        examples=training_examples,
        anchors=anchors,
        epochs=epochs,
        output_dir=destination,
        label_budget=len(training_examples),
    )
    typer.echo(json.dumps(manifest.__dict__, indent=2, sort_keys=True))


@train_app.command("distilled")
def train_distilled(
    mode: Annotated[str, typer.Option()] = "soft",
    objective: Annotated[str, typer.Option()] = "listwise",
    gold_shots: Annotated[int, typer.Option(min=1, max=16)] = 1,
    epochs: Annotated[int, typer.Option(min=1, max=10)] = 2,
    hybrid_lambda: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
) -> None:
    from anchor_distill.data import (
        anchors_from_categories,
        load_banking77,
        select_training_examples,
    )
    from anchor_distill.schemas import TeacherRecord
    from anchor_distill.training import RetrievalTrainer

    if mode not in {"hard", "soft", "hybrid"}:
        raise typer.BadParameter("mode must be hard, soft, or hybrid")
    if objective not in {"listwise", "ordinal"}:
        raise typer.BadParameter("objective must be listwise or ordinal")
    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    anchors = anchors_from_categories(example.category for example in examples)
    teacher_path = settings.path("data", "interim", "teacher_train.jsonl")
    teacher_records = [
        TeacherRecord.model_validate_json(line)
        for line in teacher_path.read_text().splitlines()
        if line.strip()
    ]
    gold_examples = (
        select_training_examples(
            examples,
            shots_per_category=gold_shots,
            seed=settings.seed,
        )
        if mode == "hybrid"
        else []
    )
    trainer_mode = f"listwise_{mode}" if objective == "listwise" else mode
    suffix = f"_{gold_shots}shot" if mode == "hybrid" else ""
    destination = settings.path(
        "models",
        f"{objective}_{mode}{suffix}_teacher_minilm",
    )
    trainer = RetrievalTrainer(seed=settings.seed)
    manifest = trainer.train(
        mode=trainer_mode,
        examples=gold_examples,
        anchors=anchors,
        teacher_records=teacher_records,
        query_text={example.example_id: example.text for example in examples},
        epochs=epochs,
        hybrid_lambda=hybrid_lambda,
        output_dir=destination,
        label_budget=len(gold_examples),
    )
    typer.echo(json.dumps(manifest.__dict__, indent=2, sort_keys=True))
