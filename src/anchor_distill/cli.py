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
app.add_typer(data_app, name="data")
app.add_typer(teacher_app, name="teacher")
app.add_typer(isolation_app, name="isolation")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(train_app, name="train")


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
    destination = settings.path(
        "artifacts",
        "benchmarks",
        "gold_1shot_vs_zero_shot.json",
    )
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
    destination = settings.path(
        "artifacts",
        "benchmarks",
        "encoder_minilm.json",
    )
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


@train_app.command("gold")
def train_gold(
    shots: Annotated[int, typer.Option(min=1, max=16)] = 1,
    epochs: Annotated[int, typer.Option(min=1, max=10)] = 2,
    base_model: Annotated[str, typer.Option()] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
) -> None:
    from anchor_distill.data import (
        anchors_from_categories,
        load_banking77,
        stratified_few_shot,
    )
    from anchor_distill.training import RetrievalTrainer

    settings = Settings()
    examples = load_banking77(settings.path("data", "raw", "banking77"))
    training_examples = stratified_few_shot(
        examples,
        shots,
        seed=settings.seed,
    )
    anchors = anchors_from_categories(example.category for example in examples)
    destination = settings.path(
        "models",
        f"gold_{shots}shot_minilm",
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
