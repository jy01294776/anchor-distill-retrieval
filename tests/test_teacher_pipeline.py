from __future__ import annotations

from pathlib import Path

import numpy as np

from anchor_distill.data import BankingExample
from anchor_distill.schemas import TeacherRecord
from anchor_distill.teacher_pipeline import (
    _relative_output_paths,
    build_teacher_dataset_plan,
    run_teacher_pilot,
)


class FakeTeacher:
    def score_pair(
        self,
        *,
        query_id: str,
        query: str,
        anchor_id: str,
        anchor: str,
    ) -> TeacherRecord:
        del query, anchor
        positive = query_id.endswith("00000") and anchor_id == "card_arrival"
        selected = 5 if positive else 1
        probabilities = (
            (0.01, 0.01, 0.01, 0.07, 0.90)
            if positive
            else (0.90, 0.07, 0.01, 0.01, 0.01)
        )
        return TeacherRecord(
            query_id=query_id,
            anchor_id=anchor_id,
            rubric_id="test-rubric",
            prompt_hash="test-prompt",
            model_requested="fake",
            model_resolved="fake-snapshot",
            selected_rating=selected,
            label_probabilities=probabilities,
            recognized_mass=1.0,
            entropy=0.4,
            accepted=True,
        )


class FakeSelector:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = {
            "query a": [1.0, 0.0],
            "query b": [1.0, 0.0],
            "anchor a": [0.0, 1.0],
            "anchor b": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=float)


def test_teacher_plan_does_not_force_gold_anchor_into_candidates() -> None:
    examples = [
        BankingExample("train-0", "train", "query a", "a"),
        BankingExample("train-1", "train", "query b", "b"),
    ]
    plan, pairs, _ = build_teacher_dataset_plan(
        examples=examples,
        anchors={"a": "anchor a", "b": "anchor b"},
        selector=FakeSelector(),
        selector_model="fake",
        train_query_count=1,
        calibration_query_count=1,
        candidates_per_query=1,
        seed=7,
    )
    assert not plan.gold_labels_used_for_candidate_selection
    assert all(pair.anchor_id == "b" for pair in pairs)


def test_relative_output_paths_do_not_expose_project_root(tmp_path: Path) -> None:
    train, calibration, summary = _relative_output_paths(
        tmp_path / "data" / "interim" / "teacher_train.jsonl",
        tmp_path / "data" / "interim" / "teacher_calibration.jsonl",
        tmp_path / "artifacts" / "reports" / "teacher_dataset_summary.json",
    )

    assert train == "data/interim/teacher_train.jsonl"
    assert calibration == "data/interim/teacher_calibration.jsonl"
    assert summary == "artifacts/reports/teacher_dataset_summary.json"
    assert str(tmp_path) not in "\n".join((train, calibration, summary))


def test_teacher_pilot_uses_training_examples_only(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.csv").write_text(
        "text,category\nwhere is my card,card_arrival\ncash missing,cash_withdrawal\n"
    )
    (raw_dir / "test.csv").write_text("text,category\nnever label this,card_arrival\n")
    result = run_teacher_pilot(
        raw_dir=raw_dir,
        output_path=tmp_path / "teacher.jsonl",
        calibration_path=tmp_path / "calibration.json",
        query_count=1,
        negative_count=1,
        seed=7,
        client=FakeTeacher(),
    )
    assert result.records == 2
    assert result.accepted_records == 2
    output = (tmp_path / "teacher.jsonl").read_text()
    assert "never label this" not in output
    assert (tmp_path / "calibration.json").exists()
