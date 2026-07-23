from __future__ import annotations

from pathlib import Path

from anchor_distill.schemas import TeacherRecord
from anchor_distill.teacher_pipeline import run_teacher_pilot


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
