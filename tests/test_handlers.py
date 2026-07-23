from __future__ import annotations

from pathlib import Path

from anchor_distill.config import Settings
from anchor_distill.constants import JobType
from anchor_distill.jobs.handlers import build_handlers


def test_all_declared_job_types_have_concrete_handlers(tmp_path: Path) -> None:
    handlers = build_handlers(Settings(project_root=tmp_path))
    assert set(handlers) == set(JobType)
