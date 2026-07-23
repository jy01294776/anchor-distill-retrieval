from __future__ import annotations

from pathlib import Path

from anchor_distill.isolation import scan_project


def test_isolation_detects_forbidden_reference(tmp_path: Path) -> None:
    forbidden = tmp_path.parent / "private-paper"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "bad.py").write_text(f'PATH = "{forbidden}/data"')
    report = scan_project(tmp_path, (forbidden,))
    assert not report.safe
    assert {issue.rule for issue in report.issues} == {"forbidden_root_reference"}


def test_env_local_is_not_read(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=not-a-real-key\n")
    report = scan_project(tmp_path)
    assert report.safe
