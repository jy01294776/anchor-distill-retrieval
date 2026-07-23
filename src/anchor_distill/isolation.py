from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from anchor_distill.config import Settings

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".ini",
    ".cfg",
    ".sh",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "models",
    "mlruns",
    "postgres-data",
    "prometheus-data",
    "redis-data",
}
PRIVATE_DATA_SUFFIXES = {".parquet", ".xlsx", ".xls", ".sav", ".dta"}
ABSOLUTE_HOME_PATTERN = re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[^/\s]+/")


@dataclass(frozen=True)
class IsolationIssue:
    path: str
    rule: str
    detail: str


@dataclass(frozen=True)
class IsolationReport:
    project_root: str
    checked_files: int
    issues: tuple[IsolationIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def safe(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "project_root": self.project_root,
            "checked_files": self.checked_files,
            "safe": self.safe,
            "issues": [
                {
                    "path": issue.path,
                    "rule": issue.rule,
                    "detail": issue.detail,
                }
                for issue in self.issues
            ],
        }


def _should_skip(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in SKIP_PARTS for part in relative_parts)


def check_isolation(settings: Settings | None = None) -> IsolationReport:
    active = settings or Settings()
    root = active.project_root
    issues: list[IsolationIssue] = []
    checked = 0
    forbidden = tuple(str(path) for path in active.forbidden_root_paths())

    for path in root.rglob("*"):
        if _should_skip(path, root):
            continue
        relative = path.relative_to(root)
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root)
            except ValueError:
                issues.append(
                    IsolationIssue(
                        str(relative),
                        "external_symlink",
                        "symlink resolves outside the project root",
                    )
                )
            continue
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in PRIVATE_DATA_SUFFIXES:
            issues.append(
                IsolationIssue(
                    str(relative),
                    "private_data_extension",
                    f"disallowed committed-data suffix {path.suffix}",
                )
            )
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "Dockerfile",
            "Makefile",
            "LICENSE",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for blocked in forbidden:
            if blocked and blocked in text:
                issues.append(
                    IsolationIssue(
                        str(relative),
                        "forbidden_root_reference",
                        "text contains a configured forbidden root",
                    )
                )
        for match in ABSOLUTE_HOME_PATTERN.finditer(text):
            issues.append(
                IsolationIssue(
                    str(relative),
                    "absolute_home_path",
                    f"absolute user-home path begins at byte {match.start()}",
                )
            )
    return IsolationReport(str(root), checked, tuple(issues))


def scan_project(
    root: Path,
    forbidden_roots: tuple[Path, ...] = (),
) -> IsolationReport:
    settings = Settings(
        project_root=root,
        forbidden_roots=":".join(str(path) for path in forbidden_roots),
    )
    return check_isolation(settings)
