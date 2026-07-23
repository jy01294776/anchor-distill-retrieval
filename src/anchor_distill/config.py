from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def discover_project_root() -> Path:
    configured = os.environ.get("ANCHOR_DISTILL_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=discover_project_root)
    openai_api_key: str | None = Field(default=None, repr=False)
    teacher_model: str = "gpt-4o-mini"
    database_url: str = "sqlite:///./work/jobs.db"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "./mlruns"
    log_level: str = "INFO"
    seed: int = 20260722
    teacher_minimum_label_mass: float = 0.95
    teacher_maximum_refusal_rate: float = 0.01
    forbidden_roots: str = ""

    @model_validator(mode="after")
    def normalize_and_validate(self) -> Self:
        self.project_root = self.project_root.expanduser().resolve()
        if self.database_url == "sqlite:///./work/jobs.db":
            self.database_url = f"sqlite:///{self.project_root / 'work' / 'jobs.db'}"
        if self.mlflow_tracking_uri == "./mlruns":
            self.mlflow_tracking_uri = str(self.project_root / "mlruns")
        if not 0 < self.teacher_minimum_label_mass <= 1:
            raise ValueError("teacher_minimum_label_mass must be in (0, 1]")
        if not 0 <= self.teacher_maximum_refusal_rate < 1:
            raise ValueError("teacher_maximum_refusal_rate must be in [0, 1)")
        return self

    def path(self, *parts: str) -> Path:
        candidate = self.project_root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {candidate}") from exc
        return candidate

    def forbidden_root_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for raw in self.forbidden_roots.split(os.pathsep):
            value = raw.strip()
            if value:
                paths.append(Path(value).expanduser().resolve())
        return tuple(paths)
