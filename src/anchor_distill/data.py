from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from anchor_distill.config import Settings

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)
FILES = {
    "train.csv": f"{BASE_URL}/train.csv",
    "test.csv": f"{BASE_URL}/test.csv",
    "categories.json": f"{BASE_URL}/categories.json",
}
EXPECTED_ROWS = {"train": 10003, "test": 3080}
EXPECTED_CATEGORIES = 77


@dataclass(frozen=True)
class BankingExample:
    example_id: str
    split: str
    text: str
    category: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def prepare_banking77(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    active = settings or Settings()
    raw_dir = active.path("data", "raw", "banking77")
    raw_dir.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    transport = client or httpx.Client(timeout=60, follow_redirects=True)
    downloaded: dict[str, dict[str, object]] = {}
    try:
        for filename, url in FILES.items():
            destination = raw_dir / filename
            response = transport.get(url)
            response.raise_for_status()
            _atomic_write(destination, response.content)
            downloaded[filename] = {
                "url": url,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
    finally:
        if owns_client:
            transport.close()

    examples = load_banking77(raw_dir)
    counts = {
        split: sum(example.split == split for example in examples)
        for split in EXPECTED_ROWS
    }
    categories = sorted({example.category for example in examples})
    if counts != EXPECTED_ROWS:
        raise ValueError(f"unexpected split counts: {counts}")
    if len(categories) != EXPECTED_CATEGORIES:
        raise ValueError(f"expected 77 categories, found {len(categories)}")

    manifest: dict[str, object] = {
        "dataset": "BANKING77",
        "license": "CC BY 4.0",
        "source": (
            "https://github.com/PolyAI-LDN/task-specific-datasets/"
            "tree/master/banking_data"
        ),
        "files": downloaded,
        "rows": counts,
        "categories": len(categories),
    }
    manifest_path = active.path("data", "raw", "banking77", "manifest.json")
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def load_banking77(raw_dir: Path) -> list[BankingExample]:
    examples: list[BankingExample] = []
    for split in ("train", "test"):
        path = raw_dir / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                examples.append(
                    BankingExample(
                        example_id=f"{split}-{index:05d}",
                        split=split,
                        text=row["text"].strip(),
                        category=row["category"].strip(),
                    )
                )
    return examples


def anchors_from_categories(categories: Iterable[str]) -> dict[str, str]:
    return {
        category: category.replace("_", " ").replace("?", "").strip()
        for category in sorted(set(categories))
    }


def stratified_few_shot(
    examples: Iterable[BankingExample],
    shots_per_category: int,
    *,
    seed: int,
) -> list[BankingExample]:
    if shots_per_category < 1:
        raise ValueError("shots_per_category must be positive")
    import random

    grouped: dict[str, list[BankingExample]] = {}
    for example in examples:
        if example.split != "train":
            continue
        grouped.setdefault(example.category, []).append(example)
    selected: list[BankingExample] = []
    for category in sorted(grouped):
        values = list(grouped[category])
        random.Random(f"{seed}:{category}").shuffle(values)
        selected.extend(values[:shots_per_category])
    return selected


def write_examples(path: Path, examples: Iterable[BankingExample]) -> None:
    rows = [asdict(example) for example in examples]
    _atomic_write(
        path,
        ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode(),
    )
