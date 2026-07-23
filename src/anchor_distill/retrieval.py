from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return cast(np.ndarray, matrix / np.clip(norms, 1e-12, None))


@dataclass
class NumpyIndex:
    anchor_ids: list[str]
    labels: list[str]
    embeddings: np.ndarray
    model_version: str

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        query = normalize_rows(np.asarray(query_embedding).reshape(1, -1))[0]
        scores = self.embeddings @ query
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "anchor_id": self.anchor_ids[index],
                "label": self.labels[index],
                "score": float(scores[index]),
            }
            for index in order
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                anchor_ids=np.asarray(self.anchor_ids),
                labels=np.asarray(self.labels),
                embeddings=self.embeddings,
                model_version=np.asarray([self.model_version]),
            )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                anchor_ids=data["anchor_ids"].tolist(),
                labels=data["labels"].tolist(),
                embeddings=data["embeddings"],
                model_version=str(data["model_version"][0]),
            )


def build_index(
    model: Any,
    anchors: dict[str, str],
    *,
    model_version: str,
) -> NumpyIndex:
    anchor_ids = sorted(anchors)
    labels = [anchors[anchor_id] for anchor_id in anchor_ids]
    embeddings = model.encode(
        labels,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return NumpyIndex(anchor_ids, labels, embeddings, model_version)


def save_index_manifest(path: Path, index: NumpyIndex) -> None:
    payload = {
        "model_version": index.model_version,
        "anchors": len(index.anchor_ids),
        "dimension": int(index.embeddings.shape[1]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
