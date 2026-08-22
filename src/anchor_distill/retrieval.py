from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from anchor_distill.data import BankingExample


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
    evidence_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    evidence_anchor_ids: list[str] = field(default_factory=list)
    evidence_embeddings: np.ndarray | None = None

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

    def search_evidence(
        self,
        query_embedding: np.ndarray,
        *,
        anchor_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.evidence_embeddings is None or not self.evidence_ids:
            return []
        query = normalize_rows(np.asarray(query_embedding).reshape(1, -1))[0]
        candidates = np.flatnonzero(np.asarray(self.evidence_anchor_ids) == anchor_id)
        if not candidates.size:
            return []
        scores = self.evidence_embeddings[candidates] @ query
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "citation_id": f"BANKING77:{self.evidence_ids[candidates[index]]}",
                "source": "BANKING77 train",
                "example_id": self.evidence_ids[candidates[index]],
                "anchor_id": self.evidence_anchor_ids[candidates[index]],
                "text": self.evidence_texts[candidates[index]],
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
                evidence_ids=np.asarray(self.evidence_ids),
                evidence_texts=np.asarray(self.evidence_texts),
                evidence_anchor_ids=np.asarray(self.evidence_anchor_ids),
                evidence_embeddings=(
                    self.evidence_embeddings
                    if self.evidence_embeddings is not None
                    else np.empty((0, self.embeddings.shape[1]), dtype=np.float32)
                ),
            )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        with np.load(path, allow_pickle=False) as data:
            evidence_embeddings = (
                data["evidence_embeddings"]
                if "evidence_embeddings" in data.files
                and data["evidence_embeddings"].shape[0]
                else None
            )
            return cls(
                anchor_ids=data["anchor_ids"].tolist(),
                labels=data["labels"].tolist(),
                embeddings=data["embeddings"],
                model_version=str(data["model_version"][0]),
                evidence_ids=(
                    data["evidence_ids"].tolist()
                    if "evidence_ids" in data.files
                    else []
                ),
                evidence_texts=(
                    data["evidence_texts"].tolist()
                    if "evidence_texts" in data.files
                    else []
                ),
                evidence_anchor_ids=(
                    data["evidence_anchor_ids"].tolist()
                    if "evidence_anchor_ids" in data.files
                    else []
                ),
                evidence_embeddings=evidence_embeddings,
            )


def build_index(
    model: Any,
    anchors: dict[str, str],
    *,
    model_version: str,
    examples: Iterable[BankingExample] = (),
) -> NumpyIndex:
    anchor_ids = sorted(anchors)
    labels = [anchors[anchor_id] for anchor_id in anchor_ids]
    embeddings = model.encode(
        labels,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    evidence = [example for example in examples if example.split == "train"]
    evidence_embeddings = (
        normalize_rows(
            np.asarray(
                model.encode(
                    [example.text for example in evidence],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
        )
        if evidence
        else None
    )
    return NumpyIndex(
        anchor_ids,
        labels,
        normalize_rows(np.asarray(embeddings)),
        model_version,
        evidence_ids=[example.example_id for example in evidence],
        evidence_texts=[example.text for example in evidence],
        evidence_anchor_ids=[example.category for example in evidence],
        evidence_embeddings=evidence_embeddings,
    )


def retrieve_with_evidence(
    index: NumpyIndex,
    query_embedding: np.ndarray,
    *,
    query: str,
    top_k: int,
    evidence_per_hit: int,
    review_margin: float = 0.08,
) -> dict[str, Any]:
    raw_hits = index.search(query_embedding, top_k)
    margin = (
        raw_hits[0]["score"] - raw_hits[1]["score"]
        if len(raw_hits) > 1
        else raw_hits[0]["score"]
    )
    hits = [
        {
            **hit,
            "evidence": index.search_evidence(
                query_embedding,
                anchor_id=hit["anchor_id"],
                top_k=evidence_per_hit,
            ),
        }
        for hit in raw_hits
    ]
    top_citations = [
        citation["citation_id"] for hit in hits[:2] for citation in hit["evidence"][:1]
    ]
    decision = (
        f"Review required because the top-two confidence margin is below "
        f"{review_margin:.2f}."
        if margin < review_margin
        else f"Top intent accepted because the top-two confidence margin is "
        f"at least {review_margin:.2f}."
    )
    citation_text = (
        " Supporting public training examples: "
        + ", ".join(f"[{value}]" for value in top_citations)
        + "."
        if top_citations
        else " No evidence index was available."
    )
    return {
        "query": query,
        "model_version": index.model_version,
        "hits": hits,
        "needs_review": margin < review_margin,
        "confidence_margin": float(margin),
        "explanation": decision + citation_text,
        "evidence_split": "train",
    }


def save_index_manifest(path: Path, index: NumpyIndex) -> None:
    payload = {
        "model_version": index.model_version,
        "anchors": len(index.anchor_ids),
        "dimension": int(index.embeddings.shape[1]),
        "evidence_records": len(index.evidence_ids),
        "evidence_split": "train",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
