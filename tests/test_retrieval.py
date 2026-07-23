from __future__ import annotations

from pathlib import Path

import numpy as np

from anchor_distill.data import BankingExample
from anchor_distill.retrieval import (
    NumpyIndex,
    build_index,
    retrieve_with_evidence,
)


class FakeModel:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = {
            "card arrival": [1.0, 0.0],
            "cash withdrawal": [0.0, 1.0],
            "where is my card": [0.9, 0.1],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_evidence_index_uses_training_split_only_and_round_trips(
    tmp_path: Path,
) -> None:
    index = build_index(
        FakeModel(),
        {
            "card_arrival": "card arrival",
            "cash_withdrawal": "cash withdrawal",
        },
        model_version="fake",
        examples=[
            BankingExample(
                "train-00001",
                "train",
                "where is my card",
                "card_arrival",
            ),
            BankingExample(
                "test-00001",
                "test",
                "never index this",
                "card_arrival",
            ),
        ],
    )
    path = tmp_path / "index.npz"
    index.save(path)
    restored = NumpyIndex.load(path)

    assert restored.evidence_ids == ["train-00001"]
    assert "never index this" not in restored.evidence_texts
    hits = restored.search_evidence(
        np.asarray([1.0, 0.0]),
        anchor_id="card_arrival",
        top_k=1,
    )
    assert hits[0]["citation_id"] == "BANKING77:train-00001"
    assert hits[0]["source"] == "BANKING77 train"

    result = retrieve_with_evidence(
        restored,
        np.asarray([1.0, 0.0]),
        query="where is it",
        top_k=2,
        evidence_per_hit=1,
    )
    assert result["evidence_split"] == "train"
    assert "[BANKING77:train-00001]" in result["explanation"]
