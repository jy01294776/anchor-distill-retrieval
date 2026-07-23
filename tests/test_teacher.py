from __future__ import annotations

import math

from anchor_distill.schemas import TokenAlternative, TokenPosition
from anchor_distill.teacher import extract_soft_distribution


def test_extracts_normalized_five_label_distribution() -> None:
    position = TokenPosition(
        token="4",
        logprob=math.log(0.60),
        bytes=[52],
        top_logprobs=[
            TokenAlternative(token="4", logprob=math.log(0.60), bytes=[52]),
            TokenAlternative(token="5", logprob=math.log(0.25), bytes=[53]),
            TokenAlternative(token="3", logprob=math.log(0.10), bytes=[51]),
            TokenAlternative(token=",", logprob=math.log(0.05), bytes=[44]),
        ],
    )
    probabilities, mass, entropy, accepted = extract_soft_distribution(
        [position], selected_rating=4, minimum_mass=0.90
    )
    assert accepted
    assert mass == 0.95
    assert sum(probabilities) == 1.0
    assert probabilities[3] > probabilities[4] > probabilities[2]
    assert entropy > 0


def test_rejects_missing_rating_position() -> None:
    position = TokenPosition(
        token="{",
        logprob=0,
        bytes=[123],
        top_logprobs=[],
    )
    probabilities, mass, entropy, accepted = extract_soft_distribution(
        [position], selected_rating=3, minimum_mass=0.90
    )
    assert probabilities == (0.0, 0.0, 1.0, 0.0, 0.0)
    assert mass == 0
    assert entropy == 0
    assert not accepted
