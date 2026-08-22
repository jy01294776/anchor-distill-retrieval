from __future__ import annotations

import math

from anchor_distill.schemas import TeacherRecord, TokenAlternative, TokenPosition
from anchor_distill.teacher import (
    estimate_teacher_cost,
    extract_soft_distribution,
)
from anchor_distill.training import teacher_expected_relevance


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


def test_teacher_cost_separates_cached_and_uncached_input() -> None:
    cost = estimate_teacher_cost(
        prompt_tokens=1000,
        cached_prompt_tokens=400,
        completion_tokens=100,
    )
    expected = (600 * 0.15 + 400 * 0.075 + 100 * 0.60) / 1_000_000
    assert cost == expected


def test_expected_relevance_uses_full_soft_distribution() -> None:
    record = TeacherRecord(
        query_id="q",
        anchor_id="a",
        rubric_id="r",
        prompt_hash="p",
        model_requested="m",
        model_resolved="m",
        selected_rating=4,
        label_probabilities=(0.0, 0.1, 0.2, 0.6, 0.1),
        recognized_mass=1.0,
        entropy=0.5,
        accepted=True,
    )
    assert teacher_expected_relevance(record) == 3.7
