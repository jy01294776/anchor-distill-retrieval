from __future__ import annotations

from anchor_distill.data import BankingExample, select_training_examples


def _example(index: int, *, split: str, category: str) -> BankingExample:
    return BankingExample(
        example_id=f"{split}-{index}",
        split=split,
        text=f"example {index}",
        category=category,
    )


def test_select_training_examples_supports_full_training_split() -> None:
    examples = [
        _example(0, split="train", category="a"),
        _example(1, split="train", category="a"),
        _example(2, split="train", category="b"),
        _example(3, split="test", category="a"),
    ]

    selected = select_training_examples(
        examples,
        shots_per_category=None,
        seed=123,
    )

    assert [example.example_id for example in selected] == [
        "train-0",
        "train-1",
        "train-2",
    ]


def test_select_training_examples_keeps_few_shot_stratified() -> None:
    examples = [
        _example(0, split="train", category="a"),
        _example(1, split="train", category="a"),
        _example(2, split="train", category="b"),
        _example(3, split="train", category="b"),
        _example(4, split="test", category="a"),
    ]

    selected = select_training_examples(
        examples,
        shots_per_category=1,
        seed=123,
    )

    assert len(selected) == 2
    assert {example.category for example in selected} == {"a", "b"}
    assert all(example.split == "train" for example in selected)
