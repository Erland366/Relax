from examples.two_action_bandit.train_sft import (
    CHOICE_A,
    CHOICE_B,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    build_two_action_bandit_dataset,
)

import pytest


def build_dataset(num_examples: int = 10, choice_a_fraction: float = 0.5):
    return build_two_action_bandit_dataset(
        num_examples=num_examples,
        choice_a_fraction=choice_a_fraction,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT,
        choice_a=CHOICE_A,
        choice_b=CHOICE_B,
        seed=42,
    )


def test_bandit_dataset_has_identical_prompts_and_balanced_choices():
    dataset = build_dataset()

    assert len(dataset) == 10
    assert len({str(example["prompt"]) for example in dataset}) == 1
    assert sum(example["completion"][0]["content"] == CHOICE_A for example in dataset) == 5
    assert sum(example["completion"][0]["content"] == CHOICE_B for example in dataset) == 5
    assert all(example["chat_template_kwargs"] == {"enable_thinking": False} for example in dataset)


@pytest.mark.parametrize("choice_a_fraction", [0.0, 1.0, -0.1, 1.1])
def test_bandit_dataset_rejects_degenerate_fraction(choice_a_fraction):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build_dataset(choice_a_fraction=choice_a_fraction)


def test_bandit_dataset_rejects_identical_or_empty_choices():
    kwargs = {
        "num_examples": 10,
        "choice_a_fraction": 0.5,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "seed": 42,
    }

    with pytest.raises(ValueError, match="distinct non-empty"):
        build_two_action_bandit_dataset(choice_a="A", choice_b="A", **kwargs)
    with pytest.raises(ValueError, match="distinct non-empty"):
        build_two_action_bandit_dataset(choice_a="", choice_b="B", **kwargs)
