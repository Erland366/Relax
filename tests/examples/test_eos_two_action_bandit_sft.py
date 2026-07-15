import json
from pathlib import Path

import pytest

from examples.eos_two_action_bandit.train_sft import (
    BANDIT_SYSTEM_PROMPT,
    BANDIT_USER_PROMPT,
    CHOICE_A,
    CHOICE_B,
    EOS_ALTERNATIVE,
    EOS_SYSTEM_PROMPT,
    EOS_USER_PROMPT,
    TASK_BANDIT,
    TASK_EOS,
    build_joint_sft_dataset,
)


def build_dataset(
    num_examples_per_task: int = 10,
    eos_fraction: float = 0.5,
    choice_a_fraction: float = 0.5,
):
    return build_joint_sft_dataset(
        num_examples_per_task=num_examples_per_task,
        eos_fraction=eos_fraction,
        choice_a_fraction=choice_a_fraction,
        seed=42,
    )


def test_joint_sft_dataset_balances_tasks_and_targets():
    dataset = build_dataset()

    assert len(dataset) == 20
    assert sum(example["task"] == TASK_EOS for example in dataset) == 10
    assert sum(example["task"] == TASK_BANDIT for example in dataset) == 10
    assert sum(example["completion"][0]["content"] == "" for example in dataset) == 5
    assert sum(example["completion"][0]["content"] == EOS_ALTERNATIVE for example in dataset) == 5
    assert sum(example["completion"][0]["content"] == CHOICE_A for example in dataset) == 5
    assert sum(example["completion"][0]["content"] == CHOICE_B for example in dataset) == 5
    assert len({str(example["prompt"]) for example in dataset}) == 2
    assert all(example["chat_template_kwargs"] == {"enable_thinking": False} for example in dataset)


@pytest.mark.parametrize("fraction_name", ["eos_fraction", "choice_a_fraction"])
@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_joint_sft_dataset_rejects_degenerate_task_mixtures(fraction_name, fraction):
    kwargs = {"eos_fraction": 0.5, "choice_a_fraction": 0.5}
    kwargs[fraction_name] = fraction

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build_dataset(**kwargs)


def test_joint_sft_dataset_rejects_too_few_examples_per_task():
    with pytest.raises(ValueError, match="at least 2"):
        build_dataset(num_examples_per_task=1)


def test_relax_prompt_batch_contains_exactly_one_row_per_task():
    prompt_path = Path("examples/eos_two_action_bandit/prompts.jsonl")
    rows = [json.loads(line) for line in prompt_path.read_text().splitlines()]

    assert len(rows) == 2
    assert [row["metadata"]["task"] for row in rows] == [TASK_EOS, TASK_BANDIT]
    assert rows[0]["prompt"] == [
        {"role": "system", "content": EOS_SYSTEM_PROMPT},
        {"role": "user", "content": EOS_USER_PROMPT},
    ]
    assert rows[1]["prompt"] == [
        {"role": "system", "content": BANDIT_SYSTEM_PROMPT},
        {"role": "user", "content": BANDIT_USER_PROMPT},
    ]
