# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.engine.rewards.two_action_bandit import get_two_action_bandit_reward
from relax.utils.types import Sample


TASK_EOS = "eos"
TASK_BANDIT = "two_action_bandit"
SUPPORTED_TASKS = (TASK_EOS, TASK_BANDIT)


def get_eos_two_action_bandit_reward(sample: Sample) -> dict[str, float]:
    """Route a mixed-task sample while keeping one numeric primary reward key."""
    task = sample.metadata.get("task") if isinstance(sample.metadata, dict) else None
    if task == TASK_EOS:
        score = -float(sample.response_length)
        is_immediate_eos = sample.response_length == 1 and sample.status == Sample.Status.COMPLETED
        return {
            "score": score,
            "eos_reward": score,
            "eos_response_length": float(sample.response_length),
            "immediate_eos": float(is_immediate_eos),
            "task_eos": 1.0,
            "task_bandit": 0.0,
        }
    if task == TASK_BANDIT:
        bandit_reward = get_two_action_bandit_reward(sample)
        return {
            "score": bandit_reward["score"],
            "bandit_score": bandit_reward["score"],
            "action_a": bandit_reward["action_a"],
            "action_b": bandit_reward["action_b"],
            "valid_action": bandit_reward["valid_action"],
            "task_eos": 0.0,
            "task_bandit": 1.0,
        }
    raise ValueError(
        f"Mixed EOS/bandit reward requires sample.metadata['task'] to be one of {SUPPORTED_TASKS}, got {task!r}"
    )
