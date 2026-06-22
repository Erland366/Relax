# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.types import Sample


def get_completion_length_reward(sample: Sample) -> float:
    """Reward shorter completions by negating generated token count."""
    completion_length = int(sample.response_length)
    assert completion_length >= 0, f"response_length must be non-negative, got {completion_length}"
    return -float(completion_length)
