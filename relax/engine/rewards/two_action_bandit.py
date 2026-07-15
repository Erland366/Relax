# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.types import Sample


QWEN_EOS_TEXT = "<|im_end|>"


def get_two_action_bandit_reward(sample: Sample) -> dict[str, float]:
    """Score one action token followed by EOS, exposing action-rate metrics."""
    response = sample.response.strip()
    if response.endswith(QWEN_EOS_TEXT):
        response = response[: -len(QWEN_EOS_TEXT)].strip()
    is_action_a = sample.response_length == 2 and response == "A"
    is_action_b = sample.response_length == 2 and response == "B"
    is_valid = is_action_a or is_action_b
    score = 1.0 if is_action_a else 0.0 if is_action_b else -1.0
    return {
        "score": score,
        "action_a": float(is_action_a),
        "action_b": float(is_action_b),
        "valid_action": float(is_valid),
    }
