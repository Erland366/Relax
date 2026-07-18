# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.types import Sample


QWEN_EOS_TEXT = "<|im_end|>"
VALID_ACTIONS = ("A", "B")
VALID_COMBINATIONS = ("00", "01", "10", "11")


def _validate_private_answer(sample: Sample) -> tuple[str, str]:
    if sample.label not in VALID_ACTIONS:
        raise ValueError(f"Visual XOR sample label must be one of {VALID_ACTIONS}, got {sample.label!r}")
    if not isinstance(sample.metadata, dict):
        raise ValueError("Visual XOR sample metadata must be a dictionary")

    required = ("left_bit", "right_bit", "combination")
    missing = [key for key in required if key not in sample.metadata]
    if missing:
        raise ValueError(f"Visual XOR sample metadata is missing required keys: {missing}")

    left_bit = sample.metadata["left_bit"]
    right_bit = sample.metadata["right_bit"]
    combination = sample.metadata["combination"]
    if left_bit not in (0, 1) or right_bit not in (0, 1) or combination not in VALID_COMBINATIONS:
        raise ValueError(
            "Visual XOR metadata must contain binary left/right bits and a valid combination; "
            f"got left_bit={left_bit!r}, right_bit={right_bit!r}, combination={combination!r}"
        )
    expected_combination = f"{left_bit}{right_bit}"
    if combination != expected_combination:
        raise ValueError(
            f"Visual XOR metadata is inconsistent: bits imply {expected_combination!r}, got {combination!r}"
        )
    expected_label = "A" if left_bit == right_bit else "B"
    if sample.label != expected_label:
        raise ValueError(
            f"Visual XOR label is inconsistent with combination {combination}: expected {expected_label!r}, "
            f"got {sample.label!r}"
        )
    return sample.label, combination


def get_visual_xor_reward(sample: Sample) -> dict[str, float]:
    """Score an exact A/B + EOS response for a balanced visual XOR sample."""
    label, combination = _validate_private_answer(sample)
    response = sample.response.strip()
    if response.endswith(QWEN_EOS_TEXT):
        response = response[: -len(QWEN_EOS_TEXT)].strip()

    completed = sample.status == Sample.Status.COMPLETED
    action_a = completed and sample.response_length == 2 and response == "A"
    action_b = completed and sample.response_length == 2 and response == "B"
    valid_action = action_a or action_b
    correct_action = valid_action and response == label
    score = 1.0 if correct_action else 0.0 if valid_action else -1.0

    reward = {
        "score": score,
        "correct_action": float(correct_action),
        "valid_action": float(valid_action),
        "action_a": float(action_a),
        "action_b": float(action_b),
        "target_a": float(label == "A"),
        "target_b": float(label == "B"),
        f"combo_{combination}_accuracy": float(correct_action),
        f"target_{label.lower()}_accuracy": float(correct_action),
    }
    return reward
