from copy import deepcopy

from examples.visual_xor.evaluate_refinement_sft import validate_refinement_summary


def _accepted_summary() -> dict:
    return {
        "xor_heldout": {
            "target_accuracy": 0.70,
            "valid_response_rate": 1.0,
            "action_a_rate": 0.50,
            "action_b_rate": 0.50,
            "mixed_group_rate": 0.95,
            "truncated_rate": 0.0,
            "combination_accuracy": {"00": 0.70, "01": 0.68, "10": 0.72, "11": 0.69},
        },
        "permuted_control": {"target_accuracy": 0.50, "valid_response_rate": 1.0},
        "constant_control": {"target_accuracy": 0.50, "valid_response_rate": 1.0},
        "counterfactual": {
            "mean_left_flip_directional_probability_shift": 0.35,
            "mean_right_flip_directional_probability_shift": 0.32,
            "mean_both_flip_absolute_probability_shift": 0.08,
        },
    }


def test_refinement_acceptance_requires_balanced_image_conditioned_headroom():
    assert validate_refinement_summary(_accepted_summary()) == []


def test_refinement_acceptance_rejects_global_action_bias_and_missing_right_glyph_sensitivity():
    summary = deepcopy(_accepted_summary())
    summary["xor_heldout"]["action_a_rate"] = 0.85
    summary["xor_heldout"]["action_b_rate"] = 0.15
    summary["xor_heldout"]["combination_accuracy"]["10"] = 0.40
    summary["counterfactual"]["mean_right_flip_directional_probability_shift"] = 0.02

    errors = validate_refinement_summary(summary)

    assert any("action A rate" in error for error in errors)
    assert any("action B rate" in error for error in errors)
    assert any("combination 10" in error for error in errors)
    assert any("right-flip" in error for error in errors)


def test_refinement_acceptance_rejects_nonvisual_control_performance():
    summary = deepcopy(_accepted_summary())
    summary["permuted_control"]["target_accuracy"] = 0.90
    summary["constant_control"]["target_accuracy"] = 0.10

    errors = validate_refinement_summary(summary)

    assert any("permuted control accuracy" in error for error in errors)
    assert any("constant control accuracy" in error for error in errors)
