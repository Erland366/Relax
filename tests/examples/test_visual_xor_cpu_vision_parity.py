import json

import pytest
import torch

from examples.visual_xor.cpu_vision_parity import (
    FEATURE_STREAM_NAMES,
    compare_feature_streams,
    compare_response_logits,
    write_parity_artifact,
)


def _feature_streams() -> dict[str, torch.Tensor]:
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    return {
        "vision_embeds": base.clone(),
        "deepstack_visual_embeds_0": (base + 1.0).clone(),
        "deepstack_visual_embeds_1": (base + 2.0).clone(),
        "deepstack_visual_embeds_2": (base + 3.0).clone(),
    }


def _response_logits() -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    candidate = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.01, 1.99]], dtype=torch.float64)
    return reference, candidate


def test_feature_comparison_reports_each_named_final_and_deepstack_stream():
    reference = _feature_streams()
    candidate = _feature_streams()
    candidate["vision_embeds"][1, 1] += 0.01

    metrics = compare_feature_streams(reference, candidate)
    expected_cosine = torch.nn.functional.cosine_similarity(
        reference["vision_embeds"].flatten(),
        candidate["vision_embeds"].flatten(),
        dim=0,
    ).item()

    assert tuple(metrics) == FEATURE_STREAM_NAMES
    final_metrics = metrics["vision_embeds"]
    assert final_metrics["shape"] == [2, 2]
    assert final_metrics["cosine_similarity"] == pytest.approx(expected_cosine)
    assert final_metrics["max_absolute_error"] == pytest.approx(0.01)
    assert final_metrics["mean_absolute_error"] == pytest.approx(0.0025)
    assert final_metrics["within_rtol_atol"] is True
    assert final_metrics["rtol_atol_close_fraction"] == 1.0

    deepstack_metrics = metrics["deepstack_visual_embeds_2"]
    assert deepstack_metrics["shape"] == [2, 2]
    assert deepstack_metrics["cosine_similarity"] == pytest.approx(1.0)
    assert deepstack_metrics["max_absolute_error"] == 0.0
    assert deepstack_metrics["mean_absolute_error"] == 0.0
    assert deepstack_metrics["within_rtol_atol"] is True
    assert deepstack_metrics["rtol_atol_close_fraction"] == 1.0


def test_feature_comparison_rejects_missing_or_renamed_streams():
    reference = _feature_streams()
    candidate = _feature_streams()
    candidate["deepstack_0"] = candidate.pop("deepstack_visual_embeds_0")

    with pytest.raises(ValueError, match="feature stream names"):
        compare_feature_streams(reference, candidate)


def test_feature_comparison_rejects_shape_mismatch():
    reference = _feature_streams()
    candidate = _feature_streams()
    candidate["deepstack_visual_embeds_1"] = candidate["deepstack_visual_embeds_1"].flatten()

    with pytest.raises(ValueError, match="deepstack_visual_embeds_1.*shape"):
        compare_feature_streams(reference, candidate)


def test_feature_comparison_rejects_nonfinite_values():
    reference = _feature_streams()
    candidate = _feature_streams()
    candidate["vision_embeds"][0, 0] = torch.nan

    with pytest.raises(ValueError, match="vision_embeds.*finite"):
        compare_feature_streams(reference, candidate)


def test_feature_comparison_applies_cosine_gate():
    reference = _feature_streams()
    candidate = _feature_streams()
    reference["vision_embeds"] = torch.tensor([[0.001, 0.0], [0.0, 0.001]], dtype=torch.float64)
    candidate["vision_embeds"] = torch.tensor([[0.0, 0.001], [0.001, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="vision_embeds.*cosine"):
        compare_feature_streams(reference, candidate)


def test_feature_comparison_applies_rtol_and_atol_gate():
    reference = _feature_streams()
    candidate = _feature_streams()
    reference["vision_embeds"] = torch.ones((100, 10), dtype=torch.float64)
    candidate["vision_embeds"] = reference["vision_embeds"].clone()
    candidate["vision_embeds"][0, 0] = 1.1

    with pytest.raises(ValueError, match="vision_embeds.*rtol.*atol"):
        compare_feature_streams(reference, candidate)


def test_feature_comparison_allows_sparse_bfloat16_cross_device_rounding():
    reference = {
        name: torch.ones((100_000,), dtype=torch.float64)
        for name in FEATURE_STREAM_NAMES
    }
    candidate = {name: tensor.clone() for name, tensor in reference.items()}
    candidate["vision_embeds"][0] += 0.03125

    metrics = compare_feature_streams(reference, candidate)

    final_metrics = metrics["vision_embeds"]
    assert final_metrics["within_rtol_atol"] is False
    assert final_metrics["rtol_atol_close_fraction"] == pytest.approx(0.99999)
    assert final_metrics["max_absolute_error"] == 0.03125


def test_feature_comparison_rejects_sparse_large_absolute_error():
    reference = {
        name: torch.ones((100_000,), dtype=torch.float64)
        for name in FEATURE_STREAM_NAMES
    }
    candidate = {name: tensor.clone() for name, tensor in reference.items()}
    candidate["vision_embeds"][0] += 0.1

    with pytest.raises(ValueError, match="vision_embeds.*maximum absolute error"):
        compare_feature_streams(reference, candidate)


def test_response_logit_comparison_reports_raw_per_sample_metrics():
    reference, candidate = _response_logits()

    metrics = compare_response_logits(
        reference,
        candidate,
        action_a_token_id=0,
        action_b_token_id=2,
    )
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    expected_kl = (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1)

    assert metrics["mean_kl_divergence"] == pytest.approx(expected_kl.mean().item())
    assert metrics["max_per_sample_kl_divergence"] == pytest.approx(expected_kl.max().item())
    assert metrics["top_token_match_rate"] == 1.0
    assert len(metrics["per_sample"]) == 2
    for sample_index, expected_top_token_id in enumerate((0, 2)):
        sample = metrics["per_sample"][sample_index]
        assert sample["kl_divergence"] == pytest.approx(expected_kl[sample_index].item())
        assert sample["reference_top_token_id"] == expected_top_token_id
        assert sample["candidate_top_token_id"] == expected_top_token_id
        assert sample["top_token_match"] is True
        assert sample["reference_p_a"] == pytest.approx(reference_probs[sample_index, 0].item())
        assert sample["candidate_p_a"] == pytest.approx(candidate_probs[sample_index, 0].item())
        assert sample["p_a_absolute_delta"] == pytest.approx(
            abs(reference_probs[sample_index, 0].item() - candidate_probs[sample_index, 0].item())
        )
        assert sample["reference_p_b"] == pytest.approx(reference_probs[sample_index, 2].item())
        assert sample["candidate_p_b"] == pytest.approx(candidate_probs[sample_index, 2].item())
        assert sample["p_b_absolute_delta"] == pytest.approx(
            abs(reference_probs[sample_index, 2].item() - candidate_probs[sample_index, 2].item())
        )


def test_response_logit_comparison_rejects_shape_mismatch():
    reference, candidate = _response_logits()

    with pytest.raises(ValueError, match="logit shape"):
        compare_response_logits(
            reference,
            candidate[:, :2],
            action_a_token_id=0,
            action_b_token_id=1,
        )


def test_response_logit_comparison_rejects_nonfinite_values():
    reference, candidate = _response_logits()
    candidate[0, 0] = torch.inf

    with pytest.raises(ValueError, match="logits.*finite"):
        compare_response_logits(
            reference,
            candidate,
            action_a_token_id=0,
            action_b_token_id=2,
        )


def test_response_logit_comparison_applies_mean_kl_gate():
    reference = torch.zeros((2, 4), dtype=torch.float64)
    candidate = torch.tensor([[0.1, -0.1, 0.0, 0.0], [0.1, -0.1, 0.0, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="mean KL"):
        compare_response_logits(
            reference,
            candidate,
            action_a_token_id=2,
            action_b_token_id=3,
        )


def test_response_logit_comparison_applies_per_sample_kl_gate():
    reference = torch.zeros((20, 4), dtype=torch.float64)
    candidate = reference.clone()
    candidate[0] = torch.tensor([0.25, -0.25, 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="per-sample KL"):
        compare_response_logits(
            reference,
            candidate,
            action_a_token_id=2,
            action_b_token_id=3,
        )


def test_response_logit_comparison_rejects_top_token_mismatch_below_numeric_gates():
    reference = torch.tensor([[0.001, 0.0, -1.0, -1.0]], dtype=torch.float64)
    candidate = torch.tensor([[0.0, 0.001, -1.0, -1.0]], dtype=torch.float64)

    with pytest.raises(
        ValueError,
        match=r"sample 0 top-token mismatch: reference 0, candidate 1",
    ):
        compare_response_logits(
            reference,
            candidate,
            action_a_token_id=2,
            action_b_token_id=3,
        )


@pytest.mark.parametrize(
    ("changed_token_id", "error_pattern"),
    [(2, r"P\(A\).*0.01"), (3, r"P\(B\).*0.01")],
)
def test_response_logit_comparison_applies_action_probability_delta_gate(changed_token_id, error_pattern):
    reference = torch.zeros((1, 4), dtype=torch.float64)
    candidate = reference.clone()
    candidate[0, changed_token_id] = 0.06

    with pytest.raises(ValueError, match=error_pattern):
        compare_response_logits(
            reference,
            candidate,
            action_a_token_id=2,
            action_b_token_id=3,
        )


def test_parity_artifact_is_versioned_json_with_raw_metrics(tmp_path):
    feature_metrics = compare_feature_streams(_feature_streams(), _feature_streams())
    reference_logits, candidate_logits = _response_logits()
    response_metrics = compare_response_logits(
        reference_logits,
        candidate_logits,
        action_a_token_id=0,
        action_b_token_id=2,
    )
    output_path = tmp_path / "nested" / "cpu_vision_parity.json"

    artifact = write_parity_artifact(
        output_path,
        feature_metrics=feature_metrics,
        response_metrics=response_metrics,
    )

    assert json.loads(output_path.read_text()) == artifact
    assert artifact["schema_version"] == 1
    assert artifact["passed"] is True
    assert "metadata" not in artifact
    assert artifact["gates"] == {
        "feature_cosine_min": 0.999,
        "feature_rtol": 0.01,
        "feature_atol": 0.01,
        "feature_close_fraction_min": 0.9999,
        "feature_max_absolute_error_max": 0.05,
        "mean_kl_max": 0.001,
        "per_sample_kl_max": 0.01,
        "action_probability_delta_max": 0.01,
    }
    assert artifact["feature_streams"]["vision_embeds"]["shape"] == [2, 2]
    assert len(artifact["response_logits"]["per_sample"]) == 2


def test_parity_artifact_includes_metadata_when_provided(tmp_path):
    metadata = {
        "checkpoint": "/models/qwen3-vl",
        "feature_ids": ["feature-1"],
        "vision_revision": "revision-1",
    }

    artifact = write_parity_artifact(
        tmp_path / "cpu_vision_parity.json",
        feature_metrics={"vision_embeds": {"shape": [4, 8]}},
        response_metrics={"mean_kl_divergence": 0.0, "per_sample": []},
        metadata=metadata,
    )

    assert artifact["metadata"] == metadata
