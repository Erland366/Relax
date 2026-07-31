"""Pure comparison helpers for frozen CPU vision parity checks."""

import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


FEATURE_STREAM_NAMES = (
    "vision_embeds",
    "deepstack_visual_embeds_0",
    "deepstack_visual_embeds_1",
    "deepstack_visual_embeds_2",
)

PARITY_GATES = {
    "feature_cosine_min": 0.999,
    "feature_rtol": 0.01,
    "feature_atol": 0.01,
    "feature_close_fraction_min": 0.9999,
    "feature_max_absolute_error_max": 0.05,
    "mean_kl_max": 0.001,
    "per_sample_kl_max": 0.01,
    "action_probability_delta_max": 0.01,
}


def compare_feature_streams(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    """Compare the final and DeepStack vision feature streams."""
    expected_names = set(FEATURE_STREAM_NAMES)
    if set(reference) != expected_names or set(candidate) != expected_names:
        raise ValueError(f"feature stream names must be exactly {FEATURE_STREAM_NAMES}")

    metrics = {}
    for name in FEATURE_STREAM_NAMES:
        reference_tensor = reference[name]
        candidate_tensor = candidate[name]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"{name} shape mismatch: reference {list(reference_tensor.shape)}, "
                f"candidate {list(candidate_tensor.shape)}"
            )
        if not torch.isfinite(reference_tensor).all() or not torch.isfinite(candidate_tensor).all():
            raise ValueError(f"{name} values must be finite")

        absolute_error = (reference_tensor - candidate_tensor).abs()
        cosine_similarity = F.cosine_similarity(
            reference_tensor.flatten(),
            candidate_tensor.flatten(),
            dim=0,
        ).item()
        within_rtol_atol_mask = torch.isclose(
            reference_tensor,
            candidate_tensor,
            rtol=PARITY_GATES["feature_rtol"],
            atol=PARITY_GATES["feature_atol"],
        )
        within_rtol_atol = bool(within_rtol_atol_mask.all())
        rtol_atol_close_fraction = within_rtol_atol_mask.to(dtype=torch.float64).mean().item()
        metrics[name] = {
            "shape": list(reference_tensor.shape),
            "cosine_similarity": cosine_similarity,
            "max_absolute_error": absolute_error.max().item(),
            "mean_absolute_error": absolute_error.mean().item(),
            "within_rtol_atol": within_rtol_atol,
            "rtol_atol_close_fraction": rtol_atol_close_fraction,
        }

        if cosine_similarity < PARITY_GATES["feature_cosine_min"]:
            raise ValueError(
                f"{name} cosine similarity {cosine_similarity} is below "
                f"{PARITY_GATES['feature_cosine_min']}"
            )
        if rtol_atol_close_fraction < PARITY_GATES["feature_close_fraction_min"]:
            raise ValueError(
                f"{name} rtol={PARITY_GATES['feature_rtol']} and "
                f"atol={PARITY_GATES['feature_atol']} close fraction "
                f"{rtol_atol_close_fraction} is below "
                f"{PARITY_GATES['feature_close_fraction_min']}"
            )
        if metrics[name]["max_absolute_error"] > PARITY_GATES["feature_max_absolute_error_max"]:
            raise ValueError(
                f"{name} maximum absolute error {metrics[name]['max_absolute_error']} exceeds "
                f"{PARITY_GATES['feature_max_absolute_error_max']}"
            )

    return metrics


def compare_response_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    action_a_token_id: int,
    action_b_token_id: int,
) -> dict[str, object]:
    """Compare response logits and action probabilities sample by sample."""
    if reference.shape != candidate.shape:
        raise ValueError(f"logit shape mismatch: reference {list(reference.shape)}, candidate {list(candidate.shape)}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("response logits must be finite")

    reference_log_probs = torch.log_softmax(reference, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    per_sample_kl = (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1)
    reference_top_tokens = reference.argmax(dim=-1)
    candidate_top_tokens = candidate.argmax(dim=-1)
    top_token_matches = reference_top_tokens == candidate_top_tokens

    per_sample_metrics = []
    for sample_index in range(reference.shape[0]):
        reference_p_a = reference_probs[sample_index, action_a_token_id].item()
        candidate_p_a = candidate_probs[sample_index, action_a_token_id].item()
        reference_p_b = reference_probs[sample_index, action_b_token_id].item()
        candidate_p_b = candidate_probs[sample_index, action_b_token_id].item()
        per_sample_metrics.append(
            {
                "kl_divergence": per_sample_kl[sample_index].item(),
                "reference_top_token_id": reference_top_tokens[sample_index].item(),
                "candidate_top_token_id": candidate_top_tokens[sample_index].item(),
                "top_token_match": top_token_matches[sample_index].item(),
                "reference_p_a": reference_p_a,
                "candidate_p_a": candidate_p_a,
                "p_a_absolute_delta": abs(reference_p_a - candidate_p_a),
                "reference_p_b": reference_p_b,
                "candidate_p_b": candidate_p_b,
                "p_b_absolute_delta": abs(reference_p_b - candidate_p_b),
            }
        )

    metrics = {
        "mean_kl_divergence": per_sample_kl.mean().item(),
        "max_per_sample_kl_divergence": per_sample_kl.max().item(),
        "top_token_match_rate": top_token_matches.float().mean().item(),
        "per_sample": per_sample_metrics,
    }

    if metrics["mean_kl_divergence"] > PARITY_GATES["mean_kl_max"]:
        raise ValueError(
            f"mean KL divergence {metrics['mean_kl_divergence']} exceeds {PARITY_GATES['mean_kl_max']}"
        )
    if metrics["max_per_sample_kl_divergence"] > PARITY_GATES["per_sample_kl_max"]:
        raise ValueError(
            "maximum per-sample KL divergence "
            f"{metrics['max_per_sample_kl_divergence']} exceeds {PARITY_GATES['per_sample_kl_max']}"
        )
    action_probability_deltas = (
        ("P(A)", "p_a_absolute_delta"),
        ("P(B)", "p_b_absolute_delta"),
    )
    for sample_index, sample_metrics in enumerate(per_sample_metrics):
        for action_name, delta_name in action_probability_deltas:
            if sample_metrics[delta_name] > PARITY_GATES["action_probability_delta_max"]:
                raise ValueError(
                    f"sample {sample_index} {action_name} delta {sample_metrics[delta_name]} exceeds "
                    f"{PARITY_GATES['action_probability_delta_max']}"
                )
        if not sample_metrics["top_token_match"]:
            raise ValueError(
                f"sample {sample_index} top-token mismatch: "
                f"reference {sample_metrics['reference_top_token_id']}, "
                f"candidate {sample_metrics['candidate_top_token_id']}"
            )

    return metrics


def write_parity_artifact(
    output_path: str | Path,
    *,
    feature_metrics: Mapping[str, object],
    response_metrics: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write a versioned JSON artifact containing raw parity metrics."""
    artifact = {
        "schema_version": 1,
        "passed": True,
        "gates": PARITY_GATES.copy(),
        "feature_streams": dict(feature_metrics),
        "response_logits": dict(response_metrics),
    }
    if metadata is not None:
        artifact["metadata"] = dict(metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact
