# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout conversion helpers for frozen Qwen3-VL vision features."""

from typing import Any

import torch

from relax.backends.vision.qwen3_vl import (
    Qwen3VLFrozenVisionFeatures,
    build_qwen3_vl_actor_feature_inputs,
    build_sglang_precomputed_image_data,
)


def prepare_qwen3_vl_precomputed_rollout_inputs(
    *,
    payload: dict[str, Any],
    multimodal_train_inputs: dict[str, Any],
    features: Qwen3VLFrozenVisionFeatures,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route one frozen feature bundle to SGLang and the training actor."""
    rollout_payload = dict(payload)
    rollout_payload["image_data"] = [build_sglang_precomputed_image_data(features)]

    actor_inputs = {
        key: value
        for key, value in multimodal_train_inputs.items()
        if key != "pixel_values"
    }
    actor_inputs.update(build_qwen3_vl_actor_feature_inputs(features))
    return rollout_payload, actor_inputs


def serialize_sglang_precomputed_image_data(image_data: dict[str, Any]) -> dict[str, Any]:
    """Convert the tensor payload to the JSON representation accepted by SGLang."""
    if image_data.get("format") != "precomputed_embedding":
        raise ValueError(f"Expected precomputed_embedding image data, got {image_data.get('format')!r}")
    feature = image_data.get("feature")
    image_grid_thw = image_data.get("image_grid_thw")
    if not isinstance(feature, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        raise TypeError("SGLang precomputed feature and image_grid_thw must be torch tensors before serialization")
    return {
        "format": "precomputed_embedding",
        "feature": feature.detach().to(device="cpu", dtype=torch.float32).tolist(),
        "image_grid_thw": image_grid_thw.detach().to(device="cpu", dtype=torch.int64).tolist(),
        "feature_id": image_data.get("feature_id", ""),
        "vision_revision": image_data.get("vision_revision", ""),
    }
