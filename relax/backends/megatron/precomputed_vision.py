# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron input adapter for frozen Qwen3-VL vision features."""

from types import MethodType
from typing import Any

import torch

from relax.backends.vision.qwen3_vl import (
    IMAGE_GRID_THW_KEY,
    VISION_EMBEDS_KEY,
    Qwen3VLFrozenVisionFeatures,
    deepstack_visual_embeds_key,
)


class OmittedQwen3VLVisionModel(torch.nn.Module):
    """Parameterless guard used when the frozen visual tower lives on CPU."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen3-VL GPU vision weights were omitted; provide precomputed vision features"
        )


def build_qwen3_vl_precomputed_forward_kwargs(
    *,
    multimodal_train_inputs: dict[str, Any],
    input_ids: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    hidden_size: int,
    expected_deepstack_count: int,
    context_parallel_size: int,
) -> dict[str, Any]:
    """Validate and assemble Qwen3-VL's final and DeepStack feature streams."""
    if context_parallel_size != 1:
        raise NotImplementedError("Precomputed Qwen3-VL vision features do not support context parallel input")
    if any(key.startswith("video") or "video_" in key for key in multimodal_train_inputs):
        raise NotImplementedError("Precomputed Qwen3-VL vision features do not support video input")

    deepstack_visual_embeds = tuple(
        multimodal_train_inputs[deepstack_visual_embeds_key(index)]
        for index in range(expected_deepstack_count)
    )
    features = Qwen3VLFrozenVisionFeatures(
        image_grid_thw=multimodal_train_inputs[IMAGE_GRID_THW_KEY],
        vision_embeds=multimodal_train_inputs[VISION_EMBEDS_KEY],
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    image_token_count = int((input_ids == image_token_id).sum().item())
    features.validate(
        spatial_merge_size=spatial_merge_size,
        image_token_count=image_token_count,
        hidden_size=hidden_size,
        expected_deepstack_count=expected_deepstack_count,
    )

    return {
        "image_grid_thw": features.image_grid_thw,
        "vision_embeds": features.vision_embeds,
        "deepstack_visual_embeds": features.deepstack_visual_embeds,
    }


def install_qwen3_vl_precomputed_vision_forward(model: Any) -> None:
    """Make one Bridge Qwen3-VL instance consume projected CPU features."""
    original_forward = model.forward
    if getattr(original_forward, "_relax_precomputed_vision_forward", False):
        return

    def _forward_with_precomputed_vision(*args, **kwargs):
        vision_embeds = kwargs.pop("vision_embeds", None)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        has_numbered_deepstack_keys = any(
            key.startswith("deepstack_visual_embeds_") for key in kwargs
        )
        if vision_embeds is None and deepstack_visual_embeds is None and not has_numbered_deepstack_keys:
            return original_forward(*args, **kwargs)

        expected_deepstack_count = len(model.vision_transformer_config.deepstack_visual_indexes)
        expected_numbered_deepstack_keys = tuple(
            deepstack_visual_embeds_key(index) for index in range(expected_deepstack_count)
        )
        if has_numbered_deepstack_keys:
            if deepstack_visual_embeds is not None:
                raise ValueError(
                    "Provide either deepstack_visual_embeds or numbered DeepStack feature keys, not both"
                )
            missing_keys = tuple(key for key in expected_numbered_deepstack_keys if key not in kwargs)
            if missing_keys:
                raise ValueError(
                    "Incomplete numbered DeepStack feature stream; "
                    f"missing {', '.join(missing_keys)}"
                )
            deepstack_visual_embeds = tuple(
                kwargs.pop(key) for key in expected_numbered_deepstack_keys
            )
        if vision_embeds is None or deepstack_visual_embeds is None:
            raise ValueError("Both vision_embeds and deepstack_visual_embeds are required")
        if kwargs.get("pixel_values") is not None:
            raise ValueError("Precomputed Qwen3-VL input must not also provide pixel_values")
        if kwargs.get("pixel_values_videos") is not None or kwargs.get("video_grid_thw") is not None:
            raise NotImplementedError("Precomputed Qwen3-VL vision features do not support video input")
        if model.vision_model is None:
            raise RuntimeError("Precomputed Qwen3-VL vision requires the first pipeline stage vision module")

        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            if not args:
                raise ValueError("Qwen3-VL forward requires input_ids")
            input_ids = args[0]

        image_grid_thw = kwargs.get("image_grid_thw")
        forward_inputs = {
            IMAGE_GRID_THW_KEY: image_grid_thw,
            VISION_EMBEDS_KEY: vision_embeds,
            **{
                deepstack_visual_embeds_key(index): feature
                for index, feature in enumerate(deepstack_visual_embeds)
            },
        }
        validated = build_qwen3_vl_precomputed_forward_kwargs(
            multimodal_train_inputs=forward_inputs,
            input_ids=input_ids,
            image_token_id=model.image_token_id,
            spatial_merge_size=getattr(
                model.vision_transformer_config,
                "spatial_merge_size",
                model.config.spatial_merge_size,
            ),
            hidden_size=model.config.hidden_size,
            expected_deepstack_count=expected_deepstack_count,
            context_parallel_size=model.pg_collection.cp.size(),
        )

        raw_patch_count = int(validated["image_grid_thw"].prod(dim=1).sum().item())
        dummy_pixel_values = torch.empty(
            (raw_patch_count, 1),
            dtype=validated["vision_embeds"].dtype,
            device=validated["vision_embeds"].device,
        )
        original_vision_forward = model.vision_model.forward

        def _return_precomputed(_vision_model, hidden_states, *, grid_thw, **_vision_kwargs):
            if hidden_states.shape[0] != raw_patch_count:
                raise ValueError(
                    "Qwen3-VL dummy patch count changed before the frozen vision boundary: "
                    f"expected {raw_patch_count}, got {hidden_states.shape[0]}"
                )
            if not torch.equal(grid_thw, validated["image_grid_thw"]):
                raise ValueError("Qwen3-VL image_grid_thw changed before the frozen vision boundary")
            return validated["vision_embeds"], validated["deepstack_visual_embeds"]

        model.vision_model.forward = MethodType(_return_precomputed, model.vision_model)
        try:
            kwargs["pixel_values"] = dummy_pixel_values
            return original_forward(*args, **kwargs)
        finally:
            model.vision_model.forward = original_vision_forward

    _forward_with_precomputed_vision._relax_precomputed_vision_forward = True
    _forward_with_precomputed_vision._relax_original_forward = original_forward
    model.forward = _forward_with_precomputed_vision


def patch_qwen3_vl_model_instance_for_precomputed_vision(model: Any) -> None:
    """Backward-compatible public name for the instance-level Bridge patch."""
    install_qwen3_vl_precomputed_vision_forward(model)
