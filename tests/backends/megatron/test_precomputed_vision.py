# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib

import pytest
import torch


IMAGE_TOKEN_ID = 99


def _actor_inputs() -> dict[str, torch.Tensor]:
    return {
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
        "vision_embeds": torch.full((4, 2), 1.0),
        "deepstack_visual_embeds_0": torch.full((4, 2), 2.0),
        "deepstack_visual_embeds_1": torch.full((4, 2), 3.0),
        "deepstack_visual_embeds_2": torch.full((4, 2), 4.0),
    }


def _build_forward_kwargs(multimodal_train_inputs, *, context_parallel_size=1, image_token_count=4):
    module = importlib.import_module("relax.backends.megatron.precomputed_vision")
    return module.build_qwen3_vl_precomputed_forward_kwargs(
        multimodal_train_inputs=multimodal_train_inputs,
        input_ids=torch.tensor([[7, *([IMAGE_TOKEN_ID] * image_token_count), 8]]),
        image_token_id=IMAGE_TOKEN_ID,
        spatial_merge_size=2,
        hidden_size=2,
        expected_deepstack_count=3,
        context_parallel_size=context_parallel_size,
    )


def test_megatron_adapter_validates_and_supplies_final_and_deepstack_streams():
    actor_inputs = _actor_inputs()

    forward_kwargs = _build_forward_kwargs(actor_inputs)

    assert forward_kwargs["image_grid_thw"] is actor_inputs["image_grid_thw"]
    assert forward_kwargs["vision_embeds"] is actor_inputs["vision_embeds"]
    assert forward_kwargs["deepstack_visual_embeds"] == (
        actor_inputs["deepstack_visual_embeds_0"],
        actor_inputs["deepstack_visual_embeds_1"],
        actor_inputs["deepstack_visual_embeds_2"],
    )


def test_actor_identity_round_trip_keeps_megatron_batch_and_model_kwargs_tensor_only():
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")

    def actor_inputs(marker):
        features = vision_module.Qwen3VLFrozenVisionFeatures(
            feature_id=f"{marker:064x}",
            vision_revision="a" * 64,
            image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int64),
            vision_embeds=torch.full((4, 2), float(marker)),
            deepstack_visual_embeds=tuple(
                torch.full((4, 2), float(marker + offset)) for offset in (1, 2, 3)
            ),
        )
        packaged = vision_module.build_qwen3_vl_actor_feature_inputs(features)
        assert "feature_id" not in packaged
        assert "vision_revision" not in packaged
        assert all(isinstance(value, torch.Tensor) for value in packaged.values())
        return packaged

    per_sample_inputs = [actor_inputs(1), actor_inputs(2)]
    batched_inputs = {
        key: torch.cat([sample[key] for sample in per_sample_inputs], dim=0)
        for key in per_sample_inputs[0]
    }

    forward_kwargs = _build_forward_kwargs(batched_inputs, image_token_count=8)

    assert set(forward_kwargs) == {
        "image_grid_thw",
        "vision_embeds",
        "deepstack_visual_embeds",
    }
    assert all(
        isinstance(value, torch.Tensor)
        for key, value in forward_kwargs.items()
        if key != "deepstack_visual_embeds"
    )
    assert all(isinstance(value, torch.Tensor) for value in forward_kwargs["deepstack_visual_embeds"])


def test_megatron_adapter_fails_loudly_for_context_parallel_input():
    with pytest.raises(NotImplementedError, match="context parallel"):
        _build_forward_kwargs(_actor_inputs(), context_parallel_size=2)


def test_megatron_adapter_fails_loudly_for_video_input():
    actor_inputs = _actor_inputs()
    actor_inputs["video_grid_thw"] = torch.tensor([[1, 4, 4]], dtype=torch.int64)

    with pytest.raises(NotImplementedError, match="video"):
        _build_forward_kwargs(actor_inputs)
