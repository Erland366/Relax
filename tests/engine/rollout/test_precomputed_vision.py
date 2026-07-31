# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib

import torch


def test_rollout_replaces_raw_image_payload_and_preserves_actor_feature_streams():
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    rollout_module = importlib.import_module("relax.engine.rollout.precomputed_vision")
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    main = torch.full((4, 2), 1.0)
    deepstack = tuple(torch.full((4, 2), float(index)) for index in (2, 3, 4))
    features = vision_module.Qwen3VLFrozenVisionFeatures(
        feature_id="feature-id",
        vision_revision="vision-revision",
        image_grid_thw=grid,
        vision_embeds=main,
        deepstack_visual_embeds=deepstack,
    )
    payload = {"input_ids": [10, 11], "image_data": ["raw-base64-image"]}
    processor_marker = torch.tensor([17])
    multimodal_train_inputs = {
        "pixel_values": torch.ones((16, 8)),
        "image_grid_thw": grid,
        "processor_marker": processor_marker,
    }

    rollout_payload, actor_inputs = rollout_module.prepare_qwen3_vl_precomputed_rollout_inputs(
        payload=payload,
        multimodal_train_inputs=multimodal_train_inputs,
        features=features,
    )

    assert len(rollout_payload["image_data"]) == 1
    assert rollout_payload["image_data"][0]["format"] == "precomputed_embedding"
    assert rollout_payload["image_data"][0]["feature_id"] == "feature-id"
    assert rollout_payload["image_data"][0]["vision_revision"] == "vision-revision"
    torch.testing.assert_close(
        rollout_payload["image_data"][0]["feature"],
        torch.cat((main, *deepstack), dim=-1),
    )
    serialized_image_data = rollout_module.serialize_sglang_precomputed_image_data(
        rollout_payload["image_data"][0]
    )
    assert serialized_image_data["feature_id"] == "feature-id"
    assert serialized_image_data["vision_revision"] == "vision-revision"
    assert "pixel_values" not in actor_inputs
    assert "feature_id" not in actor_inputs
    assert "vision_revision" not in actor_inputs
    assert all(isinstance(value, torch.Tensor) for value in actor_inputs.values())
    assert actor_inputs["image_grid_thw"] is grid
    assert actor_inputs["vision_embeds"] is main
    assert actor_inputs["deepstack_visual_embeds_0"] is deepstack[0]
    assert actor_inputs["deepstack_visual_embeds_1"] is deepstack[1]
    assert actor_inputs["deepstack_visual_embeds_2"] is deepstack[2]
    assert actor_inputs["processor_marker"] is processor_marker
