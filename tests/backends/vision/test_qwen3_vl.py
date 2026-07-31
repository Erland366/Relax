# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib

import pytest
import torch


def _vision_module():
    return importlib.import_module("relax.backends.vision.qwen3_vl")


def _features():
    module = _vision_module()
    return module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int64),
        vision_embeds=torch.full((4, 2), 1.0),
        deepstack_visual_embeds=(
            torch.full((4, 2), 2.0),
            torch.full((4, 2), 3.0),
            torch.full((4, 2), 4.0),
        ),
    )


def test_qwen3_vl_frozen_features_validate_placeholder_and_deepstack_contract():
    features = _features()

    features.validate(
        spatial_merge_size=2,
        image_token_count=4,
        hidden_size=2,
        expected_deepstack_count=3,
    )

    with pytest.raises(ValueError, match="image token count"):
        features.validate(
            spatial_merge_size=2,
            image_token_count=3,
            hidden_size=2,
            expected_deepstack_count=3,
        )


def test_qwen3_vl_frozen_features_build_sglang_precomputed_payload_in_stream_order():
    module = _vision_module()

    image_data = module.build_sglang_precomputed_image_data(_features())

    assert image_data["format"] == "precomputed_embedding"
    torch.testing.assert_close(
        image_data["feature"],
        torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            ]
        ),
    )
    torch.testing.assert_close(image_data["image_grid_thw"], torch.tensor([[1, 4, 4]], dtype=torch.int64))

