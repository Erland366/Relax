# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from types import SimpleNamespace

import pytest
import torch


def test_model_instance_patch_accepts_real_batched_tensor_keys_and_bypasses_vision_forward():
    module = importlib.import_module("relax.backends.megatron.precomputed_vision")

    class VisionModel:
        def __init__(self):
            self.forward_calls = 0

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, hidden_states, grid_thw):
            self.forward_calls += 1
            raise AssertionError("the frozen GPU vision model must be bypassed")

    class Model:
        image_token_id = 99
        square_merge_size = 4

        def __init__(self):
            self.vision_model = VisionModel()
            self.config = SimpleNamespace(hidden_size=2, spatial_merge_size=2)
            self.vision_transformer_config = SimpleNamespace(
                spatial_merge_size=2,
                deepstack_visual_indexes=[1, 3, 5],
            )
            self.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1))

        def forward(self, *, input_ids, pixel_values=None, image_grid_thw=None, **kwargs):
            return self.vision_model(hidden_states=pixel_values, grid_thw=image_grid_thw)

    model = Model()
    module.install_qwen3_vl_precomputed_vision_forward(model)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    final = torch.ones((4, 2))
    deepstack = tuple(torch.full((4, 2), value) for value in (2.0, 3.0, 4.0))
    batched_vision_inputs = {
        "image_grid_thw": grid,
        "vision_embeds": final,
        "deepstack_visual_embeds_0": deepstack[0],
        "deepstack_visual_embeds_1": deepstack[1],
        "deepstack_visual_embeds_2": deepstack[2],
    }

    output = model.forward(
        input_ids=torch.tensor([[7, 99, 99, 99, 99, 8]]),
        **batched_vision_inputs,
    )

    assert model.vision_model.forward_calls == 0
    assert output[0] is final
    assert len(output[1]) == len(deepstack)
    assert all(actual is expected for actual, expected in zip(output[1], deepstack, strict=True))

    with pytest.raises(ValueError, match="image token count"):
        model.forward(
            input_ids=torch.tensor([[7, 99, 99, 99, 8]]),
            **batched_vision_inputs,
        )
    assert model.vision_model.forward_calls == 0
