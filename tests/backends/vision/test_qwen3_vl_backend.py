# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from dataclasses import FrozenInstanceError

import pytest
import torch


def _vision_module():
    return importlib.import_module("relax.backends.vision.qwen3_vl")


def _features():
    module = _vision_module()
    return module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int64),
        vision_embeds=torch.ones((4, 2), dtype=torch.bfloat16),
        deepstack_visual_embeds=tuple(
            torch.full((4, 2), value, dtype=torch.bfloat16) for value in (2.0, 3.0, 4.0)
        ),
    )


def test_load_visual_state_dict_reads_only_qwen3_vl_visual_tensors(monkeypatch, tmp_path):
    module = _vision_module()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    requested = []
    tensors = {
        "model.visual.patch_embed.proj.weight": torch.tensor([1.0]),
        "model.visual.blocks.0.norm1.weight": torch.tensor([2.0]),
        "model.language_model.layers.0.weight": torch.tensor([3.0]),
        "lm_head.weight": torch.tensor([4.0]),
    }

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def keys(self):
            return tensors.keys()

        def get_tensor(self, name):
            requested.append(name)
            return tensors[name]

    def safe_open(path, *, framework, device):
        assert path == str(checkpoint)
        assert framework == "pt"
        assert device == "cpu"
        return FakeSafeOpen()

    safetensors_module = types.ModuleType("safetensors")
    safetensors_module.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)

    state_dict = module.load_qwen3_vl_visual_state_dict(tmp_path)

    assert state_dict == {
        "patch_embed.proj.weight": tensors["model.visual.patch_embed.proj.weight"],
        "blocks.0.norm1.weight": tensors["model.visual.blocks.0.norm1.weight"],
    }
    assert requested == [
        "model.visual.patch_embed.proj.weight",
        "model.visual.blocks.0.norm1.weight",
    ]


def test_cpu_backend_encodes_one_image_into_final_and_deepstack_features():
    module = _vision_module()

    class FakeVisualModel:
        def __init__(self):
            self.eval_calls = 0
            self.calls = []

        def eval(self):
            self.eval_calls += 1
            return self

        def __call__(self, pixel_values, *, grid_thw):
            self.calls.append((pixel_values, grid_thw))
            final = pixel_values[:4]
            return final, [final + 1, final + 2, final + 3]

    visual_model = FakeVisualModel()
    backend = module.Qwen3VLCPUVisionBackend(visual_model=visual_model, revision="vision-rev")
    pixel_values = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)

    features = backend.encode(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert visual_model.eval_calls == 1
    assert visual_model.calls == [(pixel_values, image_grid_thw)]
    assert features.image_grid_thw is image_grid_thw
    torch.testing.assert_close(features.vision_embeds, pixel_values[:4])
    assert len(features.deepstack_visual_embeds) == 3
    torch.testing.assert_close(features.deepstack_visual_embeds[2], pixel_values[:4] + 3)
    assert features.feature_id == module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision="vision-rev",
        output_dtype=torch.bfloat16,
    )
    assert features.vision_revision == "vision-rev"
    with pytest.raises(FrozenInstanceError):
        features.feature_id = "replacement-feature-id"
    with pytest.raises(FrozenInstanceError):
        features.vision_revision = "replacement-vision-revision"


def test_visual_revision_and_cache_key_are_deterministic_and_content_addressed():
    module = _vision_module()
    state_a = {
        "blocks.0.weight": torch.tensor([1.0, 2.0]),
        "patch_embed.weight": torch.tensor([3.0]),
    }
    state_b = {
        "patch_embed.weight": state_a["patch_embed.weight"].clone(),
        "blocks.0.weight": state_a["blocks.0.weight"].clone(),
    }

    revision_a = module.compute_qwen3_vl_visual_revision(state_a, config_fingerprint="processor-v1")
    revision_b = module.compute_qwen3_vl_visual_revision(state_b, config_fingerprint="processor-v1")
    changed_revision = module.compute_qwen3_vl_visual_revision(
        {**state_b, "patch_embed.weight": torch.tensor([4.0])},
        config_fingerprint="processor-v1",
    )

    assert revision_a == revision_b
    assert revision_a != changed_revision

    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    cache_key = module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision=revision_a,
        output_dtype=torch.bfloat16,
    )
    assert cache_key == module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values.clone(),
        image_grid_thw=image_grid_thw.clone(),
        vision_revision=revision_a,
        output_dtype=torch.bfloat16,
    )
    assert cache_key != module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values + 1,
        image_grid_thw=image_grid_thw,
        vision_revision=revision_a,
        output_dtype=torch.bfloat16,
    )
    assert cache_key != module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 1, 4]], dtype=torch.int64),
        vision_revision=revision_a,
        output_dtype=torch.bfloat16,
    )
    assert cache_key != module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision=f"{revision_a}-replacement",
        output_dtype=torch.bfloat16,
    )
    assert cache_key != module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision=revision_a,
        output_dtype=torch.float32,
    )


def test_feature_cache_key_invalidates_when_schema_version_changes():
    module = _vision_module()
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)

    schema_v1_key = module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision="vision-revision",
        feature_schema_version="qwen3-vl-frozen-vision-v1",
        output_dtype=torch.bfloat16,
    )
    schema_v2_key = module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision="vision-revision",
        feature_schema_version="qwen3-vl-frozen-vision-v2",
        output_dtype=torch.bfloat16,
    )

    assert schema_v1_key != schema_v2_key


def test_frozen_feature_byte_size_counts_all_tensor_storage():
    features = _features()

    expected = features.image_grid_thw.numel() * features.image_grid_thw.element_size()
    expected += features.vision_embeds.numel() * features.vision_embeds.element_size()
    expected += sum(
        tensor.numel() * tensor.element_size() for tensor in features.deepstack_visual_embeds
    )

    assert features.nbytes == expected
