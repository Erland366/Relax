# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def test_transformers_auto_patch_maps_feature_and_preserves_grid_for_mrope(monkeypatch):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class MultimodalInputFormat:
        PRECOMPUTED_EMBEDDING = object()

    item = SimpleNamespace(
        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
        feature=[[1.0, 2.0], [3.0, 4.0]],
        precomputed_embeddings=None,
        image_grid_thw=[[1, 2, 2]],
        model_specific_data={},
    )

    class BaseMultimodalProcessor:
        def collect_mm_items_from_processor_output(self, processor_output, modality=None):
            return processor_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            return [item], [1, 2], None

    schedule_batch_module = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch_module.MultimodalInputFormat = MultimodalInputFormat
    base_processor_module = types.ModuleType("sglang.srt.multimodal.processors.base_processor")
    base_processor_module.BaseMultimodalProcessor = BaseMultimodalProcessor
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", schedule_batch_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal.processors.base_processor", base_processor_module)

    module.install_qwen3_vl_precomputed_vision_patch()
    processor = BaseMultimodalProcessor()
    items, input_ids, processor_output = processor.process_and_combine_mm_data(object(), object())

    assert items == [item]
    assert input_ids == [1, 2]
    assert item.feature is None
    torch.testing.assert_close(
        item.precomputed_embeddings,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        item.model_specific_data["image_grid_thw"],
        torch.tensor([[1, 2, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(processor_output.image_grid_thw, torch.tensor([[1, 2, 2]], dtype=torch.int64))


def test_transformers_omit_mode_replaces_visual_before_materialization_and_skips_its_checkpoint_weights():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))

        def forward(self, hidden_states, grid_thw):
            return hidden_states, grid_thw

    class Qwen3VLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.language_model = torch.nn.Linear(2, 2, device="meta")

    class TransformersBase:
        def __init__(self, model):
            self.model = model
            self.skip_prefixes = []

        def load_weights(self, weights):
            assert "model.visual." in self.skip_prefixes
            return {
                name
                for name, _ in weights
                if not any(name.startswith(prefix) for prefix in self.skip_prefixes)
            }

    module.patch_qwen3_vl_transformers_for_omitted_vision(
        qwen3_vl_model_cls=Qwen3VLModel,
        transformers_base_cls=TransformersBase,
        enabled=True,
    )

    model = Qwen3VLModel()
    assert list(model.visual.parameters()) == []
    with pytest.raises(RuntimeError, match="omitted.*precomputed"):
        model.visual(
            hidden_states=torch.empty((4, 2)),
            grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.int64),
        )

    # SGLang constructs the HF model on meta, then recursively replaces and
    # materializes its children. The omission must already be visible here.
    visual_parameter_count_at_materialization = sum(
        parameter.numel() for parameter in model.visual.parameters()
    )
    assert visual_parameter_count_at_materialization == 0

    wrapper = TransformersBase(model)
    loaded = wrapper.load_weights(
        [
            ("model.visual.patch_embed.weight", torch.ones(1)),
            ("model.language_model.layers.0.weight", torch.ones(1)),
        ]
    )

    assert loaded == {"model.language_model.layers.0.weight"}


def test_transformers_resident_mode_leaves_visual_materialization_unchanged():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class Qwen3VLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2, device="meta")

    class TransformersBase:
        def __init__(self, model):
            self.model = model
            self.skip_prefixes = []

        def load_weights(self, weights):
            return {name for name, _ in weights}

    original_qwen_init = Qwen3VLModel.__init__
    original_transformers_load_weights = TransformersBase.load_weights
    module.patch_qwen3_vl_transformers_for_omitted_vision(
        qwen3_vl_model_cls=Qwen3VLModel,
        transformers_base_cls=TransformersBase,
        enabled=False,
    )

    model = Qwen3VLModel()
    wrapper = TransformersBase(model)
    loaded = wrapper.load_weights([("model.visual.weight", torch.ones(1))])

    assert Qwen3VLModel.__init__ is original_qwen_init
    assert TransformersBase.load_weights is original_transformers_load_weights
    assert list(model.visual.parameters())
    assert wrapper.skip_prefixes == []
    assert loaded == {"model.visual.weight"}
