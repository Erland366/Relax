# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _collect_precomputed_processor_output(module, processor_output):
    class BaseMultimodalProcessor:
        def collect_mm_items_from_processor_output(self, output, modality=None):
            return output

    module.patch_transformers_auto_precomputed_embedding_processor(BaseMultimodalProcessor)
    return BaseMultimodalProcessor().collect_mm_items_from_processor_output(
        processor_output,
        modality="image",
    )


def _packed_bf16_processor_output(feature_bits: torch.Tensor) -> dict:
    feature_bits = feature_bits.to(device="cpu", dtype=torch.uint16).contiguous()
    return {
        "format": "precomputed_embedding",
        "feature_b64": base64.b64encode(feature_bits.numpy().tobytes()).decode("ascii"),
        "feature_dtype": "bfloat16",
        "feature_shape": list(feature_bits.shape),
        "image_grid_thw": [[1, 2, 2]],
    }


def test_sglang_media_loader_passes_id_only_precomputed_payload_to_processor_cache():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    id_only_payload = {
        "format": "precomputed_embedding_id",
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
        "image_grid_thw": [[1, 2, 2]],
    }

    class BaseMultimodalProcessor:
        @classmethod
        def _load_single_item(cls, data, modality, **kwargs):
            raise ValueError(f"Invalid image: {data}")

    assert module.patch_sglang_precomputed_embedding_id_loader(BaseMultimodalProcessor)
    assert BaseMultimodalProcessor._load_single_item(id_only_payload, modality="image") is id_only_payload
    assert not module.patch_sglang_precomputed_embedding_id_loader(BaseMultimodalProcessor)


def test_sglang_media_loader_leaves_other_inputs_on_original_path():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    calls = []

    class BaseMultimodalProcessor:
        @classmethod
        def _load_single_item(cls, data, modality, **kwargs):
            calls.append((cls, data, modality, kwargs))
            return "original"

    module.patch_sglang_precomputed_embedding_id_loader(BaseMultimodalProcessor)

    assert BaseMultimodalProcessor._load_single_item("image.png", modality="image", discard_alpha_channel=True) == (
        "original"
    )
    assert calls == [(BaseMultimodalProcessor, "image.png", "image", {"discard_alpha_channel": True})]


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
        @classmethod
        def _load_single_item(cls, data, modality, **kwargs):
            return data

        def collect_mm_items_from_processor_output(self, processor_output, modality=None):
            return processor_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            return [item], [1, 2], None

    class MultiModalMixin:
        def _forward_hidden_states(self, input_ids, positions, forward_batch, input_embeds=None):
            return None

    schedule_batch_module = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch_module.MultimodalInputFormat = MultimodalInputFormat
    base_processor_module = types.ModuleType("sglang.srt.multimodal.processors.base_processor")
    base_processor_module.BaseMultimodalProcessor = BaseMultimodalProcessor
    transformers_module = types.ModuleType("sglang.srt.models.transformers")
    transformers_module.MultiModalMixin = MultiModalMixin
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", schedule_batch_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal.processors.base_processor", base_processor_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.models.transformers", transformers_module)

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


def test_install_patch_materializes_inline_bf16_before_original_process_reads_feature(monkeypatch):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    monkeypatch.setenv("RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "4096")
    feature_bits = torch.tensor(
        [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1, 0x8000, 0xBF80, 0xFFFF],
        dtype=torch.uint16,
    ).reshape(2, 4)
    organized_results = [
        (
            "image",
            {
                **_packed_bf16_processor_output(feature_bits),
                "feature_id": "feature-a",
                "vision_revision": "revision-a",
                "feature_schema_version": "schema-a",
            },
        )
    ]
    original_process_features = []

    class MultimodalInputFormat:
        PRECOMPUTED_EMBEDDING = object()

    class BaseMultimodalProcessor:
        @classmethod
        def _load_single_item(cls, data, modality, **kwargs):
            return data

        def collect_mm_items_from_processor_output(self, processor_output, modality=None):
            return processor_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            _, dict_item = base_output.organize_results()[0]
            feature = dict_item.pop("feature")
            original_process_features.append(feature)
            item = SimpleNamespace(
                format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                feature=feature,
                precomputed_embeddings=None,
                image_grid_thw=dict_item["image_grid_thw"],
                model_specific_data={},
            )
            return [item], [1, 2], None

    class MultiModalMixin:
        def _forward_hidden_states(self, input_ids, positions, forward_batch, input_embeds=None):
            return None

    schedule_batch_module = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch_module.MultimodalInputFormat = MultimodalInputFormat
    base_processor_module = types.ModuleType("sglang.srt.multimodal.processors.base_processor")
    base_processor_module.BaseMultimodalProcessor = BaseMultimodalProcessor
    transformers_module = types.ModuleType("sglang.srt.models.transformers")
    transformers_module.MultiModalMixin = MultiModalMixin
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", schedule_batch_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal.processors.base_processor", base_processor_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.models.transformers", transformers_module)

    module.install_qwen3_vl_precomputed_vision_patch()
    base_output = SimpleNamespace(organize_results=lambda: organized_results)

    items, _, _ = BaseMultimodalProcessor().process_and_combine_mm_data(base_output, object())
    organized_results[0] = (
        "image",
        {
            "format": "precomputed_embedding_id",
            "feature_id": "feature-a",
            "vision_revision": "revision-a",
            "feature_schema_version": "schema-a",
            "image_grid_thw": [[1, 2, 2]],
        },
    )
    cached_items, _, _ = BaseMultimodalProcessor().process_and_combine_mm_data(base_output, object())

    assert len(original_process_features) == 2
    assert original_process_features[0].dtype == torch.bfloat16
    assert torch.equal(original_process_features[0].view(torch.uint16), feature_bits)
    assert torch.equal(original_process_features[1].view(torch.uint16), feature_bits)
    assert torch.equal(items[0].precomputed_embeddings.view(torch.uint16), feature_bits)
    assert torch.equal(cached_items[0].precomputed_embeddings.view(torch.uint16), feature_bits)


def test_install_patch_leaves_native_media_output_untouched(monkeypatch):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class NativeImage:
        pass

    native_image = NativeImage()
    organized_results = [("image", native_image)]
    original_calls = []

    class MultimodalInputFormat:
        PRECOMPUTED_EMBEDDING = object()
        RAW_IMAGE = object()

    raw_item = SimpleNamespace(format=MultimodalInputFormat.RAW_IMAGE)

    class BaseMultimodalProcessor:
        @classmethod
        def _load_single_item(cls, data, modality, **kwargs):
            return data

        def collect_mm_items_from_processor_output(self, processor_output, modality=None):
            return processor_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            original_calls.append((base_output, mm_tokens, kwargs))
            return [raw_item], [1, 2], native_image

    class MultiModalMixin:
        def _forward_hidden_states(self, input_ids, positions, forward_batch, input_embeds=None):
            return None

    schedule_batch_module = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch_module.MultimodalInputFormat = MultimodalInputFormat
    base_processor_module = types.ModuleType("sglang.srt.multimodal.processors.base_processor")
    base_processor_module.BaseMultimodalProcessor = BaseMultimodalProcessor
    transformers_module = types.ModuleType("sglang.srt.models.transformers")
    transformers_module.MultiModalMixin = MultiModalMixin
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", schedule_batch_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal.processors.base_processor", base_processor_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.models.transformers", transformers_module)

    module.install_qwen3_vl_precomputed_vision_patch()
    base_output = SimpleNamespace(organize_results=lambda: organized_results)
    mm_tokens = object()

    result = BaseMultimodalProcessor().process_and_combine_mm_data(base_output, mm_tokens, request_id="native")

    assert result == ([raw_item], [1, 2], native_image)
    assert original_calls == [(base_output, mm_tokens, {"request_id": "native"})]
    assert organized_results == [("image", native_image)]


def test_transformers_auto_patch_decodes_inline_bf16_without_changing_bits():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    feature_bits = torch.tensor(
        [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1, 0x8000, 0xBF80, 0xFFFF],
        dtype=torch.uint16,
    ).reshape(2, 4)

    collected = _collect_precomputed_processor_output(
        module,
        _packed_bf16_processor_output(feature_bits),
    )

    assert collected["precomputed_embeddings"].dtype == torch.bfloat16
    assert tuple(collected["precomputed_embeddings"].shape) == (2, 4)
    assert torch.equal(collected["precomputed_embeddings"].view(torch.uint16), feature_bits)
    torch.testing.assert_close(
        collected["image_grid_thw"],
        torch.tensor([[1, 2, 2]], dtype=torch.int64),
    )
    assert "feature_b64" not in collected
    assert "feature_dtype" not in collected
    assert "feature_shape" not in collected


def test_sglang_vision_feature_cache_disabled_preserves_legacy_inline_payload():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=0)
    inline_payload = _packed_bf16_processor_output(
        torch.tensor([[0x3F80, 0x4000]], dtype=torch.uint16)
    )

    resolved = cache.resolve(inline_payload)

    assert resolved == inline_payload


def test_sglang_vision_feature_cache_publishes_inline_feature_for_id_only_lookup():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=4096)
    feature_bits = torch.tensor(
        [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1, 0x8000, 0xBF80, 0xFFFF],
        dtype=torch.uint16,
    ).reshape(2, 4)
    inline_payload = {
        **_packed_bf16_processor_output(feature_bits),
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
    }

    assert cache.resolve(inline_payload) == inline_payload
    resolved = cache.resolve(
        {
            "format": "precomputed_embedding_id",
            "feature_id": "feature-a",
            "vision_revision": "vision-revision-a",
            "feature_schema_version": "qwen3-vl-frozen-vision-v1",
            "image_grid_thw": [[1, 2, 2]],
        }
    )

    assert resolved["format"] == "precomputed_embedding"
    assert resolved["feature_id"] == "feature-a"
    assert resolved["vision_revision"] == "vision-revision-a"
    assert resolved["feature_schema_version"] == "qwen3-vl-frozen-vision-v1"
    assert resolved["image_grid_thw"] == [[1, 2, 2]]
    assert resolved["feature"].dtype == torch.bfloat16
    assert torch.equal(resolved["feature"].view(torch.uint16), feature_bits)


def test_transformers_processor_patch_publishes_inline_and_resolves_id_only_from_host_cache(monkeypatch):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    monkeypatch.setenv("RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "4096")
    feature_bits = torch.tensor(
        [0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1, 0x8000, 0xBF80, 0xFFFF],
        dtype=torch.uint16,
    ).reshape(2, 4)
    inline_payload = {
        **_packed_bf16_processor_output(feature_bits),
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
    }
    id_only_payload = {
        "format": "precomputed_embedding_id",
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
        "image_grid_thw": [[1, 2, 2]],
    }

    class BaseMultimodalProcessor:
        def collect_mm_items_from_processor_output(self, processor_output, modality=None):
            return processor_output

    module.patch_transformers_auto_precomputed_embedding_processor(BaseMultimodalProcessor)
    processor = BaseMultimodalProcessor()

    published = processor.collect_mm_items_from_processor_output(inline_payload, modality="image")
    resolved = processor.collect_mm_items_from_processor_output(id_only_payload, modality="image")

    assert torch.equal(published["precomputed_embeddings"].view(torch.uint16), feature_bits)
    assert torch.equal(resolved["precomputed_embeddings"].view(torch.uint16), feature_bits)
    torch.testing.assert_close(
        resolved["image_grid_thw"],
        torch.tensor([[1, 2, 2]], dtype=torch.int64),
    )


def test_sglang_vision_feature_cache_raises_typed_miss_for_unknown_identity():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=4096)

    with pytest.raises(module.SGLangVisionFeatureCacheMiss) as raised:
        cache.resolve(
            {
                "format": "precomputed_embedding_id",
                "feature_id": "missing-feature",
                "vision_revision": "vision-revision-a",
                "feature_schema_version": "qwen3-vl-frozen-vision-v1",
                "image_grid_thw": [[1, 2, 2]],
            }
        )

    assert raised.value.feature_id == "missing-feature"
    assert raised.value.vision_revision == "vision-revision-a"
    assert raised.value.feature_schema_version == "qwen3-vl-frozen-vision-v1"


@pytest.mark.parametrize(
    ("identity_update", "missing_value"),
    [
        ({"vision_revision": "vision-revision-b"}, "vision-revision-b"),
        ({"feature_schema_version": "qwen3-vl-frozen-vision-v2"}, "qwen3-vl-frozen-vision-v2"),
    ],
    ids=["revision", "schema"],
)
def test_sglang_vision_feature_cache_does_not_reuse_mismatched_identity(identity_update, missing_value):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=4096)
    inline_payload = {
        **_packed_bf16_processor_output(torch.tensor([[0x3F80, 0x4000]], dtype=torch.uint16)),
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
    }
    cache.resolve(inline_payload)
    lookup = {
        "format": "precomputed_embedding_id",
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
        "image_grid_thw": [[1, 2, 2]],
        **identity_update,
    }

    with pytest.raises(module.SGLangVisionFeatureCacheMiss, match=missing_value):
        cache.resolve(lookup)


@pytest.mark.parametrize(
    "missing_field",
    ["feature_id", "vision_revision", "feature_schema_version"],
)
def test_sglang_vision_feature_cache_rejects_incomplete_inline_identity(missing_field):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=4096)
    payload = {
        **_packed_bf16_processor_output(torch.tensor([[0x3F80, 0x4000]], dtype=torch.uint16)),
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
    }
    payload.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        cache.resolve(payload)


@pytest.mark.parametrize(
    "invalid_grid",
    [[], [[1, 2]], [[1, 0, 2]], [[1, 2, 2, 2]]],
)
def test_sglang_vision_feature_cache_rejects_invalid_grid_before_admission(invalid_grid):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    cache = module.SGLangVisionFeatureCache(max_bytes=4096)
    payload = {
        **_packed_bf16_processor_output(torch.tensor([[0x3F80, 0x4000]], dtype=torch.uint16)),
        "feature_id": "feature-a",
        "vision_revision": "vision-revision-a",
        "feature_schema_version": "qwen3-vl-frozen-vision-v1",
        "image_grid_thw": invalid_grid,
    }

    with pytest.raises(ValueError, match="image_grid_thw"):
        cache.resolve(payload)


@pytest.mark.parametrize(
    ("payload_update", "error_match"),
    [
        ({"feature_b64": "not-valid-base64!"}, "feature_b64.*valid base64"),
        ({"feature_dtype": "float32"}, "feature_dtype.*bfloat16"),
        ({"feature_shape": [2, 3]}, "byte count.*feature_shape"),
    ],
    ids=["invalid-base64", "unsupported-dtype", "byte-count-shape-mismatch"],
)
def test_transformers_auto_patch_rejects_invalid_inline_bf16_payload(payload_update, error_match):
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    payload = _packed_bf16_processor_output(torch.tensor([[0x3F80, 0x4000]], dtype=torch.uint16))
    payload.update(payload_update)

    with pytest.raises(ValueError, match=error_match):
        _collect_precomputed_processor_output(module, payload)


def test_transformers_forward_patch_consumes_packed_qwen3_vl_deepstack_features():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class ForwardMode:
        @staticmethod
        def is_decode():
            return False

    class Item:
        precomputed_embeddings = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5],
            ],
            dtype=torch.bfloat16,
        )
        offsets = [(1, 2)]

    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode(),
        mm_inputs=[SimpleNamespace(mm_items=[Item()])],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[4],
        mrope_positions=torch.arange(12).reshape(3, 4),
    )
    forward_batch.contains_mm_inputs = lambda: True

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return (kwargs["inputs_embeds"] + 10,)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(32, 2, dtype=torch.bfloat16)
            torch.nn.init.zeros_(self.embed_tokens.weight)
            self.language_model = LanguageModel()
            self.config = SimpleNamespace(
                model_type="qwen3_vl",
                image_token_id=7,
                vision_config=SimpleNamespace(deepstack_visual_indexes=[1, 2, 3]),
            )

        def get_input_embeddings(self):
            return self.embed_tokens

    class TransformersMixin:
        def _forward_hidden_states(self, input_ids, positions, forward_batch, input_embeds=None):
            return "unpatched"

    module.patch_qwen3_vl_transformers_precomputed_forward(TransformersMixin)

    wrapper = TransformersMixin()
    wrapper.model = Model()
    wrapper.text_config = SimpleNamespace(hidden_size=2)
    wrapper.attention_instances = {0: object()}
    wrapper._format_position_ids = lambda positions: positions[:, None, :]

    hidden_states = wrapper._forward_hidden_states(
        input_ids=torch.tensor([10, 7, 7, 11]),
        positions=torch.arange(4),
        forward_batch=forward_batch,
    )

    kwargs = wrapper.model.language_model.kwargs
    assert tuple(hidden_states.shape) == (4, 2)
    torch.testing.assert_close(
        kwargs["inputs_embeds"][0],
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.5, 1.5], [0.0, 0.0]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        kwargs["visual_pos_masks"],
        torch.tensor([[False, True, True, False]]),
    )
    assert len(kwargs["deepstack_visual_embeds"]) == 3
    torch.testing.assert_close(
        kwargs["deepstack_visual_embeds"][0],
        torch.tensor([[2.0, 2.0], [2.5, 2.5]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        kwargs["deepstack_visual_embeds"][1],
        torch.tensor([[3.0, 3.0], [3.5, 3.5]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        kwargs["deepstack_visual_embeds"][2],
        torch.tensor([[4.0, 4.0], [4.5, 4.5]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(kwargs["position_ids"], forward_batch.mrope_positions[:, None, :])


def test_transformers_forward_patch_preserves_native_and_decode_paths():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")

    class TransformersMixin:
        def _forward_hidden_states(self, input_ids, positions, forward_batch, input_embeds=None):
            return "original"

    module.patch_qwen3_vl_transformers_precomputed_forward(TransformersMixin)
    wrapper = TransformersMixin()

    native_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        mm_inputs=[SimpleNamespace(mm_items=[SimpleNamespace(precomputed_embeddings=None)])],
    )
    native_batch.contains_mm_inputs = lambda: True
    decode_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        mm_inputs=None,
    )
    decode_batch.contains_mm_inputs = lambda: False

    assert wrapper._forward_hidden_states(None, None, native_batch) == "original"
    assert wrapper._forward_hidden_states(None, None, decode_batch) == "original"


def test_transformers_forward_patch_slices_precomputed_features_for_chunked_prefill():
    module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    packed = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5],
        ],
        dtype=torch.bfloat16,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[2],
        extend_seq_lens_cpu=[2],
    )
    wrapper = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2),
        model=SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(deepstack_visual_indexes=[1, 2, 3])),
            get_input_embeddings=lambda: torch.nn.Embedding(32, 2, dtype=torch.bfloat16),
        ),
    )

    input_embeds, visual_mask, deepstack = module._build_qwen3_vl_precomputed_language_inputs(
        wrapper=wrapper,
        input_ids=torch.tensor([7, 11]),
        forward_batch=forward_batch,
        request_items=[[SimpleNamespace(precomputed_embeddings=packed, offsets=[(1, 2)])]],
    )

    torch.testing.assert_close(input_embeds[0, 0], packed[1, :2])
    torch.testing.assert_close(visual_mask, torch.tensor([[True, False]]))
    torch.testing.assert_close(deepstack[0], packed[1:2, 2:4])
    torch.testing.assert_close(deepstack[1], packed[1:2, 4:6])
    torch.testing.assert_close(deepstack[2], packed[1:2, 6:8])


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
            return {name for name, _ in weights if not any(name.startswith(prefix) for prefix in self.skip_prefixes)}

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
    visual_parameter_count_at_materialization = sum(parameter.numel() for parameter in model.visual.parameters())
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
