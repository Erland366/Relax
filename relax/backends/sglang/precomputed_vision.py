# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang compatibility for Qwen3-VL precomputed visual embeddings."""

import base64
import binascii
import json
import math
import os
from types import SimpleNamespace
from typing import Any

import torch

from relax.backends.vision.cache import ByteBoundedLRUCache


SGLANG_VISION_FEATURE_CACHE_MISS_MARKER = "RELAX_SGLANG_VISION_FEATURE_CACHE_MISS:"


class SGLangVisionFeatureCacheMiss(ValueError):
    """An ID-only precomputed request referenced an unavailable feature."""

    def __init__(self, *, feature_id: str, vision_revision: str, feature_schema_version: str) -> None:
        self.feature_id = feature_id
        self.vision_revision = vision_revision
        self.feature_schema_version = feature_schema_version
        super().__init__(
            SGLANG_VISION_FEATURE_CACHE_MISS_MARKER
            + json.dumps(
                {
                    "feature_id": feature_id,
                    "vision_revision": vision_revision,
                    "feature_schema_version": feature_schema_version,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


class SGLangVisionFeatureCache:
    """Byte-bounded host cache for immutable inline precomputed features."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        self._cache = None if max_bytes == 0 else ByteBoundedLRUCache(max_bytes=max_bytes)

    @staticmethod
    def _identity(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            payload.get("feature_id", ""),
            payload.get("vision_revision", ""),
            payload.get("feature_schema_version", ""),
        )

    @classmethod
    def _validated_identity(cls, payload: dict[str, Any]) -> tuple[str, str, str]:
        identity = cls._identity(payload)
        field_names = ("feature_id", "vision_revision", "feature_schema_version")
        for field_name, value in zip(field_names, identity, strict=True):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"SGLang cached vision feature requires a non-empty {field_name}, got {value!r}"
                )
        return identity

    @staticmethod
    def _validated_grid(payload: dict[str, Any]) -> torch.Tensor:
        raw_grid = payload.get("image_grid_thw")
        try:
            grid = torch.as_tensor(raw_grid)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"SGLang cached vision feature has invalid image_grid_thw: {raw_grid!r}") from error
        if (
            grid.dtype == torch.bool
            or grid.is_floating_point()
            or grid.is_complex()
            or grid.ndim != 2
            or grid.shape[0] == 0
            or grid.shape[1] != 3
            or bool((grid <= 0).any())
        ):
            raise ValueError(
                "SGLang cached vision feature image_grid_thw must have positive shape [images, 3], "
                f"got {raw_grid!r}"
            )
        return grid.to(dtype=torch.int64)

    def _miss(self, payload: dict[str, Any]) -> SGLangVisionFeatureCacheMiss:
        feature_id, vision_revision, feature_schema_version = self._identity(payload)
        return SGLangVisionFeatureCacheMiss(
            feature_id=feature_id,
            vision_revision=vision_revision,
            feature_schema_version=feature_schema_version,
        )

    def _publish_inline(self, payload: dict[str, Any]) -> int:
        identity = self._validated_identity(payload)
        self._validated_grid(payload)
        cached_payload = dict(payload)
        feature = _materialize_inline_bf16_feature(cached_payload)
        if feature is None:
            feature = torch.as_tensor(cached_payload["feature"], dtype=torch.bfloat16)
        if feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[1] == 0:
            raise ValueError(
                "SGLang cached vision feature must be a non-empty rank-2 tensor, "
                f"got shape {tuple(feature.shape)}"
            )
        size_bytes = feature.numel() * feature.element_size()
        cached_payload["feature"] = feature
        self._cache.put(identity, cached_payload, size_bytes=size_bytes)
        return size_bytes

    def cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Cache one inline feature without running SGLang generation."""
        if self._cache is None:
            raise RuntimeError("SGLang vision feature cache is disabled")
        if payload.get("format") != "precomputed_embedding":
            raise ValueError("Caching a vision feature requires an inline feature payload")

        image_grid_thw = self._validated_grid(payload).tolist()
        cached_bytes = self._publish_inline(payload)
        if cached_bytes > self._cache.max_bytes:
            raise ValueError(
                "SGLang cached vision feature exceeds cache capacity: "
                f"feature_bytes={cached_bytes}, max_bytes={self._cache.max_bytes}"
            )
        feature_id, vision_revision, feature_schema_version = self._identity(payload)
        stats = self._cache.stats
        return {
            "feature_id": feature_id,
            "vision_revision": vision_revision,
            "feature_schema_version": feature_schema_version,
            "image_grid_thw": image_grid_thw,
            "entries": stats["entries"],
            "resident_bytes": stats["resident_bytes"],
            "evictions": stats["evictions"],
            "cached_bytes": cached_bytes,
        }

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Publish an inline feature or resolve an ID-only request."""
        if self._cache is None:
            return payload

        payload_format = payload.get("format")
        if payload_format == "precomputed_embedding":
            self._publish_inline(payload)
            return payload

        if payload_format != "precomputed_embedding_id":
            return payload

        identity = self._validated_identity(payload)
        requested_grid_tensor = self._validated_grid(payload)
        cached_payload = self._cache.get(identity)
        if cached_payload is None:
            raise self._miss(payload)
        requested_grid = payload.get("image_grid_thw")
        cached_grid = cached_payload.get("image_grid_thw")
        if cached_grid is not None and not torch.equal(
            requested_grid_tensor,
            torch.as_tensor(cached_grid, dtype=torch.int64),
        ):
            raise ValueError(
                "SGLang vision feature cache lookup image_grid_thw does not match the published feature: "
                f"requested={requested_grid!r}, published={cached_grid!r}"
            )
        resolved = dict(cached_payload)
        resolved["image_grid_thw"] = requested_grid
        return resolved


def install_sglang_vision_feature_cache_route(app, processor_cls) -> bool:
    """Install the cache-only endpoint on SGLang's HTTP FastAPI app."""
    marker = "_relax_vision_feature_cache_route_installed"
    if getattr(app, marker, False):
        return False

    from fastapi import HTTPException

    @app.post("/relax/vision-features/cache")
    async def _cache_vision_feature(payload: dict[str, Any]):
        try:
            return processor_cls._relax_sglang_vision_feature_cache.cache(payload)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    setattr(app, marker, True)
    return True


class Qwen3VLPrecomputedOnlyVisual(torch.nn.Module):
    """Parameterless guard for raw-image requests when the GPU encoder is skipped."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen3-VL GPU vision encoder was skipped; provide precomputed vision features"
        )


def _get_qwen3_vl_precomputed_items(forward_batch: Any) -> list[list[Any]] | None:
    if forward_batch.forward_mode.is_decode() or not forward_batch.contains_mm_inputs():
        return None

    request_items = []
    has_precomputed = False
    has_raw_media = False
    for mm_input in forward_batch.mm_inputs or []:
        items = [] if mm_input is None else [item for item in mm_input.mm_items or [] if item is not None]
        request_items.append(items)
        for item in items:
            if getattr(item, "precomputed_embeddings", None) is None:
                has_raw_media = True
            else:
                has_precomputed = True

    if not has_precomputed:
        return None
    if has_raw_media:
        raise RuntimeError("Qwen3-VL SGLang transformers batches cannot mix GPU-computed and precomputed vision items")
    return request_items


def _build_qwen3_vl_precomputed_language_inputs(
    *,
    wrapper: Any,
    input_ids: torch.Tensor,
    forward_batch: Any,
    request_items: list[list[Any]],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    if input_ids is None:
        raise ValueError("Qwen3-VL precomputed vision requires input_ids during SGLang prefill")

    hidden_size = int(wrapper.text_config.hidden_size)
    vision_config = getattr(wrapper.model.config, "vision_config", None)
    deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
    if deepstack_indexes is None:
        raise ValueError("Qwen3-VL config is missing vision_config.deepstack_visual_indexes")
    stream_count = 1 + len(deepstack_indexes)
    packed_width = hidden_size * stream_count

    seq_lens = list(forward_batch.extend_seq_lens_cpu)
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu)
    if len(request_items) != len(seq_lens) or len(request_items) != len(prefix_lens):
        raise ValueError(
            "Qwen3-VL precomputed request metadata has inconsistent batch lengths: "
            f"mm_inputs={len(request_items)}, prefix_lens={len(prefix_lens)}, seq_lens={len(seq_lens)}"
        )

    visual_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    packed_chunks = []
    visual_token_count = 0
    request_start = 0
    for items, prefix_len, seq_len in zip(request_items, prefix_lens, seq_lens):
        request_end = request_start + seq_len
        if request_end > input_ids.shape[0]:
            raise ValueError(
                "Qwen3-VL precomputed request lengths exceed the flattened SGLang input: "
                f"request_end={request_end}, input_tokens={input_ids.shape[0]}"
            )
        chunk_end = prefix_len + seq_len
        for item in items:
            packed = torch.as_tensor(item.precomputed_embeddings)
            if packed.ndim != 2 or packed.shape[1] != packed_width:
                raise ValueError(
                    "Qwen3-VL precomputed embedding must pack final and DeepStack streams along the last "
                    f"dimension with shape [tokens, {packed_width}], got {tuple(packed.shape)}"
                )

            feature_start = 0
            for offset_start, offset_end in item.offsets or []:
                offset_length = offset_end - offset_start + 1
                feature_end = feature_start + offset_length
                if feature_end > packed.shape[0]:
                    raise ValueError(
                        "Qwen3-VL precomputed offsets require more feature rows than were supplied: "
                        f"required={feature_end}, supplied={packed.shape[0]}"
                    )

                overlap_start = max(prefix_len, offset_start)
                overlap_end = min(chunk_end, offset_end + 1)
                if overlap_start < overlap_end:
                    local_start = feature_start + overlap_start - offset_start
                    local_end = feature_start + overlap_end - offset_start
                    packed_chunks.append(packed[local_start:local_end])
                    mask_start = request_start + overlap_start - prefix_len
                    mask_end = request_start + overlap_end - prefix_len
                    visual_mask[mask_start:mask_end] = True
                    visual_token_count += overlap_end - overlap_start
                feature_start = feature_end

            if feature_start != packed.shape[0]:
                raise ValueError(
                    "Qwen3-VL precomputed feature rows do not match the item offsets: "
                    f"offset_rows={feature_start}, supplied={packed.shape[0]}"
                )
        request_start = request_end

    if request_start != input_ids.shape[0]:
        raise ValueError(
            "Qwen3-VL precomputed request lengths do not cover the flattened SGLang input: "
            f"covered={request_start}, input_tokens={input_ids.shape[0]}"
        )

    if packed_chunks:
        packed_features = torch.cat(packed_chunks, dim=0)
    else:
        packed_features = torch.empty((0, packed_width), dtype=torch.bfloat16)
    if packed_features.shape[0] != visual_token_count:
        raise ValueError(
            "Qwen3-VL precomputed feature count does not match visual positions in the active prefill chunk: "
            f"features={packed_features.shape[0]}, positions={visual_token_count}"
        )

    input_embeds = wrapper.model.get_input_embeddings()(input_ids).unsqueeze(0)
    packed_features = packed_features.to(device=input_embeds.device, dtype=input_embeds.dtype)
    input_embeds[0, visual_mask] = packed_features[:, :hidden_size]
    deepstack_visual_embeds = [
        packed_features[:, hidden_size * index : hidden_size * (index + 1)] for index in range(1, stream_count)
    ]
    return input_embeds, visual_mask.unsqueeze(0), deepstack_visual_embeds


def patch_qwen3_vl_transformers_precomputed_forward(transformers_multimodal_mixin_cls: type) -> bool:
    """Make SGLang's Transformers wrapper consume packed Qwen3-VL features."""
    original = transformers_multimodal_mixin_cls._forward_hidden_states
    if getattr(original, "_relax_qwen3_vl_precomputed_forward_patch", False):
        return False

    def _forward_with_precomputed_qwen3_vl(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
    ):
        request_items = _get_qwen3_vl_precomputed_items(forward_batch)
        if request_items is None or input_embeds is not None:
            return original(
                self,
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type != "qwen3_vl":
            return original(
                self,
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        input_embeds, visual_pos_masks, deepstack_visual_embeds = _build_qwen3_vl_precomputed_language_inputs(
            wrapper=self,
            input_ids=input_ids,
            forward_batch=forward_batch,
            request_items=request_items,
        )
        runtime_positions = getattr(forward_batch, "mrope_positions", None)
        if runtime_positions is None:
            runtime_positions = positions
        outputs = self.model.language_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            use_cache=False,
            position_ids=self._format_position_ids(runtime_positions),
            return_dict=False,
            forward_batch=forward_batch,
            attention_instances=self.attention_instances,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return outputs[0][0, ...]

    _forward_with_precomputed_qwen3_vl._relax_qwen3_vl_precomputed_forward_patch = True
    _forward_with_precomputed_qwen3_vl._relax_original = original
    transformers_multimodal_mixin_cls._forward_hidden_states = _forward_with_precomputed_qwen3_vl
    return True


def patch_qwen3_vl_transformers_to_skip_gpu_vision_encoder(
    *,
    qwen3_vl_model_cls,
    transformers_base_cls,
    enabled: bool,
) -> bool:
    """Keep the Qwen3-VL visual tower on meta and skip its checkpoint weights."""
    if not enabled:
        return False

    original_init = qwen3_vl_model_cls.__init__
    if getattr(original_init, "_relax_skipped_qwen3_vl_vision", False):
        return False

    def _init_without_gpu_vision(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.visual = Qwen3VLPrecomputedOnlyVisual()

    _init_without_gpu_vision._relax_skipped_qwen3_vl_vision = True
    _init_without_gpu_vision._relax_original = original_init
    qwen3_vl_model_cls.__init__ = _init_without_gpu_vision

    original_load_weights = transformers_base_cls.load_weights

    def _load_without_gpu_vision(self, weights):
        prefix = "model.visual."
        if prefix not in self.skip_prefixes:
            self.skip_prefixes.append(prefix)
        return original_load_weights(self, weights)

    _load_without_gpu_vision._relax_skipped_qwen3_vl_vision = True
    _load_without_gpu_vision._relax_original = original_load_weights
    transformers_base_cls.load_weights = _load_without_gpu_vision
    return True


def _materialize_inline_bf16_feature(processor_output: dict[str, Any]) -> torch.Tensor | None:
    if "feature_b64" not in processor_output:
        return None

    feature_dtype = processor_output.get("feature_dtype")
    if feature_dtype != "bfloat16":
        raise ValueError(f"SGLang precomputed feature_dtype must be 'bfloat16', got {feature_dtype!r}")
    feature_shape = processor_output.get("feature_shape")
    if (
        not isinstance(feature_shape, list)
        or not feature_shape
        or not all(isinstance(size, int) and size >= 0 for size in feature_shape)
    ):
        raise ValueError("SGLang precomputed feature_shape must be a non-empty list of non-negative integers")
    try:
        feature_bytes = base64.b64decode(processor_output["feature_b64"], validate=True)
    except (binascii.Error, TypeError, ValueError) as error:
        raise ValueError("SGLang precomputed feature_b64 must contain valid base64") from error
    expected_byte_count = math.prod(feature_shape) * torch.bfloat16.itemsize
    if len(feature_bytes) != expected_byte_count:
        raise ValueError(
            "SGLang precomputed feature byte count does not match feature_shape: "
            f"received={len(feature_bytes)}, expected={expected_byte_count}, feature_shape={feature_shape}"
        )

    processor_output.pop("feature_b64")
    processor_output.pop("feature_dtype")
    processor_output.pop("feature_shape")
    return torch.frombuffer(bytearray(feature_bytes), dtype=torch.uint16).view(torch.bfloat16).reshape(feature_shape)


def patch_transformers_auto_precomputed_embedding_processor(processor_cls) -> bool:
    """Normalize precomputed dictionaries before SGLang collects MM items."""
    if not hasattr(processor_cls, "collect_mm_items_from_processor_output"):
        return False
    original = processor_cls.collect_mm_items_from_processor_output
    if getattr(original, "_relax_qwen3_vl_precomputed_auto_patch", False):
        return False

    raw_max_bytes = os.environ.get("RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "0")
    try:
        cache_max_bytes = int(raw_max_bytes)
    except ValueError as error:
        raise ValueError(
            "RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES must be a non-negative integer, "
            f"got {raw_max_bytes!r}"
        ) from error
    feature_cache = SGLangVisionFeatureCache(max_bytes=cache_max_bytes)

    def _collect_precomputed(self, processor_output, modality=None):
        processor_output = feature_cache.resolve(processor_output)
        if processor_output.get("format") == "precomputed_embedding":
            processor_output = dict(processor_output)
            packed_feature = _materialize_inline_bf16_feature(processor_output)
            if packed_feature is not None:
                processor_output["precomputed_embeddings"] = packed_feature
            else:
                processor_output["precomputed_embeddings"] = torch.as_tensor(
                    processor_output.pop("feature"),
                    dtype=torch.bfloat16,
                )
            processor_output["image_grid_thw"] = torch.as_tensor(
                processor_output["image_grid_thw"],
                dtype=torch.int64,
            )
        return original(self, processor_output, modality=modality)

    _collect_precomputed._relax_qwen3_vl_precomputed_auto_patch = True
    _collect_precomputed._relax_original = original
    processor_cls.collect_mm_items_from_processor_output = _collect_precomputed
    processor_cls._relax_sglang_vision_feature_cache = feature_cache
    return True


def patch_sglang_precomputed_embedding_id_loader(processor_cls) -> bool:
    """Let ID-only feature references reach the processor-level host cache.

    SGLang's media loader already passes its built-in precomputed embedding
    dictionaries through without image decoding. The Relax ID-only format must
    take the same path so ``SGLangVisionFeatureCache.resolve`` can replace it
    with the previously published immutable feature bundle.
    """
    original = processor_cls._load_single_item
    original_function = getattr(original, "__func__", original)
    if getattr(original_function, "_relax_precomputed_embedding_id_loader_patch", False):
        return False

    def _load_single_item(cls, data, *args, **kwargs):
        if isinstance(data, dict) and data.get("format") == "precomputed_embedding_id":
            return data
        return original_function(cls, data, *args, **kwargs)

    _load_single_item._relax_precomputed_embedding_id_loader_patch = True
    _load_single_item._relax_original = original_function
    processor_cls._load_single_item = classmethod(_load_single_item)
    return True


def install_qwen3_vl_precomputed_vision_patch() -> None:
    """Preserve grids and consume packed Qwen3-VL final/DeepStack embeddings.

    SGLang's generic ``precomputed_embedding`` dictionary path stores the
    supplied tensor in ``feature`` and can return ``ret=None``. Qwen3-VL needs
    the same data in ``precomputed_embeddings``, needs ``image_grid_thw`` on
    ``ret`` for MRoPE, and requires the Transformers wrapper to split and
    inject the packed final plus DeepStack streams. Raw-image and decode
    requests are unchanged.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputFormat
    from sglang.srt.models.transformers import MultiModalMixin
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    patch_sglang_precomputed_embedding_id_loader(BaseMultimodalProcessor)
    patch_transformers_auto_precomputed_embedding_processor(BaseMultimodalProcessor)
    patch_qwen3_vl_transformers_precomputed_forward(MultiModalMixin)
    original = BaseMultimodalProcessor.process_and_combine_mm_data
    if getattr(original, "_relax_qwen3_vl_precomputed_vision_patch", False):
        return

    def _process_and_combine_with_precomputed_vision(self, base_output, mm_tokens, **kwargs):
        organize_results = getattr(base_output, "organize_results", None)
        if organize_results is not None:
            for _, processor_output in organize_results():
                if not isinstance(processor_output, dict):
                    continue
                feature_cache = type(self)._relax_sglang_vision_feature_cache
                resolved_output = feature_cache.resolve(processor_output)
                if resolved_output is not processor_output:
                    processor_output.clear()
                    processor_output.update(resolved_output)
                if processor_output.get("format") == "precomputed_embedding":
                    packed_feature = _materialize_inline_bf16_feature(processor_output)
                    if packed_feature is not None:
                        processor_output["feature"] = packed_feature
        items, input_ids, processor_output = original(self, base_output, mm_tokens, **kwargs)
        precomputed_items = [item for item in items if item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING]
        if not precomputed_items:
            return items, input_ids, processor_output

        if processor_output is None:
            processor_output = SimpleNamespace()

        image_grids = []
        for item in precomputed_items:
            if item.precomputed_embeddings is None:
                item.precomputed_embeddings = torch.as_tensor(item.feature, dtype=torch.bfloat16)
                item.feature = None
            grid = getattr(item, "image_grid_thw", None)
            if grid is not None:
                grid_tensor = torch.as_tensor(grid, dtype=torch.int64)
                item.model_specific_data["image_grid_thw"] = grid_tensor
                image_grids.append(grid_tensor)

        if image_grids:
            processor_output.image_grid_thw = torch.cat(image_grids, dim=0)
        return items, input_ids, processor_output

    _process_and_combine_with_precomputed_vision._relax_qwen3_vl_precomputed_vision_patch = True
    _process_and_combine_with_precomputed_vision._relax_original = original
    BaseMultimodalProcessor.process_and_combine_mm_data = _process_and_combine_with_precomputed_vision
