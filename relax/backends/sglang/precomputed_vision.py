# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang compatibility for Qwen3-VL precomputed visual embeddings."""

import base64
import binascii
import math
from types import SimpleNamespace
from typing import Any

import torch


class OmittedQwen3VLVisual(torch.nn.Module):
    """Parameterless guard for raw-image requests in GPU omission mode."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("Qwen3-VL GPU vision weights were omitted; provide precomputed vision features")


def _get_qwen3_vl_precomputed_items(forward_batch: Any) -> list[list[Any]] | None:
    if forward_batch.forward_mode.is_decode() or not forward_batch.contains_mm_inputs():
        return None

    request_items = []
    has_precomputed = False
    has_native = False
    for mm_input in forward_batch.mm_inputs or []:
        items = [] if mm_input is None else [item for item in mm_input.mm_items or [] if item is not None]
        request_items.append(items)
        for item in items:
            if getattr(item, "precomputed_embeddings", None) is None:
                has_native = True
            else:
                has_precomputed = True

    if not has_precomputed:
        return None
    if has_native:
        raise RuntimeError("Qwen3-VL SGLang transformers batches cannot mix native and precomputed vision items")
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


def patch_qwen3_vl_transformers_for_omitted_vision(
    *,
    qwen3_vl_model_cls,
    transformers_base_cls,
    enabled: bool,
) -> bool:
    """Keep the Qwen3-VL visual tower on meta and skip its checkpoint weights."""
    if not enabled:
        return False

    original_init = qwen3_vl_model_cls.__init__
    if getattr(original_init, "_relax_omitted_qwen3_vl_vision", False):
        return False

    def _init_without_gpu_vision(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.visual = OmittedQwen3VLVisual()

    _init_without_gpu_vision._relax_omitted_qwen3_vl_vision = True
    _init_without_gpu_vision._relax_original = original_init
    qwen3_vl_model_cls.__init__ = _init_without_gpu_vision

    original_load_weights = transformers_base_cls.load_weights

    def _load_without_gpu_vision(self, weights):
        prefix = "model.visual."
        if prefix not in self.skip_prefixes:
            self.skip_prefixes.append(prefix)
        return original_load_weights(self, weights)

    _load_without_gpu_vision._relax_omitted_qwen3_vl_vision = True
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

    def _collect_precomputed(self, processor_output, modality=None):
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

    patch_transformers_auto_precomputed_embedding_processor(BaseMultimodalProcessor)
    patch_qwen3_vl_transformers_precomputed_forward(MultiModalMixin)
    original = BaseMultimodalProcessor.process_and_combine_mm_data
    if getattr(original, "_relax_qwen3_vl_precomputed_vision_patch", False):
        return

    def _process_and_combine_with_precomputed_vision(self, base_output, mm_tokens, **kwargs):
        organize_results = getattr(base_output, "organize_results", None)
        if organize_results is not None:
            for _, processor_output in organize_results():
                if isinstance(processor_output, dict) and processor_output.get("format") == "precomputed_embedding":
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
