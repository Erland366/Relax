# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang compatibility for Qwen3-VL precomputed visual embeddings."""

from types import SimpleNamespace

import torch


class OmittedQwen3VLVisual(torch.nn.Module):
    """Parameterless guard for raw-image requests in GPU omission mode."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen3-VL GPU vision weights were omitted; provide precomputed vision features"
        )


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
    """Preserve Qwen3-VL grids and mark HTTP features as final embeddings.

    SGLang's generic ``precomputed_embedding`` dictionary path stores the
    supplied tensor in ``feature`` and can return ``ret=None``. Qwen3-VL needs
    the same data in ``precomputed_embeddings`` and needs ``image_grid_thw`` on
    ``ret`` for MRoPE. Patch only this conversion seam; raw-image requests are
    unchanged.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputFormat
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    patch_transformers_auto_precomputed_embedding_processor(BaseMultimodalProcessor)
    original = BaseMultimodalProcessor.process_and_combine_mm_data
    if getattr(original, "_relax_qwen3_vl_precomputed_vision_patch", False):
        return

    def _process_and_combine_with_precomputed_vision(self, base_output, mm_tokens, **kwargs):
        items, input_ids, processor_output = original(self, base_output, mm_tokens, **kwargs)
        precomputed_items = [
            item for item in items if item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        ]
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
