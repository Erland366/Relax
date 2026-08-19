# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Data contract for Qwen3-VL features produced by a frozen vision tower."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_VISUAL_STATE_PREFIX = "model.visual."
IMAGE_GRID_THW_KEY = "image_grid_thw"
VISION_EMBEDS_KEY = "vision_embeds"
QWEN3_VL_FROZEN_VISION_FEATURE_SCHEMA_VERSION = "qwen3-vl-frozen-vision-v1"


def deepstack_visual_embeds_key(index: int) -> str:
    """Return the canonical actor-input key for one ordered DeepStack stream."""
    return f"deepstack_visual_embeds_{index}"


def _update_hash_with_tensor(digest: Any, name: str, tensor: torch.Tensor) -> None:
    """Add a tensor's identity and contents to a deterministic digest."""
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("ascii"))
    digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
    digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True)
class Qwen3VLFrozenVisionFeatures:
    """Projected Qwen3-VL image features ready for the language model."""

    image_grid_thw: torch.Tensor
    vision_embeds: torch.Tensor
    deepstack_visual_embeds: tuple[torch.Tensor, ...]
    feature_id: str = ""
    vision_revision: str = ""
    feature_schema_version: str = QWEN3_VL_FROZEN_VISION_FEATURE_SCHEMA_VERSION

    @property
    def embedding_streams(self) -> tuple[torch.Tensor, ...]:
        """Return final and DeepStack embeddings in Qwen3-VL consumption order."""
        return (self.vision_embeds, *self.deepstack_visual_embeds)

    @property
    def nbytes(self) -> int:
        """Return the bytes resident in every tensor in this feature bundle."""
        tensors = (self.image_grid_thw, *self.embedding_streams)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def validate(
        self,
        *,
        spatial_merge_size: int,
        image_token_count: int,
        hidden_size: int,
        expected_deepstack_count: int,
    ) -> None:
        """Validate that one feature bundle matches Qwen3-VL's token contract."""
        if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
            raise ValueError(
                "image_grid_thw must have shape [num_images, 3], "
                f"got {tuple(self.image_grid_thw.shape)}"
            )
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")

        merged_token_count = int(self.image_grid_thw.prod(dim=1).sum()) // (spatial_merge_size**2)
        if merged_token_count != image_token_count:
            raise ValueError(
                "Qwen3-VL image token count does not match the merged image grid: "
                f"prompt has {image_token_count}, features require {merged_token_count}"
            )

        expected_shape = (image_token_count, hidden_size)
        if tuple(self.vision_embeds.shape) != expected_shape:
            raise ValueError(
                f"vision_embeds must have shape {expected_shape}, got {tuple(self.vision_embeds.shape)}"
            )
        if len(self.deepstack_visual_embeds) != expected_deepstack_count:
            raise ValueError(
                "Qwen3-VL DeepStack feature count does not match the model: "
                f"expected {expected_deepstack_count}, got {len(self.deepstack_visual_embeds)}"
            )
        for index, feature in enumerate(self.deepstack_visual_embeds):
            if tuple(feature.shape) != expected_shape:
                raise ValueError(
                    f"deepstack_visual_embeds[{index}] must have shape {expected_shape}, "
                    f"got {tuple(feature.shape)}"
                )


def build_sglang_precomputed_image_data(features: Qwen3VLFrozenVisionFeatures) -> dict[str, Any]:
    """Build SGLang's Qwen3-VL precomputed-embedding image payload."""
    return {
        "format": "precomputed_embedding",
        "feature": torch.cat(features.embedding_streams, dim=-1),
        IMAGE_GRID_THW_KEY: features.image_grid_thw,
        "feature_id": features.feature_id,
        "vision_revision": features.vision_revision,
        "feature_schema_version": features.feature_schema_version,
    }


def build_qwen3_vl_actor_feature_inputs(features: Qwen3VLFrozenVisionFeatures) -> dict[str, torch.Tensor]:
    """Build the tensor-only per-sample feature bundle consumed by Megatron."""
    actor_inputs = {
        IMAGE_GRID_THW_KEY: features.image_grid_thw,
        VISION_EMBEDS_KEY: features.vision_embeds,
    }
    actor_inputs.update(
        {
            deepstack_visual_embeds_key(index): feature
            for index, feature in enumerate(features.deepstack_visual_embeds)
        }
    )
    return actor_inputs


def load_qwen3_vl_visual_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load only ``model.visual`` tensors from a local safetensors checkpoint."""
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    checkpoint_files = (
        sorted(checkpoint_path.glob("*.safetensors"))
        if checkpoint_path.is_dir()
        else [checkpoint_path]
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {checkpoint_path}")

    visual_state_dict: dict[str, torch.Tensor] = {}
    for checkpoint_file in checkpoint_files:
        with safe_open(str(checkpoint_file), framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if not name.startswith(_VISUAL_STATE_PREFIX):
                    continue
                visual_name = name.removeprefix(_VISUAL_STATE_PREFIX)
                if visual_name in visual_state_dict:
                    raise ValueError(
                        f"Duplicate Qwen3-VL visual tensor {visual_name!r} in {checkpoint_path}"
                    )
                visual_state_dict[visual_name] = checkpoint.get_tensor(name)

    if not visual_state_dict:
        raise ValueError(f"No {_VISUAL_STATE_PREFIX} tensors found in {checkpoint_path}")
    return visual_state_dict


def compute_qwen3_vl_visual_revision(
    state_dict: Mapping[str, torch.Tensor],
    *,
    config_fingerprint: str,
) -> str:
    """Build a deterministic revision from visual weights and processor config."""
    digest = hashlib.sha256()
    digest.update(config_fingerprint.encode("utf-8"))
    for name in sorted(state_dict):
        _update_hash_with_tensor(digest, name, state_dict[name])
    return digest.hexdigest()


def build_qwen3_vl_feature_cache_key(
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    vision_revision: str,
    output_dtype: torch.dtype,
    feature_schema_version: str = QWEN3_VL_FROZEN_VISION_FEATURE_SCHEMA_VERSION,
) -> str:
    """Address one frozen feature bundle by input, model revision, and output dtype."""
    digest = hashlib.sha256()
    digest.update(feature_schema_version.encode("utf-8"))
    digest.update(vision_revision.encode("ascii"))
    digest.update(str(output_dtype).encode("ascii"))
    _update_hash_with_tensor(digest, "pixel_values", pixel_values)
    _update_hash_with_tensor(digest, "image_grid_thw", image_grid_thw)
    return digest.hexdigest()


class Qwen3VLCPUVisionBackend:
    """Run an already-constructed frozen Qwen3-VL visual module on CPU."""

    def __init__(
        self,
        *,
        visual_model: Any,
        revision: str,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.visual_model = visual_model.eval()
        self.revision = revision
        self.output_dtype = output_dtype
        self.feature_schema_version = QWEN3_VL_FROZEN_VISION_FEATURE_SCHEMA_VERSION

    def encode(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Qwen3VLFrozenVisionFeatures:
        """Encode image patches into final and DeepStack projected features."""
        with torch.inference_mode():
            output = self.visual_model(
                pixel_values,
                grid_thw=image_grid_thw,
            )
        if hasattr(output, "pooler_output"):
            vision_embeds = output.pooler_output
            deepstack_visual_embeds = output.deepstack_features
        else:
            vision_embeds, deepstack_visual_embeds = output
        return Qwen3VLFrozenVisionFeatures(
            image_grid_thw=image_grid_thw,
            vision_embeds=vision_embeds,
            deepstack_visual_embeds=tuple(deepstack_visual_embeds),
            feature_id=build_qwen3_vl_feature_cache_key(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                vision_revision=self.revision,
                output_dtype=self.output_dtype,
                feature_schema_version=self.feature_schema_version,
            ),
            vision_revision=self.revision,
            feature_schema_version=self.feature_schema_version,
        )


def build_qwen3_vl_cpu_vision_backend(
    checkpoint_path: str | Path,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    state_dict_loader=load_qwen3_vl_visual_state_dict,
    revision_builder=compute_qwen3_vl_visual_revision,
) -> Qwen3VLCPUVisionBackend:
    """Construct the frozen HF Qwen3-VL visual module without loading text weights."""
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    checkpoint_location = checkpoint_path
    checkpoint_path = Path(checkpoint_path)
    config = AutoConfig.from_pretrained(checkpoint_location, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type is not None and model_type != "qwen3_vl":
        raise ValueError(
            "The CPU vision encoder currently supports only Qwen3-VL checkpoints, "
            f"got model_type={model_type!r} from {checkpoint_path}"
        )

    vision_config = config.vision_config
    config_fingerprint = json.dumps(vision_config.to_dict(), sort_keys=True, separators=(",", ":"))
    visual_state_dict = state_dict_loader(checkpoint_location)
    revision = revision_builder(
        visual_state_dict,
        config_fingerprint=config_fingerprint,
    )

    visual_model = Qwen3VLVisionModel(vision_config)
    visual_model.load_state_dict(visual_state_dict, strict=True)
    visual_model.to(device="cpu", dtype=output_dtype)
    if hasattr(visual_model, "parameters"):
        for parameter in visual_model.parameters():
            parameter.requires_grad_(False)

    return Qwen3VLCPUVisionBackend(
        visual_model=visual_model,
        revision=revision,
        output_dtype=output_dtype,
    )
