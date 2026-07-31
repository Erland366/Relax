# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional CPU vision encoder service registration and configuration."""

from argparse import Namespace
from time import monotonic
from typing import Any, Optional

import torch

from relax.backends.vision import qwen3_vl as qwen3_vl_backend
from relax.backends.vision.cache import ByteBoundedLRUCache
from relax.backends.vision.qwen3_vl import (
    Qwen3VLFrozenVisionFeatures,
    build_qwen3_vl_cpu_vision_backend,
    build_qwen3_vl_feature_cache_key,
)
from relax.components.base import Base


VISION_ENCODER_ROLE = "vision_encoder"


def validate_vision_encoder_config(config: Namespace) -> None:
    """Validate the explicit resource and freezing contract for CPU vision."""
    backend = getattr(config, "vision_encoder_backend", "disabled")
    omit_gpu_weights = getattr(config, "vision_encoder_omit_gpu_weights", False)
    if omit_gpu_weights and backend != "pytorch":
        raise ValueError(
            "vision_encoder_omit_gpu_weights can only omit GPU weights when "
            "vision_encoder_backend='pytorch'"
        )
    if backend == "disabled":
        resource = getattr(config, "resource", None) or {}
        if VISION_ENCODER_ROLE in resource:
            raise ValueError(
                "resource contains 'vision_encoder' but --vision-encoder-backend is disabled"
            )
        return
    if backend != "pytorch":
        raise ValueError(f"Unsupported vision encoder backend: {backend!r}")

    if not getattr(config, "freeze_vision_model", False) or not getattr(
        config, "freeze_vision_projection", False
    ):
        raise ValueError(
            "The CPU vision encoder requires both freeze_vision_model and "
            "freeze_vision_projection to be enabled"
        )

    resource = getattr(config, "resource", None) or {}
    if resource.get(VISION_ENCODER_ROLE) != [1, 0]:
        raise ValueError("resource['vision_encoder'] must be [1, 0] for the CPU vision encoder service")
    if getattr(config, "vision_encoder_num_replicas", 1) <= 0:
        raise ValueError("vision_encoder_num_replicas must be positive")
    if getattr(config, "vision_encoder_num_cpus", 0) <= 0:
        raise ValueError("vision_encoder_num_cpus must be positive")
    if getattr(config, "vision_encoder_cache_max_bytes", 0) <= 0:
        raise ValueError("vision_encoder_cache_max_bytes must be positive")
    if getattr(config, "vision_encoder_max_batch_size", 0) <= 0:
        raise ValueError("vision_encoder_max_batch_size must be positive")
    if getattr(config, "context_parallel_size", 1) != 1:
        raise NotImplementedError("The CPU vision encoder currently requires context_parallel_size=1")
    if getattr(config, "pipeline_model_parallel_size", 1) != 1:
        raise NotImplementedError("The CPU vision encoder currently requires pipeline_model_parallel_size=1")
    if getattr(config, "megatron_to_hf_mode", "bridge") != "bridge":
        raise NotImplementedError("The CPU vision encoder currently requires Megatron-Bridge mode")

    multimodal_keys = getattr(config, "multimodal_keys", None)
    if multimodal_keys is not None and set(multimodal_keys) != {"image"}:
        raise NotImplementedError(
            "The CPU vision encoder currently supports image-only Qwen3-VL data; "
            f"got multimodal keys {sorted(multimodal_keys)}"
        )


class VisionEncoder(Base):
    """Passive service boundary for the frozen CPU vision backend."""

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.healthy = healthy
        self.pg = pg
        self.num_gpus = num_gpus
        self.config = config
        self.role = role
        self.runtime_env = runtime_env
        validate_vision_encoder_config(config)
        if num_gpus != 0:
            raise ValueError(f"The CPU vision encoder requires num_gpus=0, got {num_gpus}")

        torch.set_num_threads(config.vision_encoder_num_cpus)
        self.max_batch_size = config.vision_encoder_max_batch_size
        self.backend = build_qwen3_vl_cpu_vision_backend(
            config.hf_checkpoint,
            state_dict_loader=qwen3_vl_backend.load_qwen3_vl_visual_state_dict,
            revision_builder=qwen3_vl_backend.compute_qwen3_vl_visual_revision,
        )
        self.cache = ByteBoundedLRUCache(config.vision_encoder_cache_max_bytes)
        self._initialize_metrics()
        self._logger.info(
            "Loaded frozen Qwen3-VL CPU vision encoder revision=%s with %d CPU threads",
            self.backend.revision[:12],
            config.vision_encoder_num_cpus,
        )

    @classmethod
    def options(cls, **kwargs):
        """Create Ray Serve deployment options without importing Ray in test/control processes."""
        from ray import serve

        return serve.deployment(cls).options(**kwargs)

    def run(self) -> None:
        """The encoder serves requests and has no background training loop."""
        return None

    def _initialize_metrics(self) -> None:
        self._encode_requests_total = 0
        self._backend_encode_requests_total = 0
        self._backend_encoded_images_total = 0
        self._emitted_feature_bytes_total = 0
        self._backend_encode_seconds_total = 0.0

    def encode(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        requested_feature_id: Optional[str] = None,
        requested_vision_revision: Optional[str] = None,
    ) -> Qwen3VLFrozenVisionFeatures:
        """Return cached frozen features, encoding the image on a cache miss."""
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError(
                "image_grid_thw must have shape [num_images, 3], "
                f"got {tuple(image_grid_thw.shape)}"
            )
        max_batch_size = getattr(self, "max_batch_size", None)
        if max_batch_size is not None and image_grid_thw.shape[0] > max_batch_size:
            raise ValueError(
                f"CPU vision request contains {image_grid_thw.shape[0]} images, "
                f"exceeding vision_encoder_max_batch_size={max_batch_size}"
            )
        feature_id = build_qwen3_vl_feature_cache_key(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_revision=self.backend.revision,
            output_dtype=self.backend.output_dtype,
        )
        if requested_feature_id is not None and requested_feature_id != feature_id:
            raise ValueError("requested feature_id does not match the encoded input")
        if (
            requested_vision_revision is not None
            and requested_vision_revision != self.backend.revision
        ):
            raise ValueError("requested vision_revision does not match the encoder revision")
        if not hasattr(self, "_encode_requests_total"):
            self._initialize_metrics()
        self._encode_requests_total += 1
        features = self.cache.get(feature_id)
        if features is not None:
            return features

        encode_started_at = monotonic()
        features = self.backend.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        feature_bytes = features.nbytes
        self._backend_encode_seconds_total += monotonic() - encode_started_at
        self._backend_encode_requests_total += 1
        self._backend_encoded_images_total += image_grid_thw.shape[0]
        self._emitted_feature_bytes_total += feature_bytes
        self.cache.put(feature_id, features, size_bytes=feature_bytes)
        return features

    def get_metrics(self) -> dict[str, Any]:
        """Expose cache and backend-work counters through the service API."""
        if not hasattr(self, "_encode_requests_total"):
            self._initialize_metrics()
        return {
            **self.cache.stats,
            "encode_requests_total": self._encode_requests_total,
            "backend_encode_requests_total": self._backend_encode_requests_total,
            "backend_encoded_images_total": self._backend_encoded_images_total,
            "emitted_feature_bytes_total": self._emitted_feature_bytes_total,
            "backend_encode_seconds_total": self._backend_encode_seconds_total,
        }


def register_vision_encoder(config: Namespace, algo: dict) -> list[str]:
    """Register the optional vision encoder service when explicitly enabled."""
    if getattr(config, "vision_encoder_backend", "disabled") == "disabled":
        return []
    algo[VISION_ENCODER_ROLE] = VisionEncoder
    return [VISION_ENCODER_ROLE]
