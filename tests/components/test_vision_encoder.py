# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from argparse import ArgumentParser, Namespace
from dataclasses import FrozenInstanceError

import pytest
import torch


def _vision_component():
    return importlib.import_module("relax.components.vision_encoder")


def test_disabled_vision_encoder_config_preserves_existing_launchers():
    _vision_component().validate_vision_encoder_config(Namespace(vision_encoder_device="gpu"))


def test_vision_encoder_arguments_expose_opt_in_gpu_encoder_skip():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--skip-gpu-vision-encoder" in action.option_strings
    ]

    assert len(actions) == 1, "--skip-gpu-vision-encoder must be registered"
    assert actions[0].default is False
    assert actions[0].const is True


def test_vision_encoder_arguments_default_to_one_cpu_replica():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--vision-encoder-num-replicas" in action.option_strings
    ]

    assert len(actions) == 1, "--vision-encoder-num-replicas must be registered"
    assert actions[0].default == 1


def test_pytorch_vision_encoder_requires_positive_replica_count():
    config = Namespace(
        vision_encoder_device="cpu",
        vision_encoder_num_replicas=0,
        vision_encoder_num_cpus=8,
        vision_encoder_cache_max_bytes=1024,
        vision_encoder_max_images_per_request=8,
        resource={"vision_encoder": [1, 0]},
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )

    with pytest.raises(ValueError, match="vision_encoder_num_replicas must be positive"):
        _vision_component().validate_vision_encoder_config(config)


def test_multiple_cpu_vision_replicas_keep_one_cpu_only_service_resource():
    config = Namespace(
        vision_encoder_device="cpu",
        vision_encoder_num_replicas=3,
        vision_encoder_num_cpus=8,
        vision_encoder_cache_max_bytes=1024,
        vision_encoder_max_images_per_request=8,
        resource={"vision_encoder": [1, 0]},
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )

    _vision_component().validate_vision_encoder_config(config)
    assert config.resource["vision_encoder"] == [1, 0]


def test_pytorch_vision_encoder_requires_frozen_tower_and_projection():
    config = Namespace(
        vision_encoder_device="cpu",
        vision_encoder_num_cpus=8,
        resource={"vision_encoder": [1, 0]},
        freeze_vision_model=True,
        freeze_vision_projection=False,
    )

    with pytest.raises(ValueError, match="freeze.*vision.*projection"):
        _vision_component().validate_vision_encoder_config(config)


def test_skip_gpu_vision_encoder_requires_cpu_vision_encoder():
    config = Namespace(
        vision_encoder_device="gpu",
        skip_gpu_vision_encoder=True,
    )

    with pytest.raises(ValueError, match="skip_gpu_vision_encoder.*vision_encoder_device='cpu'"):
        _vision_component().validate_vision_encoder_config(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"freeze_vision_model": False}, "freeze.*vision"),
        ({"freeze_vision_projection": False}, "freeze.*vision.*projection"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=1"),
        ({"context_parallel_size": 2}, "context_parallel_size=1"),
    ],
)
def test_skip_gpu_vision_encoder_requires_frozen_pp1_cp1_config(override, message):
    values = {
        "vision_encoder_device": "cpu",
        "skip_gpu_vision_encoder": True,
        "vision_encoder_num_cpus": 8,
        "vision_encoder_cache_max_bytes": 1024,
        "vision_encoder_max_images_per_request": 8,
        "resource": {"vision_encoder": [1, 0]},
        "freeze_vision_model": True,
        "freeze_vision_projection": True,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "megatron_to_hf_mode": "bridge",
    }
    values.update(override)

    with pytest.raises((ValueError, NotImplementedError), match=message):
        _vision_component().validate_vision_encoder_config(Namespace(**values))


def test_register_vision_encoder_only_when_enabled(monkeypatch):
    # Ray rejects ROCR_VISIBLE_DEVICES during import unless the corresponding
    # HIP variable is present. This unit test does not initialize Ray.
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    vision_component = _vision_component()
    algo = {}
    config = Namespace(genrm_model_path=None, vision_encoder_device="cpu")

    roles = vision_component.register_vision_encoder(config, algo)

    assert roles == ["vision_encoder"]
    assert algo["vision_encoder"] is vision_component.VisionEncoder


def test_vision_encoder_encode_returns_replica_telemetry_with_cached_frozen_features():
    vision_component = _vision_component()
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    cache_module = importlib.import_module("relax.backends.vision.cache")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    encoded = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=image_grid_thw,
        vision_embeds=torch.ones((1, 2), dtype=torch.bfloat16),
        deepstack_visual_embeds=tuple(
            torch.full((1, 2), value, dtype=torch.bfloat16) for value in (2.0, 3.0, 4.0)
        ),
    )

    class FakeBackend:
        revision = "vision-revision"
        output_dtype = torch.bfloat16

        def __init__(self):
            self.calls = 0

        def encode(self, *, pixel_values, image_grid_thw):
            self.calls += 1
            return encoded

    backend = FakeBackend()
    encoder = vision_component.VisionEncoder.__new__(vision_component.VisionEncoder)
    encoder.backend = backend
    encoder.cache = cache_module.ByteBoundedLRUCache(max_bytes=1024)
    encoder.replica_id = "vision-replica-a"

    first = encoder.encode(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    second = encoder.encode(pixel_values=pixel_values.clone(), image_grid_thw=image_grid_thw.clone())

    expected_feature_id = vision_module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision=backend.revision,
        output_dtype=backend.output_dtype,
    )
    assert isinstance(first, vision_component.VisionEncoderResponse)
    assert first.features is encoded
    assert second.features is first.features
    assert first.replica_id == second.replica_id == "vision-replica-a"
    assert first.feature_id == second.feature_id == expected_feature_id
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.backend_encode_seconds >= 0.0
    assert first.backend_batch_size == 1
    assert second.backend_encode_seconds == 0.0
    assert second.backend_batch_size == 0
    assert backend.calls == 1
    assert first.metrics_snapshot["encode_requests_total"] == 1
    assert first.metrics_snapshot["hits"] == 0
    assert first.metrics_snapshot["misses"] == 1
    assert first.metrics_snapshot["backend_encode_requests_total"] == 1
    assert first.metrics_snapshot["backend_encoded_images_total"] == 1
    assert first.metrics_snapshot["emitted_feature_bytes_total"] == encoded.nbytes
    assert first.metrics_snapshot["process_cpu_seconds_total"] >= 0.0
    assert first.metrics_snapshot["rss_bytes"] >= 0
    assert second.metrics_snapshot["encode_requests_total"] == 2
    assert second.metrics_snapshot["hits"] == 1
    assert second.metrics_snapshot["misses"] == 1
    assert second.metrics_snapshot["backend_encode_requests_total"] == 1
    assert second.metrics_snapshot["backend_encoded_images_total"] == 1
    assert second.metrics_snapshot["emitted_feature_bytes_total"] == encoded.nbytes
    with pytest.raises(FrozenInstanceError):
        first.features.feature_id = "mutable"


def test_vision_encoder_encode_rejects_mismatched_requested_identity():
    vision_component = _vision_component()
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    cache_module = importlib.import_module("relax.backends.vision.cache")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    revision = "vision-revision"
    feature_id = vision_module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision=revision,
        output_dtype=torch.bfloat16,
    )
    encoded = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=image_grid_thw,
        vision_embeds=torch.ones((1, 2), dtype=torch.bfloat16),
        deepstack_visual_embeds=(),
    )
    object.__setattr__(encoded, "feature_id", feature_id)
    object.__setattr__(encoded, "vision_revision", revision)

    class FakeBackend:
        output_dtype = torch.bfloat16
        revision = "vision-revision"

        def encode(self, *, pixel_values, image_grid_thw):
            return encoded

    encoder = vision_component.VisionEncoder.__new__(vision_component.VisionEncoder)
    encoder.backend = FakeBackend()
    encoder.cache = cache_module.ByteBoundedLRUCache(max_bytes=1024)
    encoder.replica_id = "vision-replica-identity"

    response = encoder.encode(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        requested_feature_id=feature_id,
        requested_vision_revision=revision,
    )
    assert response.features is encoded
    assert response.feature_id == feature_id
    with pytest.raises(ValueError, match="feature_id.*match"):
        encoder.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            requested_feature_id="wrong-feature-id",
            requested_vision_revision=revision,
        )
    with pytest.raises(ValueError, match="vision_revision.*match"):
        encoder.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            requested_feature_id=feature_id,
            requested_vision_revision="wrong-vision-revision",
        )


def test_vision_encoder_response_propagates_and_validates_feature_schema_version():
    vision_component = _vision_component()
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    cache_module = importlib.import_module("relax.backends.vision.cache")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    schema_version = "qwen3-vl-frozen-vision-v1"
    feature_id = vision_module.build_qwen3_vl_feature_cache_key(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_revision="vision-revision",
        feature_schema_version=schema_version,
        output_dtype=torch.bfloat16,
    )
    encoded = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=image_grid_thw,
        vision_embeds=torch.ones((1, 2), dtype=torch.bfloat16),
        deepstack_visual_embeds=(),
        feature_id=feature_id,
        vision_revision="vision-revision",
        feature_schema_version=schema_version,
    )

    class FakeBackend:
        output_dtype = torch.bfloat16
        revision = "vision-revision"
        feature_schema_version = schema_version

        def encode(self, *, pixel_values, image_grid_thw):
            return encoded

    encoder = vision_component.VisionEncoder.__new__(vision_component.VisionEncoder)
    encoder.backend = FakeBackend()
    encoder.cache = cache_module.ByteBoundedLRUCache(max_bytes=1024)
    encoder.replica_id = "vision-replica-schema"

    response = encoder.encode(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        requested_feature_id=feature_id,
        requested_vision_revision="vision-revision",
        requested_feature_schema_version=schema_version,
    )

    assert response.feature_schema_version == schema_version
    assert response.features.feature_schema_version == schema_version
    with pytest.raises(ValueError, match="feature_schema_version.*match"):
        encoder.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            requested_feature_id=feature_id,
            requested_vision_revision="vision-revision",
            requested_feature_schema_version="qwen3-vl-frozen-vision-v2",
        )


def test_vision_encoder_init_builds_selectively_loaded_cpu_eval_backend(monkeypatch):
    vision_component = _vision_component()
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    loaded_state = {"patch_embed.weight": torch.tensor([1.0])}
    calls = {}

    class FakeVisionConfig:
        def to_dict(self):
            return {"depth": 1, "hidden_size": 2}

    full_config = Namespace(model_type="qwen3_vl", vision_config=FakeVisionConfig())

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code=True):
            calls["config_path"] = path
            return full_config

    class FakeVisualModel:
        def __init__(self):
            self.training = True

        def load_state_dict(self, state_dict, strict=True):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

        def to(self, *args, **kwargs):
            calls["to"] = (args, kwargs)
            return self

        def eval(self):
            self.training = False
            return self

        def parameters(self):
            return ()

    visual_model = FakeVisualModel()

    class Qwen3VLVisionModel:
        def __new__(cls, config):
            calls["vision_config"] = config
            return visual_model

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = AutoConfig
    qwen_modeling_module = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")
    qwen_modeling_module.Qwen3VLVisionModel = Qwen3VLVisionModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        qwen_modeling_module,
    )
    def load_visual_state(path):
        calls["checkpoint_path"] = path
        return loaded_state

    monkeypatch.setattr(
        vision_module,
        "load_qwen3_vl_visual_state_dict",
        load_visual_state,
    )
    monkeypatch.setattr(
        vision_module,
        "compute_qwen3_vl_visual_revision",
        lambda state_dict, config_fingerprint: "vision-revision",
    )

    config = Namespace(
        vision_encoder_device="cpu",
        vision_encoder_num_cpus=2,
        vision_encoder_cache_max_bytes=2048,
        vision_encoder_max_images_per_request=8,
        resource={"vision_encoder": [1, 0]},
        freeze_vision_model=True,
        freeze_vision_projection=True,
        hf_checkpoint="/models/qwen3-vl",
    )
    encoder_cls = getattr(vision_component.VisionEncoder, "func_or_class", vision_component.VisionEncoder)

    encoder = encoder_cls(
        healthy=object(),
        pg=None,
        num_gpus=0,
        config=config,
        role="vision_encoder",
    )

    assert str(calls["config_path"]) == "/models/qwen3-vl"
    assert str(calls["checkpoint_path"]) == "/models/qwen3-vl"
    assert calls["vision_config"] is full_config.vision_config
    assert calls["state_dict"] is loaded_state
    assert calls["strict"] is True
    assert encoder.backend.visual_model is visual_model
    assert encoder.backend.revision == "vision-revision"
    assert encoder.cache.max_bytes == 2048
    assert visual_model.training is False
