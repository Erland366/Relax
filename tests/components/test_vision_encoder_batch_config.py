# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from argparse import ArgumentParser, ArgumentTypeError, Namespace

import pytest
import torch


def _vision_component():
    return importlib.import_module("relax.components.vision_encoder")


def _valid_cpu_vision_config(**overrides):
    values = {
        "vision_encoder_device": "cpu",
        "vision_encoder_num_replicas": 1,
        "vision_encoder_num_cpus": 1,
        "vision_encoder_cache_max_bytes": 1024,
        "vision_encoder_max_images_per_request": 8,
        "vision_encoder_batch_wait_timeout_ms": 0.0,
        "resource": {"vision_encoder": [1, 0]},
        "freeze_vision_model": True,
        "freeze_vision_projection": True,
        "hf_checkpoint": "/models/qwen3-vl",
    }
    values.update(overrides)
    return Namespace(**values)


def test_vision_encoder_batch_wait_argument_defaults_to_disabled():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--vision-encoder-batch-wait-timeout-ms" in action.option_strings
    ]

    assert len(actions) == 1, "--vision-encoder-batch-wait-timeout-ms must be registered"
    assert actions[0].default == 0.0


def test_sglang_vision_feature_cache_argument_defaults_to_disabled():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--sglang-vision-feature-cache-max-bytes" in action.option_strings
    ]

    assert len(actions) == 1, "--sglang-vision-feature-cache-max-bytes must be registered"
    assert actions[0].default == 0


def test_preload_vision_features_argument_is_explicitly_opt_in():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--preload-vision-features" in action.option_strings
    ]

    assert len(actions) == 1, "--preload-vision-features must be registered"
    assert actions[0].default is False
    required_args = ["--rollout-batch-size", "1"]
    assert parser.parse_args(required_args).preload_vision_features is False
    assert parser.parse_args([*required_args, "--preload-vision-features"]).preload_vision_features is True


def test_sglang_vision_feature_cache_argument_rejects_negative_bytes():
    arguments = importlib.import_module("relax.utils.arguments")
    parser = arguments.get_slime_extra_args_provider()(ArgumentParser())

    actions = [
        action
        for action in parser._actions
        if "--sglang-vision-feature-cache-max-bytes" in action.option_strings
    ]

    assert len(actions) == 1, "--sglang-vision-feature-cache-max-bytes must be registered"
    with pytest.raises(ArgumentTypeError, match="non-negative"):
        actions[0].type("-1")


@pytest.mark.parametrize("timeout_ms", [-0.001, float("nan"), float("inf"), float("-inf")])
def test_vision_encoder_batch_wait_rejects_negative_or_non_finite_values(timeout_ms):
    config = _valid_cpu_vision_config(vision_encoder_batch_wait_timeout_ms=timeout_ms)

    with pytest.raises(ValueError, match="vision_encoder_batch_wait_timeout_ms"):
        _vision_component().validate_vision_encoder_config(config)


def test_vision_encoder_batch_wait_rejects_positive_values_until_batching_is_implemented():
    config = _valid_cpu_vision_config(vision_encoder_batch_wait_timeout_ms=1.0)

    with pytest.raises(NotImplementedError, match="not implemented; use 0"):
        _vision_component().validate_vision_encoder_config(config)


def test_zero_batch_wait_keeps_one_request_per_backend_forward(monkeypatch):
    vision_component = _vision_component()
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    features = vision_module.Qwen3VLFrozenVisionFeatures(
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
            return features

    backend = FakeBackend()
    monkeypatch.setattr(vision_component.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(
        vision_component,
        "build_qwen3_vl_cpu_vision_backend",
        lambda *args, **kwargs: backend,
    )
    encoder_cls = getattr(vision_component.VisionEncoder, "func_or_class", vision_component.VisionEncoder)
    encoder = encoder_cls(
        healthy=object(),
        pg=None,
        num_gpus=0,
        config=_valid_cpu_vision_config(),
        role="vision_encoder",
    )

    assert encoder.batch_wait_timeout_ms == 0.0

    encoder.encode(pixel_values=torch.zeros((4, 2)), image_grid_thw=image_grid_thw)
    assert backend.calls == 1
    encoder.encode(pixel_values=torch.ones((4, 2)), image_grid_thw=image_grid_thw)
    assert backend.calls == 2
