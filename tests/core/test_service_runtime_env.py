# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from enum import Enum

import pytest

from relax.core.service import build_service_config, build_service_runtime_env


class _FakeMegatronEnum(Enum):
    auto = "auto"


def _valid_feature_preload_config(**overrides):
    values = {
        "sglang_model_impl": "transformers",
        "preload_vision_features": True,
        "vision_encoder_device": "cpu",
        "vision_encoder_cache_max_bytes": 4096,
        "sglang_vision_feature_cache_max_bytes": 4096,
        "rollout_global_dataset": True,
        "use_streaming_dataset": False,
        "sglang_tokenizer_worker_num": 1,
        "rollout_num_gpus": 4,
        "rollout_num_gpus_per_engine": 4,
        "sglang_router_policy": "consistent_hashing",
    }
    values.update(overrides)
    return Namespace(**values)


def _valid_automatic_vision_device_config(tmp_path, **overrides):
    plan = tmp_path / "vision-device-plan.json"
    plan.write_text(
        """{
            "schema_version": 2,
            "initial_device": "gpu",
            "minimum_gap": 0.05,
            "time_models": {"gpu": {}, "cpu": {}},
            "cycles": {
                "0": {
                    "image_count": 1,
                    "visual_tokens": 1,
                    "feature_bytes": 8,
                    "overlap_seconds": 0.0
                }
            }
        }"""
    )
    values = {
        "vision_device_mode": "automatic",
        "vision_device_plan": str(plan),
        "vision_device_minimum_gap": None,
        "vision_encoder_device": "cpu",
        "skip_gpu_vision_encoder": False,
        "preload_vision_features": False,
        "freeze_vision_model": True,
        "freeze_vision_projection": True,
        "partial_rollout": False,
        "max_staleness": 0,
        "rollout_shuffle": False,
        "dynamic_sampling_filter_path": None,
        "rollout_batch_size": 32,
        "over_sampling_batch_size": 32,
        "sglang_model_impl": "transformers",
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_service_runtime_env_enables_megatron_isolation_only_for_rollout_transformers():
    config = Namespace(sglang_model_impl="transformers")
    runtime_env = {"env_vars": {"PYTHONPATH": "/tmp/one"}}

    result = build_service_runtime_env("rollout", config, runtime_env)

    assert result["env_vars"]["RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS"] == "1"
    assert runtime_env["env_vars"] == {"PYTHONPATH": "/tmp/one"}


def test_build_service_runtime_env_does_not_enable_megatron_isolation_for_actor():
    config = Namespace(sglang_model_impl="transformers")
    runtime_env = {"env_vars": {"PYTHONPATH": "/tmp/one"}}

    result = build_service_runtime_env("actor", config, runtime_env)

    assert "RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS" not in result["env_vars"]


def test_build_service_runtime_env_preserves_empty_runtime_env():
    config = Namespace(sglang_model_impl="sglang")

    result = build_service_runtime_env("rollout", config, None)

    assert result == {"env_vars": {}}


def test_build_service_runtime_env_enables_precomputed_vision_only_for_rollout():
    config = Namespace(sglang_model_impl="sglang", vision_encoder_device="cpu")

    rollout_result = build_service_runtime_env("rollout", config, None)
    actor_result = build_service_runtime_env("actor", config, None)

    assert rollout_result["env_vars"]["RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION"] == "1"
    assert "RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION" not in actor_result["env_vars"]


def test_build_service_runtime_env_skips_gpu_encoder_only_for_cpu_vision_rollout():
    enabled_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="cpu",
        skip_gpu_vision_encoder=True,
    )
    keep_gpu_encoder_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="cpu",
        skip_gpu_vision_encoder=False,
    )
    disabled_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="gpu",
        skip_gpu_vision_encoder=True,
    )

    enabled_rollout = build_service_runtime_env("rollout", enabled_config, None)
    keep_gpu_encoder_rollout = build_service_runtime_env("rollout", keep_gpu_encoder_config, None)
    disabled_rollout = build_service_runtime_env("rollout", disabled_config, None)
    enabled_actor = build_service_runtime_env("actor", enabled_config, None)

    skip_gpu_encoder_env_var = "RELAX_SGLANG_QWEN3_VL_SKIP_GPU_VISION_ENCODER"
    assert enabled_rollout["env_vars"][skip_gpu_encoder_env_var] == "1"
    assert skip_gpu_encoder_env_var not in keep_gpu_encoder_rollout["env_vars"]
    assert skip_gpu_encoder_env_var not in disabled_rollout["env_vars"]
    assert skip_gpu_encoder_env_var not in enabled_actor["env_vars"]


def test_build_service_runtime_env_propagates_sglang_vision_feature_cache_only_to_rollout():
    config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="cpu",
        skip_gpu_vision_encoder=True,
        sglang_vision_feature_cache_max_bytes=4096,
        sglang_tokenizer_worker_num=1,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        sglang_router_policy="consistent_hashing",
    )

    rollout_result = build_service_runtime_env("rollout", config, None)
    actor_result = build_service_runtime_env("actor", config, None)

    env_name = "RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES"
    assert rollout_result["env_vars"][env_name] == "4096"
    assert env_name not in actor_result["env_vars"]


def test_build_service_runtime_env_rejects_process_local_cache_with_multiple_tokenizer_workers():
    config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="cpu",
        sglang_vision_feature_cache_max_bytes=4096,
        sglang_tokenizer_worker_num=2,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=4,
        sglang_router_policy="consistent_hashing",
    )

    with pytest.raises(ValueError, match="tokenizer_worker_num.*1|tokenizer workers"):
        build_service_runtime_env("rollout", config, None)


def test_build_service_runtime_env_rejects_negative_sglang_vision_feature_cache_bytes():
    config = Namespace(sglang_vision_feature_cache_max_bytes=-1)

    with pytest.raises(ValueError, match="non-negative"):
        build_service_runtime_env("rollout", config, None)


def test_build_service_runtime_env_rejects_multi_engine_cache_without_feature_sticky_routing():
    config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_device="cpu",
        sglang_vision_feature_cache_max_bytes=4096,
        sglang_tokenizer_worker_num=1,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        sglang_router_policy="round_robin",
    )

    with pytest.raises(ValueError, match="feature.*consistent.*routing|feature-sticky"):
        build_service_runtime_env("rollout", config, None)


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        pytest.param(
            {"vision_encoder_device": "gpu"},
            "vision_encoder_device='cpu'|CPU vision",
            id="cpu-vision-not-selected",
        ),
        pytest.param(
            {"vision_encoder_cache_max_bytes": 0},
            "positive vision_encoder_cache_max_bytes",
            id="cpu-cache-disabled",
        ),
        pytest.param(
            {"sglang_vision_feature_cache_max_bytes": 0},
            "positive sglang_vision_feature_cache_max_bytes",
            id="sglang-cache-disabled",
        ),
        pytest.param(
            {"rollout_global_dataset": False},
            "rollout_global_dataset.*True|global dataset",
            id="global-dataset-disabled",
        ),
        pytest.param(
            {"use_streaming_dataset": True},
            "use_streaming_dataset.*False|streaming",
            id="streaming-dataset-enabled",
        ),
        pytest.param(
            {"rollout_num_gpus_per_engine": 2},
            "exactly one.*rollout engine|one SGLang rollout engine",
            id="multiple-rollout-engines",
        ),
    ],
)
def test_build_service_runtime_env_rejects_invalid_feature_preload_config(overrides, expected_error):
    config = _valid_feature_preload_config(**overrides)

    with pytest.raises(ValueError, match=expected_error):
        build_service_runtime_env("rollout", config, None)


def test_build_service_runtime_env_accepts_valid_automatic_vision_device(tmp_path):
    config = _valid_automatic_vision_device_config(tmp_path)

    result = build_service_runtime_env("rollout", config, None)

    assert result["env_vars"]["RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION"] == "1"


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"vision_encoder_device": "gpu"}, "vision_encoder_device='cpu'"),
        ({"sglang_model_impl": "sglang"}, "sglang_model_impl='transformers'"),
        ({"skip_gpu_vision_encoder": True}, "requires GPU vision weights"),
        ({"preload_vision_features": True}, "cannot preload all vision features"),
        ({"partial_rollout": True}, "does not support partial_rollout"),
        ({"max_staleness": 1}, "requires max_staleness=0"),
        ({"rollout_shuffle": True}, "requires rollout_shuffle=False"),
        ({"dynamic_sampling_filter_path": "filters.keep"}, "does not support dynamic sampling"),
        ({"over_sampling_batch_size": 64}, "over_sampling_batch_size=rollout_batch_size"),
        ({"freeze_vision_model": False}, "freeze_vision_model.*freeze_vision_projection"),
        ({"vision_device_plan": ""}, "requires vision_device_plan"),
    ],
)
def test_build_service_runtime_env_rejects_invalid_automatic_vision_device(
    tmp_path,
    overrides,
    expected_error,
):
    config = _valid_automatic_vision_device_config(tmp_path, **overrides)

    with pytest.raises(ValueError, match=expected_error):
        build_service_runtime_env("rollout", config, None)


def test_build_service_config_normalizes_enum_values_for_rollout():
    config = Namespace(attention_backend=_FakeMegatronEnum.auto, plain_value="keep")

    result = build_service_config("rollout", config)

    assert result.attention_backend == "auto"
    assert result.plain_value == "keep"
    assert config.attention_backend is _FakeMegatronEnum.auto


def test_build_service_config_leaves_non_rollout_config_unchanged():
    config = Namespace(attention_backend=_FakeMegatronEnum.auto)

    result = build_service_config("actor", config)

    assert result is config
