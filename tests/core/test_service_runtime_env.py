# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from enum import Enum

import pytest

from relax.core.service import build_service_config, build_service_runtime_env


class _FakeMegatronEnum(Enum):
    auto = "auto"


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
    config = Namespace(sglang_model_impl="sglang", vision_encoder_backend="pytorch")

    rollout_result = build_service_runtime_env("rollout", config, None)
    actor_result = build_service_runtime_env("actor", config, None)

    assert rollout_result["env_vars"]["RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION"] == "1"
    assert "RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION" not in actor_result["env_vars"]


def test_build_service_runtime_env_enables_gpu_vision_omission_only_for_cpu_vision_rollout():
    enabled_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_backend="pytorch",
        vision_encoder_omit_gpu_weights=True,
    )
    resident_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_backend="pytorch",
        vision_encoder_omit_gpu_weights=False,
    )
    disabled_config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_backend="disabled",
        vision_encoder_omit_gpu_weights=True,
    )

    enabled_rollout = build_service_runtime_env("rollout", enabled_config, None)
    resident_rollout = build_service_runtime_env("rollout", resident_config, None)
    disabled_rollout = build_service_runtime_env("rollout", disabled_config, None)
    enabled_actor = build_service_runtime_env("actor", enabled_config, None)

    omission_env_var = "RELAX_SGLANG_QWEN3_VL_OMIT_GPU_WEIGHTS"
    assert enabled_rollout["env_vars"][omission_env_var] == "1"
    assert omission_env_var not in resident_rollout["env_vars"]
    assert omission_env_var not in disabled_rollout["env_vars"]
    assert omission_env_var not in enabled_actor["env_vars"]


def test_build_service_runtime_env_propagates_sglang_vision_feature_cache_only_to_rollout():
    config = Namespace(
        sglang_model_impl="transformers",
        vision_encoder_backend="pytorch",
        vision_encoder_omit_gpu_weights=True,
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
        vision_encoder_backend="pytorch",
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
        vision_encoder_backend="pytorch",
        sglang_vision_feature_cache_max_bytes=4096,
        sglang_tokenizer_worker_num=1,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        sglang_router_policy="round_robin",
    )

    with pytest.raises(ValueError, match="feature.*consistent.*routing|feature-sticky"):
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
