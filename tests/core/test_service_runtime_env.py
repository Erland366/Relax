# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from enum import Enum

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
