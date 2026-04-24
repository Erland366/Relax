# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import subprocess
import sys
from types import SimpleNamespace

from relax.backends.sglang.sglang_engine import _MEGATRON_ISOLATION_ENV_VAR
from relax.components.rollout import _isolate_rollout_service_from_megatron
from relax.distributed.ray import rollout as rollout_module
from relax.distributed.ray.rollout import EngineGroup


def test_engine_group_filters_megatron_from_pythonpath_for_transformers(monkeypatch):
    pythonpath = os.pathsep.join(["/tmp/relax", "/work/Megatron-LM", "/tmp/sglang"])
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    group = EngineGroup(
        args=SimpleNamespace(sglang_model_impl="transformers", num_gpus_per_node=2),
        pg=None,
        all_engines=[],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )

    env_vars = group._build_engine_runtime_env_vars()

    assert env_vars["PYTHONPATH"] == os.pathsep.join(["/tmp/relax", "/tmp/sglang"])
    assert env_vars[_MEGATRON_ISOLATION_ENV_VAR] == "1"


def test_engine_group_preserves_pythonpath_for_non_transformers(monkeypatch):
    pythonpath = os.pathsep.join(["/tmp/relax", "/work/Megatron-LM", "/tmp/sglang"])
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    group = EngineGroup(
        args=SimpleNamespace(sglang_model_impl="native", num_gpus_per_node=2),
        pg=None,
        all_engines=[],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )

    env_vars = group._build_engine_runtime_env_vars()

    assert env_vars["PYTHONPATH"] == pythonpath
    assert env_vars[_MEGATRON_ISOLATION_ENV_VAR] == "0"


def test_importing_rollout_module_does_not_import_sglang():
    script = """
import builtins

orig_import = builtins.__import__
seen = []

def hooked(name, globals=None, locals=None, fromlist=(), level=0):
    module = orig_import(name, globals, locals, fromlist, level)
    root = name.split('.')[0]
    if root == 'sglang':
        seen.append(name)
    return module

builtins.__import__ = hooked
import relax.distributed.ray.rollout
print('\\n'.join(seen))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ""


def test_rollout_manager_isolation_enables_blocker_for_transformers(monkeypatch):
    calls = []

    def fake_install(enabled, *, source, block_imports):
        calls.append((enabled, source, block_imports))
        return enabled

    monkeypatch.setattr(rollout_module, "_install_process_megatron_isolation", fake_install)

    changed = rollout_module._isolate_rollout_process_from_megatron(
        SimpleNamespace(sglang_model_impl="transformers"),
        source="test rollout manager",
    )

    assert changed is True
    assert calls == [(True, "test rollout manager", True)]


def test_rollout_service_isolation_disables_blocker_for_non_transformers(monkeypatch):
    calls = []

    def fake_install(enabled, *, source, block_imports):
        calls.append((enabled, source, block_imports))
        return enabled

    monkeypatch.setattr("relax.components.rollout._install_process_megatron_isolation", fake_install)

    changed = _isolate_rollout_service_from_megatron(SimpleNamespace(sglang_model_impl="native"))

    assert changed is False
    assert calls == [(False, "Rollout service process", True)]
