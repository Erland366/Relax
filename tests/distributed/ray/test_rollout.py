# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import subprocess
import sys
from types import SimpleNamespace

from relax.distributed.ray.rollout import EngineGroup


def test_engine_group_preserves_pythonpath_for_transformers(monkeypatch):
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

    assert env_vars["PYTHONPATH"] == pythonpath
    assert "RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS" not in env_vars


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
    assert "RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS" not in env_vars


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
