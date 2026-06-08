# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from types import SimpleNamespace

from relax.backends.sglang.sglang_engine import (
    _disable_sglang_jit_store_cache_on_hip,
    _patched_run_scheduler_process,
    launch_server_process,
)


def _install_fake_torch(monkeypatch, hip_version: str | None = "6.3.0"):
    torch_module = types.ModuleType("torch")
    torch_module.version = SimpleNamespace(hip=hip_version)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    return torch_module


def _install_fake_sglang_memory_pool(monkeypatch):
    sglang_module = types.ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = types.ModuleType("sglang.srt")
    srt_module.__path__ = []
    mem_cache_module = types.ModuleType("sglang.srt.mem_cache")
    mem_cache_module.__path__ = []
    memory_pool_module = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool_module.can_use_store_cache = lambda size: True
    mem_cache_module.memory_pool = memory_pool_module
    model_executor_module = types.ModuleType("sglang.srt.model_executor")
    model_executor_module.__path__ = []
    forward_batch_info_module = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    native_clamp_position = object()
    forward_batch_info_module._clamp_position_native = native_clamp_position
    forward_batch_info_module.clamp_position = object()
    model_executor_module.forward_batch_info = forward_batch_info_module

    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache", mem_cache_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.memory_pool", memory_pool_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", model_executor_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info", forward_batch_info_module)
    return memory_pool_module, forward_batch_info_module


def test_patched_run_scheduler_process_preserves_megatron_imports(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_sglang_memory_pool(monkeypatch)
    managers_module = types.ModuleType("sglang.srt.managers")
    managers_module.__path__ = []
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")
    megatron_module = types.ModuleType("megatron")
    megatron_module.__path__ = []
    megatron_core_module = types.ModuleType("megatron.core")

    def fake_run_scheduler_process(*args, **kwargs):
        assert importlib.import_module("megatron") is megatron_module
        assert importlib.import_module("megatron.core") is megatron_core_module
        return "megatron-available"

    scheduler_module.run_scheduler_process = fake_run_scheduler_process
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_module)
    monkeypatch.setenv("RELAX_OPTIMIZE_ROUTING_REPLAY", "0")

    result = _patched_run_scheduler_process(SimpleNamespace(model_impl="transformers"))

    assert result == "megatron-available"


def test_disable_sglang_jit_store_cache_on_hip_patches_memory_pool(monkeypatch):
    _install_fake_torch(monkeypatch)
    memory_pool_module, forward_batch_info_module = _install_fake_sglang_memory_pool(monkeypatch)

    changed = _disable_sglang_jit_store_cache_on_hip()

    assert changed is True
    assert memory_pool_module.can_use_store_cache(2048) is False
    assert memory_pool_module._RELAX_DISABLE_JIT_STORE_CACHE_ON_HIP is True
    assert forward_batch_info_module.clamp_position is forward_batch_info_module._clamp_position_native
    assert forward_batch_info_module._RELAX_DISABLE_JIT_CLAMP_POSITION_ON_HIP is True
    assert _disable_sglang_jit_store_cache_on_hip() is False


def test_launch_server_process_kills_tree_when_health_check_fails(monkeypatch):
    killed_pids = []

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.pid = 43210

        def start(self):
            return None

        def is_alive(self):
            return True

    def fake_wait_server_healthy(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("relax.backends.sglang.sglang_engine.multiprocessing.Process", FakeProcess)
    monkeypatch.setattr("relax.backends.sglang.sglang_engine._wait_server_healthy", fake_wait_server_healthy)
    monkeypatch.setattr("relax.backends.sglang.sglang_engine._kill_process_tree", killed_pids.append)

    try:
        launch_server_process(
            SimpleNamespace(
                model_impl="transformers",
                host="127.0.0.1",
                node_rank=0,
                api_key=None,
                url=lambda: "http://127.0.0.1:8000",
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("launch_server_process should re-raise health-check failures")

    assert killed_pids == [43210]
