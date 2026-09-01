# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import builtins
import importlib
import os
import subprocess
import sys
import types
from types import SimpleNamespace

from relax.backends.sglang.sglang_engine import (
    _MEGATRON_ISOLATION_ENV_VAR,
    _blocked_megatron_imports,
    _disable_sglang_jit_store_cache_on_hip,
    _filtered_pythonpath_without_megatron,
    _get_server_args_cls,
    _patched_run_scheduler_process,
    _remove_megatron_from_current_process,
    _temporary_pythonpath_without_megatron,
    launch_server_process,
)


def test_filtered_pythonpath_without_megatron_removes_checkout_entries():
    pythonpath = os.pathsep.join(["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"])

    assert _filtered_pythonpath_without_megatron(pythonpath) == os.pathsep.join(["/tmp/pkg", "/tmp/other"])


def test_filtered_pythonpath_without_megatron_preserves_empty_value():
    assert _filtered_pythonpath_without_megatron(None) is None


def test_remove_megatron_from_current_process_filters_pythonpath_and_sys_path(monkeypatch):
    original = os.pathsep.join(["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setattr(sys, "path", ["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"])

    changed = _remove_megatron_from_current_process(True)

    assert changed is True
    assert os.environ["PYTHONPATH"] == os.pathsep.join(["/tmp/pkg", "/tmp/other"])
    assert sys.path == ["/tmp/pkg", "/tmp/other"]


def test_remove_megatron_from_current_process_noops_when_disabled(monkeypatch):
    original = os.pathsep.join(["/tmp/pkg", "/work/Megatron-LM"])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setattr(sys, "path", ["/tmp/pkg", "/work/Megatron-LM"])

    changed = _remove_megatron_from_current_process(False)

    assert changed is False
    assert os.environ["PYTHONPATH"] == original
    assert sys.path == ["/tmp/pkg", "/work/Megatron-LM"]


def test_blocked_megatron_imports_raises_module_not_found():
    with _blocked_megatron_imports(True):
        megatron_module = importlib.import_module("megatron")
        assert list(megatron_module.__path__) == []
        try:
            importlib.import_module("megatron.core")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("megatron.core import should be blocked")


def test_blocked_megatron_imports_temporarily_removes_editable_finder(monkeypatch):
    class _EditableFinder:
        pass

    _EditableFinder.__module__ = "__editable___megatron_core_0_16_0rc0_finder"
    editable_finder = _EditableFinder()
    editable_path_entry = "__editable__.megatron_core-0.16.0rc0.finder.__path_hook__"

    monkeypatch.setattr(sys, "meta_path", [editable_finder, object()])
    monkeypatch.setattr(sys, "path_hooks", [editable_finder, object()])
    monkeypatch.setattr(sys, "path", ["/tmp/pkg", editable_path_entry])
    monkeypatch.setattr(sys, "path_importer_cache", {editable_path_entry: object(), "/tmp/pkg": object()})
    monkeypatch.setitem(
        sys.modules,
        "__editable___megatron_core_0_16_0rc0_finder",
        types.ModuleType("__editable___megatron_core_0_16_0rc0_finder"),
    )

    with _blocked_megatron_imports(True):
        assert editable_finder not in sys.meta_path
        assert editable_finder not in sys.path_hooks
        assert editable_path_entry not in sys.path
        assert editable_path_entry not in sys.path_importer_cache
        assert "__editable___megatron_core_0_16_0rc0_finder" not in sys.modules

    assert editable_finder in sys.meta_path
    assert editable_finder in sys.path_hooks
    assert editable_path_entry in sys.path
    assert editable_path_entry in sys.path_importer_cache
    assert "__editable___megatron_core_0_16_0rc0_finder" in sys.modules


def test_temporary_pythonpath_without_megatron_filters_child_path(monkeypatch):
    original = os.pathsep.join(["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setattr(sys, "path", ["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"])

    with _temporary_pythonpath_without_megatron(True):
        assert os.environ["PYTHONPATH"] == os.pathsep.join(["/tmp/pkg", "/tmp/other"])
        assert sys.path == ["/tmp/pkg", "/tmp/other"]

    assert os.environ["PYTHONPATH"] == original
    assert sys.path == ["/tmp/pkg", "/work/Megatron-LM", "/tmp/other"]


def test_temporary_pythonpath_without_megatron_leaves_path_when_disabled(monkeypatch):
    original = os.pathsep.join(["/tmp/pkg", "/work/Megatron-LM"])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setattr(sys, "path", ["/tmp/pkg", "/work/Megatron-LM"])

    with _temporary_pythonpath_without_megatron(False):
        assert os.environ["PYTHONPATH"] == original
        assert sys.path == ["/tmp/pkg", "/work/Megatron-LM"]

    assert os.environ["PYTHONPATH"] == original
    assert sys.path == ["/tmp/pkg", "/work/Megatron-LM"]


def test_module_import_env_blocks_megatron_before_actor_init():
    script = f"""
import importlib
import os

os.environ[{_MEGATRON_ISOLATION_ENV_VAR!r}] = "1"
import relax.backends.sglang.sglang_engine

megatron_module = importlib.import_module("megatron")
print("namespace", list(megatron_module.__path__))
try:
    importlib.import_module("megatron.core")
except ModuleNotFoundError:
    print("blocked")
else:
    print("not-blocked")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "namespace" in result.stdout
    assert result.stdout.strip().endswith("blocked")


def test_checkpoint_client_imports_under_megatron_blocker():
    script = """
from relax.backends.sglang.sglang_engine import _blocked_megatron_imports

with _blocked_megatron_imports(True):
    from relax.distributed.checkpoint_service.client.engine import create_client
    from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend

    print(create_client.__name__, DeviceDirectBackend.__name__)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert "create_client DeviceDirectBackend" in result.stdout


def test_get_server_args_installs_sgl_kernel_stub_before_import(monkeypatch):
    import relax.backends.sglang.sglang_engine as sglang_engine

    calls = []

    def fake_install_sgl_kernel_stub():
        calls.append("install")

    class FakeServerArgs:
        pass

    sglang_module = types.ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = types.ModuleType("sglang.srt")
    srt_module.__path__ = []
    server_args_module = types.ModuleType("sglang.srt.server_args")
    server_args_module.ServerArgs = FakeServerArgs
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)
    monkeypatch.setattr(sglang_engine, "_install_optional_sgl_kernel_stub_on_hip", fake_install_sgl_kernel_stub)

    original_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.server_args":
            calls.append("import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    assert _get_server_args_cls() is FakeServerArgs
    assert calls == ["install", "import"]


def test_precomputed_vision_startup_patch_also_skips_gpu_vision_encoder(monkeypatch):
    import relax.backends.sglang.precomputed_vision as precomputed_vision
    import relax.backends.sglang.sglang_engine as sglang_engine

    calls = []

    class FakeQwen3VLModel:
        pass

    class FakeTransformersBase:
        pass

    qwen_modeling_module = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")
    qwen_modeling_module.Qwen3VLModel = FakeQwen3VLModel
    sglang_transformers_module = types.ModuleType("sglang.srt.models.transformers")
    sglang_transformers_module.TransformersBase = FakeTransformersBase
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        qwen_modeling_module,
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.models.transformers", sglang_transformers_module)
    monkeypatch.setattr(
        precomputed_vision,
        "install_qwen3_vl_precomputed_vision_patch",
        lambda: calls.append(("precomputed",)),
    )

    def record_skip_gpu_encoder_patch(*, qwen3_vl_model_cls, transformers_base_cls, enabled):
        calls.append(("skip_gpu_encoder", qwen3_vl_model_cls, transformers_base_cls, enabled))

    monkeypatch.setattr(
        precomputed_vision,
        "patch_qwen3_vl_transformers_to_skip_gpu_vision_encoder",
        record_skip_gpu_encoder_patch,
    )
    monkeypatch.setenv("RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION", "1")
    monkeypatch.setenv("RELAX_SGLANG_QWEN3_VL_SKIP_GPU_VISION_ENCODER", "1")

    sglang_engine._maybe_install_precomputed_vision_patch()

    assert calls == [
        ("precomputed",),
        ("skip_gpu_encoder", FakeQwen3VLModel, FakeTransformersBase, True),
    ]


def test_precomputed_vision_startup_patch_does_not_import_precomputed_only_classes_when_disabled(monkeypatch):
    import relax.backends.sglang.precomputed_vision as precomputed_vision
    import relax.backends.sglang.sglang_engine as sglang_engine

    calls = []
    blocked_imports = {
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "sglang.srt.models.transformers",
    }
    original_import = builtins.__import__

    def fail_if_precomputed_only_class_is_imported(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_imports:
            raise AssertionError(f"precomputed-only class imported while GPU encoder skipping is disabled: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        precomputed_vision,
        "install_qwen3_vl_precomputed_vision_patch",
        lambda: calls.append("precomputed"),
    )
    monkeypatch.setattr(
        precomputed_vision,
        "patch_qwen3_vl_transformers_to_skip_gpu_vision_encoder",
        lambda **kwargs: calls.append("skip_gpu_encoder"),
    )
    monkeypatch.setattr(builtins, "__import__", fail_if_precomputed_only_class_is_imported)
    monkeypatch.setenv("RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION", "1")
    monkeypatch.delenv("RELAX_SGLANG_QWEN3_VL_SKIP_GPU_VISION_ENCODER", raising=False)

    sglang_engine._maybe_install_precomputed_vision_patch()

    assert calls == ["precomputed"]


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


def test_patched_run_scheduler_process_blocks_megatron_in_scheduler_child(monkeypatch):
    _install_fake_sglang_memory_pool(monkeypatch)
    managers_module = types.ModuleType("sglang.srt.managers")
    managers_module.__path__ = []
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    def fake_run_scheduler_process(*args, **kwargs):
        megatron_module = importlib.import_module("megatron")
        assert megatron_module is not None
        try:
            importlib.import_module("megatron.core")
        except ModuleNotFoundError:
            return "blocked-core"
        raise AssertionError("megatron.core import should be blocked inside scheduler child")

    scheduler_module.run_scheduler_process = fake_run_scheduler_process
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    monkeypatch.setenv("RELAX_OPTIMIZE_ROUTING_REPLAY", "0")

    result = _patched_run_scheduler_process(SimpleNamespace(model_impl="transformers"))

    assert result == "blocked-core"


def test_disable_sglang_jit_store_cache_on_hip_patches_memory_pool(monkeypatch):
    import torch

    memory_pool_module, forward_batch_info_module = _install_fake_sglang_memory_pool(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "6.3.0", raising=False)

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
