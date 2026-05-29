# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import os
import subprocess
import sys
import types
from types import SimpleNamespace

from relax.backends.sglang.sglang_engine import (
    _MEGATRON_ISOLATION_ENV_VAR,
    _blocked_megatron_imports,
    _filtered_pythonpath_without_megatron,
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


def test_patched_run_scheduler_process_blocks_megatron_in_scheduler_child(monkeypatch):
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
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    monkeypatch.setenv("RELAX_OPTIMIZE_ROUTING_REPLAY", "0")

    result = _patched_run_scheduler_process(SimpleNamespace(model_impl="transformers"))

    assert result == "blocked-core"


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
