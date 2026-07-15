# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import importlib.abc
import os
import sys
import types
from builtins import __import__ as _builtin_import
from contextlib import contextmanager

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


_MEGATRON_ISOLATION_ENV_VAR = "RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS"
_MEGATRON_BLOCKED_PREFIXES = ("megatron.core",)
_PROCESS_MEGATRON_IMPORT_BLOCKER = None
_PROCESS_MEGATRON_PATH_PRUNED = False
_PROCESS_MEGATRON_IMPORTS_BLOCKED = False


def _is_megatron_checkout_path(path_entry: str) -> bool:
    normalized = os.path.normpath(path_entry)
    parts = normalized.split(os.sep)
    return "Megatron-LM" in parts


def _filtered_pythonpath_without_megatron(pythonpath: str | None) -> str | None:
    if not pythonpath:
        return pythonpath

    path_entries = pythonpath.split(os.pathsep)
    filtered_entries = [entry for entry in path_entries if not _is_megatron_checkout_path(entry)]
    return os.pathsep.join(filtered_entries)


def _env_requests_megatron_isolation() -> bool:
    return os.environ.get(_MEGATRON_ISOLATION_ENV_VAR) == "1"


def _remove_megatron_from_current_process(enabled: bool) -> bool:
    if not enabled:
        return False

    changed = False

    original_pythonpath = os.environ.get("PYTHONPATH")
    filtered_pythonpath = _filtered_pythonpath_without_megatron(original_pythonpath)
    if filtered_pythonpath != original_pythonpath:
        if filtered_pythonpath:
            os.environ["PYTHONPATH"] = filtered_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)
        changed = True

    original_sys_path = list(sys.path)
    filtered_sys_path = [entry for entry in original_sys_path if not _is_megatron_checkout_path(entry)]
    if filtered_sys_path != original_sys_path:
        sys.path[:] = filtered_sys_path
        changed = True

    return changed


def _is_megatron_editable_finder(obj: object) -> bool:
    module_name = getattr(type(obj), "__module__", "") or getattr(obj, "__module__", "")
    return module_name.startswith("__editable___megatron_core_")


def _is_megatron_editable_path_entry(path_entry: object) -> bool:
    if not isinstance(path_entry, str):
        return False
    return "__editable__.megatron_core-" in path_entry and ".__path_hook__" in path_entry


def _is_blocked_megatron_module(fullname: str) -> bool:
    return any(fullname == prefix or fullname.startswith(f"{prefix}.") for prefix in _MEGATRON_BLOCKED_PREFIXES)


def _prune_megatron_import_state(enabled: bool, *, source: str) -> bool:
    global _PROCESS_MEGATRON_PATH_PRUNED

    if not enabled or _PROCESS_MEGATRON_PATH_PRUNED:
        return False

    changed = _remove_megatron_from_current_process(True)

    original_meta_path = list(sys.meta_path)
    original_path_hooks = list(sys.path_hooks)
    original_sys_path = list(sys.path)
    original_path_importer_cache = dict(sys.path_importer_cache)
    blocked_modules = [
        name
        for name in list(sys.modules)
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _MEGATRON_BLOCKED_PREFIXES)
    ]
    editable_finder_modules = [
        name
        for name in list(sys.modules)
        if name.startswith("__editable___megatron_core_") and name.endswith("_finder")
    ]

    for name in blocked_modules + editable_finder_modules:
        sys.modules.pop(name, None)

    filtered_meta_path = [finder for finder in original_meta_path if not _is_megatron_editable_finder(finder)]
    if filtered_meta_path != original_meta_path:
        sys.meta_path[:] = filtered_meta_path
        changed = True

    filtered_path_hooks = [hook for hook in original_path_hooks if not _is_megatron_editable_finder(hook)]
    if filtered_path_hooks != original_path_hooks:
        sys.path_hooks[:] = filtered_path_hooks
        changed = True

    filtered_sys_path = [entry for entry in original_sys_path if not _is_megatron_editable_path_entry(entry)]
    if filtered_sys_path != original_sys_path:
        sys.path[:] = filtered_sys_path
        changed = True

    filtered_importer_cache = {
        key: value for key, value in original_path_importer_cache.items() if not _is_megatron_editable_path_entry(key)
    }
    if filtered_importer_cache != original_path_importer_cache:
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(filtered_importer_cache)
        changed = True

    if blocked_modules or editable_finder_modules:
        changed = True

    if changed:
        logger.info("Removed Megatron-LM from %s PYTHONPATH and sys.path", source)
    _PROCESS_MEGATRON_PATH_PRUNED = True
    return True


class _MegatronImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _is_blocked_megatron_module(fullname):
            raise ModuleNotFoundError(f"Blocked import of {fullname} for SGLang transformers backend")
        return None


@contextmanager
def _blocked_megatron_imports(enabled: bool):
    if not enabled:
        yield
        return

    blocker = _MegatronImportBlocker()
    original_meta_path = list(sys.meta_path)
    original_path_hooks = list(sys.path_hooks)
    original_sys_path = list(sys.path)
    original_path_importer_cache = dict(sys.path_importer_cache)
    original_import = _builtin_import
    original_import_module = importlib.import_module
    blocked_modules = {name: module for name, module in list(sys.modules.items()) if _is_blocked_megatron_module(name)}
    editable_finder_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith("__editable___megatron_core_") and name.endswith("_finder")
    }
    for name in blocked_modules:
        sys.modules.pop(name, None)
    for name in editable_finder_modules:
        sys.modules.pop(name, None)

    original_megatron_module = sys.modules.get("megatron")
    megatron_stub = types.ModuleType("megatron")
    megatron_stub.__path__ = []
    sys.modules["megatron"] = megatron_stub

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if _is_blocked_megatron_module(name):
            raise ModuleNotFoundError(f"Blocked import of {name} for SGLang transformers backend")
        return original_import(name, globals, locals, fromlist, level)

    def _blocked_import_module(name, package=None):
        if _is_blocked_megatron_module(name):
            raise ModuleNotFoundError(f"Blocked import of {name} for SGLang transformers backend")
        return original_import_module(name, package)

    import builtins

    builtins.__import__ = _blocked_import
    importlib.import_module = _blocked_import_module

    sys.meta_path[:] = [finder for finder in original_meta_path if not _is_megatron_editable_finder(finder)]
    sys.path_hooks[:] = [hook for hook in original_path_hooks if not _is_megatron_editable_finder(hook)]
    sys.path[:] = [entry for entry in original_sys_path if not _is_megatron_editable_path_entry(entry)]
    sys.path_importer_cache.clear()
    sys.path_importer_cache.update(
        {
            key: value
            for key, value in original_path_importer_cache.items()
            if not _is_megatron_editable_path_entry(key)
        }
    )
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        import builtins

        builtins.__import__ = original_import
        importlib.import_module = original_import_module
        sys.meta_path[:] = original_meta_path
        sys.path_hooks[:] = original_path_hooks
        sys.path[:] = original_sys_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(original_path_importer_cache)
        if original_megatron_module is None:
            sys.modules.pop("megatron", None)
        else:
            sys.modules["megatron"] = original_megatron_module
        sys.modules.update(blocked_modules)
        sys.modules.update(editable_finder_modules)


def _install_process_megatron_isolation(enabled: bool, *, source: str, block_imports: bool) -> bool:
    global _PROCESS_MEGATRON_IMPORT_BLOCKER, _PROCESS_MEGATRON_IMPORTS_BLOCKED

    changed = _prune_megatron_import_state(enabled, source=source)
    if not enabled or not block_imports or _PROCESS_MEGATRON_IMPORTS_BLOCKED:
        return changed

    _PROCESS_MEGATRON_IMPORT_BLOCKER = _blocked_megatron_imports(True)
    _PROCESS_MEGATRON_IMPORT_BLOCKER.__enter__()
    logger.info("Installed Megatron import blocker in %s", source)
    _PROCESS_MEGATRON_IMPORTS_BLOCKED = True
    return True


@contextmanager
def _temporary_pythonpath_without_megatron(enabled: bool):
    """Temporarily remove local Megatron-LM checkouts from child paths."""
    if not enabled:
        yield
        return

    original_pythonpath = os.environ.get("PYTHONPATH")
    original_sys_path = list(sys.path)

    changed = _remove_megatron_from_current_process(True)

    if changed:
        logger.info("Removed Megatron-LM from PYTHONPATH and sys.path for SGLang transformers backend")
    try:
        yield
    finally:
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath
        sys.path[:] = original_sys_path
