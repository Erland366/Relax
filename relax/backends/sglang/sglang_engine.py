# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import importlib
import importlib.abc
import ipaddress
import multiprocessing
import os
import signal
import sys
import threading
import time
import types
from builtins import __import__ as _builtin_import
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

import ray
import requests
from packaging.version import parse
from urllib3.exceptions import NewConnectionError

from relax.distributed.ray.ray_actor import RayActor
from relax.utils import device as device_utils
from relax.utils.async_utils import run
from relax.utils.http_utils import get_host_info
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


_MEGATRON_ISOLATION_ENV_VAR = "RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS"
_PROCESS_MEGATRON_IMPORT_BLOCKER = None
_PROCESS_MEGATRON_PATH_PRUNED = False
_PROCESS_MEGATRON_IMPORTS_BLOCKED = False
_PROCESS_SGL_KERNEL_STUB_INSTALLED = False
_MEGATRON_BLOCKED_PREFIXES = ("megatron.core",)
_DISABLE_MEMORY_SAVER_ENV_VAR = "RELAX_SGLANG_DISABLE_MEMORY_SAVER"


if TYPE_CHECKING:
    pass


def _get_server_args_cls():
    _install_optional_sgl_kernel_stub_on_hip()

    from sglang.srt.server_args import ServerArgs

    return ServerArgs


def _get_sglang_router():
    import sglang_router

    return sglang_router


def _sglang_memory_saver_enabled(offload_rollout: bool) -> bool:
    if not offload_rollout:
        return False

    if os.environ.get(_DISABLE_MEMORY_SAVER_ENV_VAR) != "1":
        return True

    logger.info("Disabled SGLang memory saver via %s=1", _DISABLE_MEMORY_SAVER_ENV_VAR)
    return False


def _kill_process_tree(pid: int) -> None:
    from sglang.srt.utils import kill_process_tree

    kill_process_tree(pid)


def _create_checkpoint_client(**kwargs):
    from relax.distributed.checkpoint_service.client.engine import create_client

    return create_client(**kwargs)


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


def _is_megatron_editable_finder(obj: object) -> bool:
    module_name = getattr(type(obj), "__module__", "") or getattr(obj, "__module__", "")
    return module_name.startswith("__editable___megatron_core_")


def _is_megatron_editable_path_entry(path_entry: object) -> bool:
    if not isinstance(path_entry, str):
        return False
    return "__editable__.megatron_core-" in path_entry and ".__path_hook__" in path_entry


def _is_blocked_megatron_module(fullname: str) -> bool:
    return any(fullname == prefix or fullname.startswith(f"{prefix}.") for prefix in _MEGATRON_BLOCKED_PREFIXES)


def _prepend_rocm_sgl_kernel_stub_path() -> bool:
    stub_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "rocm_stubs"))
    if not os.path.isdir(stub_root):
        raise FileNotFoundError(f"ROCm sgl_kernel stub path does not exist: {stub_root}")

    changed = False
    if stub_root not in sys.path:
        sys.path.insert(0, stub_root)
        changed = True

    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if stub_root not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([stub_root, *pythonpath_entries])
        changed = True

    return changed


def _install_sgl_kernel_stub_on_hip(import_error: BaseException) -> bool:
    global _PROCESS_SGL_KERNEL_STUB_INSTALLED

    changed = _prepend_rocm_sgl_kernel_stub_path()
    if _PROCESS_SGL_KERNEL_STUB_INSTALLED:
        return changed

    unavailable_message = (
        "sgl_kernel is unavailable on this ROCm runtime. This Relax path only stubs it to let "
        "SGLang import optional CUDA-only LoRA/MoE modules for dense transformers serving. "
        f"Original import error: {import_error}"
    )

    class _UnavailableSglKernelSymbol:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            raise RuntimeError(unavailable_message)

        def __bool__(self):
            return False

        def __repr__(self):
            return "<unavailable ROCm sgl_kernel symbol>"

    class _SglKernelStubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__path__ = []
            module.__getattr__ = lambda name: _UnavailableSglKernelSymbol()

    class _SglKernelStubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                return importlib.machinery.ModuleSpec(fullname, _SglKernelStubLoader(), is_package=True)
            return None

    for name in list(sys.modules):
        if name == "sgl_kernel" or name.startswith("sgl_kernel."):
            sys.modules.pop(name, None)

    sys.meta_path.insert(0, _SglKernelStubFinder())
    _PROCESS_SGL_KERNEL_STUB_INSTALLED = True
    logger.warning("Installed ROCm sgl_kernel import stub for optional SGLang CUDA-only modules")
    return True


def _install_optional_sgl_kernel_stub_on_hip() -> bool:
    import torch

    if torch.version.hip is None:
        return False

    try:
        import sgl_kernel  # noqa: F401

        return False
    except (ImportError, OSError) as exc:
        return _install_sgl_kernel_stub_on_hip(exc)


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


@contextmanager
def _temporary_pythonpath_without_megatron(enabled: bool):
    """Temporarily remove local Megatron-LM checkouts from ``PYTHONPATH``.

    SGLang's Transformers backend should not accidentally discover a local
    Megatron checkout and switch into Megatron-FSDP code paths during rollout
    engine startup. Because the SGLang server is launched via Python
    ``multiprocessing``, we must filter both ``PYTHONPATH`` and the current
    interpreter's ``sys.path`` so the spawned child cannot inherit the local
    checkout through either mechanism.
    """
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


_install_process_megatron_isolation(
    _env_requests_megatron_isolation(),
    source="sglang_engine module import",
    block_imports=True,
)


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    visible_env = device_utils.get_visible_devices_env_var()
    cvd = os.environ.get(visible_env)
    if not cvd:
        return physical_gpu_id  # no remapping
    # Visible devices can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"Device id {physical_gpu_id} is not valid under {visible_env}={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
    )


def _disable_sglang_jit_store_cache_on_hip() -> bool:
    import torch

    if torch.version.hip is None:
        return False

    changed = _install_optional_sgl_kernel_stub_on_hip()

    from sglang.srt.mem_cache import memory_pool

    if not getattr(memory_pool, "_RELAX_DISABLE_JIT_STORE_CACHE_ON_HIP", False):

        def _can_use_store_cache_on_hip(size: int) -> bool:
            return False

        memory_pool.can_use_store_cache = _can_use_store_cache_on_hip
        memory_pool._RELAX_DISABLE_JIT_STORE_CACHE_ON_HIP = True
        logger.info("Disabled SGLang JIT KV-cache store on ROCm; using tensor assignment fallback")
        changed = True

    from sglang.srt.model_executor import forward_batch_info

    if not hasattr(forward_batch_info, "_clamp_position_native"):
        raise RuntimeError("SGLang forward_batch_info is missing _clamp_position_native fallback")

    if not getattr(forward_batch_info, "_RELAX_DISABLE_JIT_CLAMP_POSITION_ON_HIP", False):
        forward_batch_info.clamp_position = forward_batch_info._clamp_position_native
        forward_batch_info._RELAX_DISABLE_JIT_CLAMP_POSITION_ON_HIP = True
        logger.info("Disabled SGLang JIT clamp_position on ROCm; using torch fallback")
        changed = True

    return changed


def _patched_run_scheduler_process(*args, **kwargs):
    """Run the scheduler entrypoint under the same Megatron isolation used by
    the top-level SGLang server process.

    SGLang launches scheduler workers via ``multiprocessing`` with the
    ``spawn`` start method, so those workers start in fresh interpreters that
    do not inherit the parent process' temporary import blocker.  If we only
    guard the top-level ``launch_server`` call, the scheduler can still import
    the editable ``megatron_core`` package from the environment and switch into
    Megatron-FSDP code paths.  Apply the blocker again here before importing
    SGLang's scheduler module.
    """
    server_args = args[0]
    enabled = server_args.model_impl.lower() == "transformers"
    optimize_routing_replay = os.environ.get("RELAX_OPTIMIZE_ROUTING_REPLAY", "0") == "1"

    with _blocked_megatron_imports(enabled), _temporary_pythonpath_without_megatron(enabled):
        _disable_sglang_jit_store_cache_on_hip()

        if optimize_routing_replay:
            from relax.backends.sglang.routing_replay_patch import apply_patch

            apply_patch()

        from sglang.srt.managers.scheduler import run_scheduler_process

        return run_scheduler_process(*args, **kwargs)


def _launch_server_with_patch(server_args):
    """Top-level picklable target for ``multiprocessing.Process`` when the
    async D→H optimisation is enabled.

    Passes ``_patched_run_scheduler_process`` into ``launch_server`` so that
    every scheduler subprocess applies the monkey-patch.
    """
    enabled = server_args.model_impl.lower() == "transformers"
    with _blocked_megatron_imports(enabled), _temporary_pythonpath_without_megatron(enabled):
        _disable_sglang_jit_store_cache_on_hip()

        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(
            server_args,
            run_scheduler_process_func=_patched_run_scheduler_process,
        )


def _launch_server(server_args):
    enabled = server_args.model_impl.lower() == "transformers"
    with _blocked_megatron_imports(enabled), _temporary_pythonpath_without_megatron(enabled):
        _disable_sglang_jit_store_cache_on_hip()

        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args, run_scheduler_process_func=_patched_run_scheduler_process)


def launch_server_process(server_args) -> multiprocessing.Process:
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")

    optimize = os.environ.get("RELAX_OPTIMIZE_ROUTING_REPLAY", "0") == "1"
    if optimize:
        logger.info("Launching SGLang server with async D→H routing-replay patch")
        target_func = _launch_server_with_patch
    else:
        target_func = _launch_server

    with _temporary_pythonpath_without_megatron(server_args.model_impl.lower() == "transformers"):
        p = multiprocessing.Process(target=target_func, args=(server_args,))
        p.start()

    if server_args.node_rank != 0:
        return

    try:
        _wait_server_healthy(
            base_url=server_args.url(),
            api_key=server_args.api_key,
            is_process_alive=lambda: p.is_alive(),
        )
    except BaseException:
        # Failed SGLang startup can leave scheduler / detokenizer descendants
        # alive after the engine actor exits. Tear down the full tree here
        # before propagating the init error back to Ray.
        try:
            _kill_process_tree(p.pid)
        except Exception:
            pass
        raise

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive, timeout=None):
    """Wait until the server at *base_url* is healthy.

    Args:
        base_url: Base URL of the engine (e.g. ``http://host:port``).
        api_key: Bearer token for the engine API (may be ``None``).
        is_process_alive: Callable returning ``False`` when the server
            process has exited.
        timeout: Maximum wall-clock seconds to wait.  ``None`` means no
            limit (backward-compatible default).  A ``TimeoutError`` is
            raised when the deadline is exceeded.
    """
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }
    # Per-request timeout for individual HTTP calls so that a single
    # ``requests.get`` does not block indefinitely on a black-holed host.
    _REQUEST_TIMEOUT = 10  # seconds (connect + read)

    deadline = (time.monotonic() + timeout) if timeout is not None else None

    def _check_deadline(phase: str):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for server to become healthy at {base_url} (phase={phase}, timeout={timeout}s)"
            )

    with requests.Session() as session:
        while True:
            _check_deadline("health_generate")
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers, timeout=_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            _check_deadline("flush_cache")
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
        register_sigterm_handler: bool = False,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self._evicted = threading.Event()
        self._is_weight_updating: bool = False
        self._isolated_from_megatron = False
        self._megatron_import_blocker = None
        self._isolate_from_megatron_if_needed()
        if register_sigterm_handler:
            self._register_sigterm_handler()

    def _isolate_from_megatron_if_needed(self) -> None:
        if self._isolated_from_megatron:
            return
        enabled = self.args.sglang_model_impl.lower() == "transformers" or _env_requests_megatron_isolation()
        _install_process_megatron_isolation(enabled, source="SGLangEngine process", block_imports=True)
        self._isolated_from_megatron = True

    def set_weight_updating(self, is_updating: bool) -> None:
        """Set whether a weight update is currently in progress.

        Called by RolloutManager before and after each weight sync so that the
        SIGTERM handler can wait for the update to finish before unregistering
        from the router.
        """
        self._is_weight_updating = is_updating

    def _register_sigterm_handler(self):
        """Register SIGTERM handler for platform-initiated pod eviction.

        When the platform needs to evict or replace a pod, it sends SIGTERM to the
        user process. We catch this signal and perform lightweight cleanup so the
        RolloutManager can detect the eviction and treat it as a scale-in event.

        If a weight update is in progress (``_is_weight_updating``), the handler
        blocks until the update finishes before unregistering from the router, to
        avoid disrupting NCCL communication groups. The k8s PreStop timeout
        (default 30s) serves as the hard deadline.
        """
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(signum, frame):
            actor_id = ""
            try:
                actor_id = ray.get_runtime_context().get_actor_id()
            except Exception:
                pass
            logger.warning(
                f"[SGLangEngine] Received SIGTERM (rank={self.rank}, actor_id={actor_id}), "
                f"marking as evicted for graceful scale-in"
            )
            self._evicted.set()

            # Wait for any in-progress weight update to finish before cleaning up,
            # so we don't disrupt NCCL communication groups during weight sync.
            # k8s PreStop hard deadline is typically 30s; leave a safety margin.
            weight_update_timeout = 20
            wait_start = time.time()
            while self._is_weight_updating:
                if time.time() - wait_start > weight_update_timeout:
                    logger.warning(
                        f"[SGLangEngine] Weight update did not finish within {weight_update_timeout}s, "
                        f"proceeding with eviction cleanup (rank={self.rank})"
                    )
                    break
                logger.warning(
                    f"[SGLangEngine] SIGTERM received but weight update in progress "
                    f"(rank={self.rank}, actor_id={actor_id}), waiting..."
                )
                time.sleep(1)

            # Best-effort: unregister from router so new requests are not routed here.
            # This is a quick HTTP call; if it fails the router will detect the engine
            # as unhealthy anyway.
            try:
                self.unregister_from_router()
            except Exception as e:
                logger.warning(f"[SGLangEngine] Failed to unregister from router during SIGTERM handling: {e}")

        signal.signal(signal.SIGTERM, _handle_sigterm)

    def is_evicted(self) -> bool:
        """Check whether this engine has received a SIGTERM eviction signal."""
        return self._evicted.is_set()

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        init_external_kwargs: Optional[dict] = None,
        skip_dcs_registration: bool = False,
        skip_router_registration: bool = False,
    ):
        """Initialize the SGLang engine.

        Args:
            skip_router_registration: If True, do not register to router during init.
                This is used during scale-out to ensure the engine receives weights
                before accepting requests. The caller must call register_to_router()
                after weight sync completes.
        """
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port
        self._skip_router_registration = skip_router_registration

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )
        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        # Start the engine first so the server is healthy before we create the
        # DCS client.  Creating the client before the server is ready can cause
        # Actor's weight-update path to reach a DCS endpoint that does not yet
        # exist, especially for scaled-out engines whose init() runs concurrently
        # with the training loop.
        self.checkpoint_engine_client = None
        if self.args.rollout_external or init_external_kwargs:
            if not init_external_kwargs:
                init_external_kwargs = {"external_engine_need_check_fields": external_engine_need_check_fields}
            self._init_external(server_args_dict, **init_external_kwargs)
        else:
            self._init_normal(server_args_dict)

        # Register to DCS coordinator only if not skipped (e.g., for scaled-out engines)
        # Scaled-out engines use direct weight sync from seed engine instead of DCS.
        # Done after engine startup so the coordinator can immediately reach the server.
        if not skip_dcs_registration:
            self.register_dcs()

    def register_dcs(self):
        if self.node_rank == 0 and self.args.fully_async:
            # Resolve effective num_gpus_per_engine for this engine
            effective_num_gpus = self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine
            self.checkpoint_engine_client = run(
                _create_checkpoint_client(
                    args=self.args,
                    coordinator_url=self.args.coordinator_url,
                    role="rollout",
                    ip=self.server_host,
                    port=self.server_port,
                    rank=self.rank,
                    metadata={"num_gpus_per_engine": effective_num_gpus},
                )
            )

    def _init_external(self, expect_server_args, external_engine_need_check_fields, timeout: float = 300):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info", timeout=10)
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert actual_value == expect_value, (
                    f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"
                )

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
            timeout=timeout,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

        if not self._skip_router_registration:
            self.register_to_router()

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        if getattr(self.args, "optimize_routing_replay", False):
            os.environ["RELAX_OPTIMIZE_ROUTING_REPLAY"] = "1"
        self.process = launch_server_process(_get_server_args_cls()(**server_args_dict))

        bootstrap_port = (
            server_args_dict.get("disaggregation_bootstrap_port") if self.worker_type == "prefill" else None
        )
        # Only register to router if skip_router_registration=False
        if not self._skip_router_registration:
            self.register_to_router(bootstrap_port=bootstrap_port)

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given
        payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """

        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """Update model weights from tensor data. The HTTP server will only
        post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        self.unregister_from_router()
        # external rollout has no process
        if hasattr(self, "process"):
            _kill_process_tree(self.process.pid)

    def __del__(self):
        """Safety net: kill SGLang child processes when the actor is garbage-
        collected.

        This prevents orphaned sglang::scheduler / sglang::detokenizer
        processes from lingering after training completes, in case shutdown()
        was not called explicitly (e.g. Ray actor GC without explicit cleanup).
        """
        process = getattr(self, "process", None)
        if process is not None and process.is_alive():
            try:
                _kill_process_tree(process.pid)
            except Exception:
                pass

    def get_url(self) -> str | None:
        """Return the HTTP URL of this engine, or None for non-node-0
        engines."""
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def get_pid_and_node_id(self) -> dict:
        """Return the PID and Ray node ID of this engine.

        Returns:
            dict with 'pid' (int) and 'node_id' (str) keys.
        """
        node_id = ""
        try:
            node_id = ray.get_runtime_context().get_node_id()
        except Exception:
            pass
        return {"pid": os.getpid(), "node_id": node_id}

    def register_to_router(self, bootstrap_port: int | None = None, strict: bool = True) -> bool:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return True

        worker_url = f"http://{self.server_host}:{self.server_port}"
        try:
            sglang_router = _get_sglang_router()
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                if self.worker_type != "regular":
                    msg = "pd disaggregation is not supported in old router or slime router."
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return False
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
                    timeout=30,
                )
            else:
                payload = {
                    "url": worker_url,
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill" and bootstrap_port is not None:
                    payload["bootstrap_port"] = bootstrap_port
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                    timeout=30,
                )
            response.raise_for_status()
            logger.info(f"Registered engine {worker_url} to router {self.router_ip}:{self.router_port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to register engine to router: {e}")
            return False

    def unregister_from_router(self) -> bool:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return True

        worker_url = f"http://{self.server_host}:{self.server_port}"
        try:
            sglang_router = _get_sglang_router()
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url={worker_url}",
                    timeout=30,
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                response = requests.delete(
                    f"http://{self.router_ip}:{self.router_port}/workers/{quote(worker_url, safe='')}",
                    timeout=30,
                )
            else:
                all_workers = requests.get(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    timeout=30,
                ).json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = requests.delete(
                            f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}",
                            timeout=30,
                        )
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during unregister.")
                    return False
            response.raise_for_status()
            logger.info(f"Unregistered engine {worker_url} from router {self.router_ip}:{self.router_port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to unregister engine from router: {e}")
            return False

    def get_rank(self) -> int:
        """Return the engine rank assigned during __init__."""
        return self.rank

    def get_weight_version(self) -> Optional[str]:
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """Available tags for multi-stage resume: weights, kv_cache."""
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def init_weights_send_group_for_remote_instance(
        self, master_address, ports, group_rank, world_size, group_name="weight_send_group", backend="nccl"
    ):
        return self._make_request(
            "init_weights_send_group_for_remote_instance",
            {
                "master_address": master_address,
                "ports": ports,
                "group_rank": group_rank,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def send_weights_to_remote_instance(self, master_address, ports, group_name="weight_send_group"):
        return self._make_request(
            "send_weights_to_remote_instance",
            {
                "master_address": master_address,
                "ports": ports,
                "group_name": group_name,
            },
        )

    def pause_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()

    def unregister_dcs(self):
        if self.node_rank == 0 and self.checkpoint_engine_client is not None:
            logger.info(f"Unregistering checkpoint engine client for engine {self.server_host}:{self.server_port}...")
            run(self.checkpoint_engine_client.unregister())


class GenRMEngine(SGLangEngine):
    """GenRM Engine for Generative Reward Model.

    Inherits from SGLangEngine and overrides initialization to use genrm-
    specific arguments (model path, GPU count, sampling parameters, etc.).
    """

    def init(self, dist_init_addr, port, nccl_port, host=None, disaggregation_bootstrap_port=None):
        """Initialize the genRM engine with genrm-specific arguments."""
        self.router_ip = ""
        self.router_port = 0
        self._skip_router_registration = True

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_genrm_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)


def _compute_genrm_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
):
    """Compute server arguments for genRM engine.

    This is similar to _compute_server_args but uses genrm-specific arguments:
    - model_path from genrm_model_path
    - tp_size from genrm_num_gpus_per_engine
    - max_total_tokens from genrm_max_context_len
    - max_decode_steps from genrm_max_response_len
    - sampling parameters from genrm_temperature, genrm_top_p, genrm_top_k
    """
    nnodes = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)

    kwargs = {
        "model_path": args.genrm_model_path,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": _sglang_memory_saver_enabled(args.offload_rollout),
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": args.genrm_num_gpus_per_engine,
        "dp_size": args.genrm_engine_config.get("dp_size", 1),
        "pp_size": args.genrm_engine_config.get("pp_size", 1),
        "ep_size": args.genrm_engine_config.get("ep_size", 1),
        # # context and response length
        # "max_total_tokens": args.genrm_engine_config['max_total_tokens'],
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": False,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # GenRM Only
        "enable_weights_cpu_backup": True,
    }

    # Allow per-genrm SGLang mem_fraction_static via --genrm-engine-config; this overrides
    # the global --sglang-mem-fraction-static below so rollout and genrm can share GPUs.
    if "mem_fraction_static" in args.genrm_engine_config:
        kwargs["mem_fraction_static"] = args.genrm_engine_config["mem_fraction_static"]

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert disaggregation_bootstrap_port is not None, (
            "disaggregation_bootstrap_port must be set for prefill worker"
        )
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(_get_server_args_cls()):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": _sglang_memory_saver_enabled(args.offload_rollout),
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        "enable_metrics": True,
    }

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert disaggregation_bootstrap_port is not None, (
            "disaggregation_bootstrap_port must be set for prefill worker"
        )
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(_get_server_args_cls()):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-engine-group overrides from --sglang-config YAML.
    # Applied after base args so they take highest priority.
    if sglang_overrides:
        for key, value in sglang_overrides.items():
            if key in kwargs:
                logger.info(f"sglang_overrides: overriding {key}={kwargs[key]} -> {value} (rank={rank})")
            kwargs[key] = value
            unused_keys.discard(key)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
