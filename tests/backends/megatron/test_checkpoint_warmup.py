import hashlib
import importlib
import os
import sys
import types
from pathlib import Path


def _install_fake_megatron(monkeypatch):
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []

    core = types.ModuleType("megatron.core")
    core.__path__ = []
    core_utils = types.ModuleType("megatron.core.utils")
    core_utils.unwrap_model = lambda model: [model]

    training = types.ModuleType("megatron.training")
    training.__path__ = []
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    checkpointing.load_checkpoint = lambda *args, **kwargs: None
    checkpointing.save_checkpoint = lambda *args, **kwargs: None
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: None

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", core_utils)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars)


def _load_checkpoint_module(monkeypatch):
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.backends.megatron.checkpoint", None)
    return importlib.import_module("relax.backends.megatron.checkpoint")


def _marker_paths(checkpoint_path: Path) -> tuple[Path, Path]:
    abs_path = os.path.abspath(checkpoint_path)
    digest = hashlib.sha1(abs_path.encode()).hexdigest()[:16]
    marker_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    return (
        marker_dir / f"relax_hf_warmup_{digest}.lock",
        marker_dir / f"relax_hf_warmup_{digest}.done",
    )


def test_warm_hf_checkpoint_page_cache_single_process_dist_leads_with_nonzero_local_rank(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    import torch.distributed as torch_dist

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors").write_bytes(b"weights")
    _, done_path = _marker_paths(hf_dir)
    done_path.unlink(missing_ok=True)

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch_dist, "is_available", lambda: True)
    monkeypatch.setattr(torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_dist, "get_world_size", lambda: 1)

    checkpoint._warm_hf_checkpoint_page_cache(str(hf_dir))

    assert done_path.read_text() == os.path.abspath(hf_dir)


def test_warm_hf_checkpoint_page_cache_nonleader_waits_for_marker(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    import torch.distributed as torch_dist

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors").write_bytes(b"weights")
    _, done_path = _marker_paths(hf_dir)
    done_path.unlink(missing_ok=True)

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RELAX_HF_WARMUP_TIMEOUT_S", "0")
    monkeypatch.setattr(torch_dist, "is_available", lambda: True)
    monkeypatch.setattr(torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_dist, "get_world_size", lambda: 2)

    checkpoint._warm_hf_checkpoint_page_cache(str(hf_dir))

    assert not done_path.exists()
