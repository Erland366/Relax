# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron import model_provider
from relax.backends.megatron import optimizer_utils
from relax.backends.megatron.model_provider import (
    patch_rocm_transformer_block_final_norm,
    patch_torch_norm_sequence_parallel_for_rocm,
)
from relax.backends.megatron.optimizer_utils import (
    normalize_single_rank_optimizer_args,
    patch_rocm_optimizer_cuda_graph_health_check,
    patch_rocm_torch_optimizer_for_megatron,
    reject_rocm_actor_cpu_offload_optimizer,
)
from relax.backends.megatron.weight_backup_utils import should_disable_pinned_host_weight_backups


def test_normalize_single_rank_optimizer_args_disables_distributed_optimizer_flags():
    args = Namespace(
        data_parallel_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.use_precision_aware_optimizer is False
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_uses_world_size_fallback():
    args = Namespace(
        world_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        overlap_grad_reduce=True,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="critic")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.overlap_grad_reduce is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_keeps_multi_rank_flags():
    args = Namespace(
        data_parallel_size=2,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is False
    assert args.use_distributed_optimizer is True
    assert args.use_precision_aware_optimizer is True
    assert args.overlap_grad_reduce is True
    assert args.overlap_param_gather is True
    assert args.overlap_param_gather_with_optimizer_step is True
    assert args.overlap_cpu_optimizer_d2h_h2d is True


def test_reject_rocm_actor_cpu_offload_optimizer_fails_loudly(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True)

    with pytest.raises(RuntimeError, match="ROCm actor optimizer CPU offload is disabled"):
        reject_rocm_actor_cpu_offload_optimizer(args, role="actor")


def test_reject_rocm_actor_cpu_offload_optimizer_rejects_multi_rank(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=2, optimizer_cpu_offload=True)

    with pytest.raises(RuntimeError, match="ROCm actor optimizer CPU offload is disabled"):
        reject_rocm_actor_cpu_offload_optimizer(args, role="actor")


def test_reject_rocm_actor_cpu_offload_optimizer_rejects_non_actor_role(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True)

    with pytest.raises(RuntimeError, match="ROCm critic optimizer CPU offload is disabled"):
        reject_rocm_actor_cpu_offload_optimizer(args, role="critic")


def test_reject_rocm_actor_cpu_offload_optimizer_allows_gpu_optimizer(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=False)

    reject_rocm_actor_cpu_offload_optimizer(args, role="actor")


def test_reject_rocm_actor_cpu_offload_optimizer_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", None)
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True)

    reject_rocm_actor_cpu_offload_optimizer(args, role="actor")


def test_patch_rocm_torch_optimizer_for_megatron_forces_torch_optimizer(monkeypatch):
    import megatron.core.optimizer as megatron_optimizer

    original_adam = megatron_optimizer.Adam
    original_sgd = megatron_optimizer.SGD
    original_using_torch = megatron_optimizer.USING_PYTORCH_OPTIMIZER
    original_torch_adam = torch.optim.Adam
    original_torch_adamw = torch.optim.AdamW
    had_original_globals = hasattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals")
    original_globals = getattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals", None)

    class FakeFusedAdam:
        pass

    class FakeFusedSGD:
        pass

    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    monkeypatch.setattr(optimizer_utils, "_ROCM_TORCH_OPTIMIZER_PATCHED", False)
    monkeypatch.delenv("RELAX_ROCM_TORCH_OPTIMIZER_PATCH", raising=False)
    megatron_optimizer.Adam = FakeFusedAdam
    megatron_optimizer.SGD = FakeFusedSGD
    megatron_optimizer.USING_PYTORCH_OPTIMIZER = False
    if hasattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals"):
        delattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals")

    try:
        args = Namespace(
            data_parallel_size=1,
            fp16=False,
            optimizer_cpu_offload=False,
            use_precision_aware_optimizer=False,
        )

        changed = patch_rocm_torch_optimizer_for_megatron(args, role="actor")

        assert changed is True
        assert megatron_optimizer.Adam is optimizer_utils._RelaxRocmSingleTensorAdamW
        assert torch.optim.Adam is optimizer_utils._RelaxRocmSingleTensorAdam
        assert torch.optim.AdamW is optimizer_utils._RelaxRocmSingleTensorAdamW
        assert megatron_optimizer.SGD is torch.optim.SGD
        assert megatron_optimizer.USING_PYTORCH_OPTIMIZER is True
        assert megatron_optimizer._relax_rocm_original_optimizer_globals == {
            "Adam": FakeFusedAdam,
            "SGD": FakeFusedSGD,
            "USING_PYTORCH_OPTIMIZER": False,
            "torch_Adam": original_torch_adam,
            "torch_AdamW": original_torch_adamw,
        }

        param = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([param], lr=1e-3, foreach=True, fused=True)
        assert optimizer.defaults["foreach"] is False
        assert optimizer.defaults["fused"] is False
    finally:
        megatron_optimizer.Adam = original_adam
        megatron_optimizer.SGD = original_sgd
        megatron_optimizer.USING_PYTORCH_OPTIMIZER = original_using_torch
        torch.optim.Adam = original_torch_adam
        torch.optim.AdamW = original_torch_adamw
        if had_original_globals:
            megatron_optimizer._relax_rocm_original_optimizer_globals = original_globals
        elif hasattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals"):
            delattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals")
        optimizer_utils._ROCM_TORCH_OPTIMIZER_PATCHED = False


def test_patch_rocm_torch_optimizer_for_megatron_skips_when_disabled(monkeypatch):
    import megatron.core.optimizer as megatron_optimizer

    original_using_torch = megatron_optimizer.USING_PYTORCH_OPTIMIZER
    original_torch_adam = torch.optim.Adam
    original_torch_adamw = torch.optim.AdamW
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    monkeypatch.setattr(optimizer_utils, "_ROCM_TORCH_OPTIMIZER_PATCHED", False)
    monkeypatch.setenv("RELAX_ROCM_TORCH_OPTIMIZER_PATCH", "0")
    megatron_optimizer.USING_PYTORCH_OPTIMIZER = False

    try:
        args = Namespace(
            data_parallel_size=1,
            fp16=False,
            optimizer_cpu_offload=False,
            use_precision_aware_optimizer=False,
        )

        changed = patch_rocm_torch_optimizer_for_megatron(args, role="actor")

        assert changed is False
        assert megatron_optimizer.USING_PYTORCH_OPTIMIZER is False
        assert torch.optim.Adam is original_torch_adam
        assert torch.optim.AdamW is original_torch_adamw
    finally:
        megatron_optimizer.USING_PYTORCH_OPTIMIZER = original_using_torch
        torch.optim.Adam = original_torch_adam
        torch.optim.AdamW = original_torch_adamw
        optimizer_utils._ROCM_TORCH_OPTIMIZER_PATCHED = False


def test_patch_rocm_torch_optimizer_for_megatron_rejects_precision_aware_mode(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    monkeypatch.setattr(optimizer_utils, "_ROCM_TORCH_OPTIMIZER_PATCHED", False)
    args = Namespace(
        data_parallel_size=1,
        fp16=False,
        optimizer_cpu_offload=False,
        use_precision_aware_optimizer=True,
    )

    with pytest.raises(RuntimeError, match="torch optimizer fallback is incompatible"):
        patch_rocm_torch_optimizer_for_megatron(args, role="actor")


def test_patch_rocm_optimizer_cuda_graph_health_check_noops_on_non_graph_rocm(monkeypatch):
    from torch.optim.optimizer import Optimizer

    original_health_check = Optimizer._cuda_graph_capture_health_check
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    monkeypatch.setattr(optimizer_utils, "_ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED", False)
    monkeypatch.delenv("RELAX_ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCH", raising=False)

    try:
        args = Namespace(enable_cuda_graph=False, external_cuda_graph=False, cuda_graph_impl="none")

        changed = patch_rocm_optimizer_cuda_graph_health_check(args, role="actor")

        assert changed is True
        assert Optimizer._cuda_graph_capture_health_check is not original_health_check
        assert Optimizer._cuda_graph_capture_health_check._relax_rocm_noop_patch is True
        assert (
            Optimizer._cuda_graph_capture_health_check._relax_rocm_original_health_check
            is original_health_check
        )
    finally:
        Optimizer._cuda_graph_capture_health_check = original_health_check
        optimizer_utils._ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED = False


def test_patch_rocm_optimizer_cuda_graph_health_check_rejects_training_cuda_graph(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    monkeypatch.setattr(optimizer_utils, "_ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED", False)
    args = Namespace(enable_cuda_graph=True, external_cuda_graph=False, cuda_graph_impl="none")

    with pytest.raises(RuntimeError, match="requires training CUDA graphs to be disabled"):
        patch_rocm_optimizer_cuda_graph_health_check(args, role="actor")


def test_should_disable_pinned_host_weight_backups_for_single_rank_rocm_actor(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.weight_backup_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, offload_train=False)

    assert should_disable_pinned_host_weight_backups(args, role="actor") is True


def test_should_disable_pinned_host_weight_backups_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.weight_backup_utils.torch.version.hip", None)
    args = Namespace(data_parallel_size=1, offload_train=True)

    assert should_disable_pinned_host_weight_backups(args, role="actor") is False


def test_patch_torch_norm_sequence_parallel_for_rocm_marks_norm_params(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.model_provider.torch.version.hip", "6.3.0")
    patch_torch_norm_sequence_parallel_for_rocm()

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    config = SimpleNamespace(
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        sequence_parallel=True,
        memory_efficient_layer_norm=False,
        normalization="RMSNorm",
    )
    norm = WrappedTorchNorm(config=config, hidden_size=8, eps=1e-6)

    assert norm.sequence_parallel is True
    assert norm.weight.sequence_parallel is True
    x = torch.randn(4, 8, requires_grad=True)
    norm(x).sum().backward()


def test_patch_rocm_transformer_block_final_norm_uses_wrapped_torch_norm(monkeypatch):
    from megatron.core.transformer import transformer_block
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    class FakeTENorm:
        pass

    original_layer_norm_impl = transformer_block.LayerNormImpl
    original_saved_impl = getattr(transformer_block, "_relax_rocm_original_layer_norm_impl", None)
    had_saved_impl = hasattr(transformer_block, "_relax_rocm_original_layer_norm_impl")

    monkeypatch.setattr("relax.backends.megatron.model_provider.torch.version.hip", "6.3.0")
    monkeypatch.setattr(model_provider, "_ROCM_TRANSFORMER_BLOCK_FINAL_NORM_PATCHED", False)
    transformer_block.LayerNormImpl = FakeTENorm
    try:
        patch_rocm_transformer_block_final_norm()

        assert transformer_block.LayerNormImpl is WrappedTorchNorm
        assert transformer_block._relax_rocm_original_layer_norm_impl is FakeTENorm
    finally:
        transformer_block.LayerNormImpl = original_layer_norm_impl
        if had_saved_impl:
            transformer_block._relax_rocm_original_layer_norm_impl = original_saved_impl
        elif hasattr(transformer_block, "_relax_rocm_original_layer_norm_impl"):
            delattr(transformer_block, "_relax_rocm_original_layer_norm_impl")
        model_provider._ROCM_TRANSFORMER_BLOCK_FINAL_NORM_PATCHED = False
