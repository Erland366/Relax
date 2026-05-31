# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.model_provider import patch_torch_norm_sequence_parallel_for_rocm
from relax.backends.megatron.optimizer_utils import (
    normalize_single_rank_optimizer_args,
    reject_rocm_actor_cpu_offload_optimizer,
)
from relax.backends.megatron.weight_backup_utils import should_disable_pinned_host_weight_backups


def test_normalize_single_rank_optimizer_args_disables_distributed_optimizer_flags():
    args = Namespace(
        data_parallel_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.use_precision_aware_optimizer is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_uses_world_size_fallback():
    args = Namespace(
        world_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="critic")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_keeps_multi_rank_flags():
    args = Namespace(
        data_parallel_size=2,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is False
    assert args.use_distributed_optimizer is True
    assert args.use_precision_aware_optimizer is True
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
