# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class TorchCPUAdamW(torch.optim.AdamW):
    """Torch AdamW wrapper compatible with Megatron CPU-offload kwargs."""

    def __init__(self, params, **kwargs):
        kwargs.pop("bias_correction", None)
        kwargs.pop("fused", None)
        # Force the simplest CPU AdamW path on ROCm. The default foreach path
        # can still route through a multi-tensor implementation that is harder
        # to reason about when the actor already wedges inside optimizer.step().
        kwargs["foreach"] = False
        super().__init__(params, **kwargs)


def _get_data_parallel_size(args: Namespace) -> int | None:
    data_parallel_size = getattr(args, "data_parallel_size", None)
    if data_parallel_size is None:
        data_parallel_size = getattr(args, "world_size", None)
    return data_parallel_size


def normalize_single_rank_optimizer_args(args: Namespace, role: str) -> bool:
    """Disable single-rank optimizer features that only help multi-rank paths."""

    if _get_data_parallel_size(args) != 1:
        return False

    disabled_flags = []
    for field_name in (
        "use_distributed_optimizer",
        "use_precision_aware_optimizer",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
        "overlap_cpu_optimizer_d2h_h2d",
    ):
        if getattr(args, field_name, False):
            setattr(args, field_name, False)
            disabled_flags.append(field_name)

    if disabled_flags:
        logger.info(
            "Disabled distributed optimizer features for single-rank %s path: %s",
            role,
            ", ".join(disabled_flags),
        )
        return True
    return False


def apply_single_rank_rocm_cpu_offload_safety_args(args: Namespace, role: str) -> bool:
    """Use a safer CPU-offload configuration on single-rank ROCm actor training."""

    if role != "actor" or _get_data_parallel_size(args) != 1:
        return False
    if torch.version.hip is None or not getattr(args, "optimizer_cpu_offload", False):
        return False

    changed_fields = []
    for field_name, safe_value in (
        ("use_torch_optimizer_for_cpu_offload", True),
        ("pin_cpu_grads", False),
        ("pin_cpu_params", False),
    ):
        if getattr(args, field_name, None) != safe_value:
            setattr(args, field_name, safe_value)
            changed_fields.append(field_name)

    if changed_fields:
        logger.info(
            "Applied single-rank ROCm CPU-offload safety args for %s path: %s",
            role,
            ", ".join(changed_fields),
        )
        return True
    return False


def apply_single_rank_rocm_cpu_offload_optimizer_config(args: Namespace, role: str, kwargs: dict) -> bool:
    """Avoid Megatron fp32 main-param wrapping for the ROCm CPU-offload fallback."""

    if role != "actor" or _get_data_parallel_size(args) != 1:
        return False
    if torch.version.hip is None or not getattr(args, "optimizer_cpu_offload", False):
        return False

    changed_fields = []
    for field_name, safe_value in (
        ("bf16", False),
        ("fp16", False),
        ("use_precision_aware_optimizer", False),
    ):
        if kwargs.get(field_name, None) != safe_value:
            kwargs[field_name] = safe_value
            changed_fields.append(field_name)

    if changed_fields:
        logger.info(
            "Applied single-rank ROCm CPU-offload optimizer config for %s path: %s",
            role,
            ", ".join(changed_fields),
        )
        return True
    return False


def should_disable_pinned_host_weight_backups(args: Namespace, role: str) -> bool:
    """Avoid large pinned host weight snapshots on fragile single-rank ROCm actor init."""

    return (
        role == "actor"
        and _get_data_parallel_size(args) == 1
        and torch.version.hip is not None
    )


def patch_megatron_cpu_offload_optimizer_for_rocm(args: Namespace, role: str, module=None) -> bool:
    """Patch Megatron CPU offload to use a safer torch AdamW wrapper on ROCm."""

    if role != "actor" or _get_data_parallel_size(args) != 1:
        return False
    if torch.version.hip is None or not getattr(args, "optimizer_cpu_offload", False):
        return False
    if not getattr(args, "use_torch_optimizer_for_cpu_offload", False):
        return False

    megatron_optimizer_module = module
    if megatron_optimizer_module is None:
        import megatron.core.optimizer as megatron_optimizer_module

    if getattr(megatron_optimizer_module, "CPUAdam", None) is TorchCPUAdamW:
        return False

    megatron_optimizer_module.CPUAdam = TorchCPUAdamW
    logger.info("Patched Megatron CPUAdam with torch AdamW wrapper for single-rank ROCm CPU offload")
    return True


def refresh_hybrid_device_optimizer_param_groups(optimizer) -> bool:
    """Refresh HybridDeviceOptimizer mappings after main-param wrapping."""

    wrapped_optimizers = getattr(optimizer, "chained_optimizers", None)
    if wrapped_optimizers is None:
        wrapped_optimizers = [optimizer]

    refreshed = False
    for wrapped_optimizer in wrapped_optimizers:
        if wrapped_optimizer.__class__.__name__ == "FP32Optimizer":
            continue
        inner_optimizer = getattr(wrapped_optimizer, "optimizer", None)
        if inner_optimizer is None:
            continue
        if inner_optimizer.__class__.__name__ != "HybridDeviceOptimizer":
            continue
        inner_optimizer._init_sub_optimizers()
        inner_optimizer._sync_hdo_param_groups_to_sub_optimizers()
        refreshed = True

    if refreshed:
        logger.info("Refreshed HybridDeviceOptimizer param-group mappings after Megatron main-param wrapping")
    return refreshed
