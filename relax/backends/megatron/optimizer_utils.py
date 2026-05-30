# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections import defaultdict
from types import MethodType

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
    """Disable single-rank optimizer features that only help multi-rank
    paths."""

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
    """Use a safer CPU-offload configuration on single-rank ROCm actor
    training."""

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
    """Avoid Megatron fp32 main-param wrapping for the ROCm CPU-offload
    fallback."""

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
    """Avoid large pinned host weight snapshots on fragile single-rank ROCm
    actor init."""

    return role == "actor" and _get_data_parallel_size(args) == 1 and torch.version.hip is not None


def patch_megatron_cpu_offload_optimizer_for_rocm(args: Namespace, role: str, module=None) -> bool:
    """Patch Megatron CPU offload to use a safer torch AdamW wrapper on
    ROCm."""

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


def _iter_wrapped_megatron_optimizers(optimizer):
    wrapped_optimizers = getattr(optimizer, "chained_optimizers", None)
    if wrapped_optimizers is None:
        wrapped_optimizers = [optimizer]
    return wrapped_optimizers


def _is_hybrid_device_optimizer(optimizer) -> bool:
    return optimizer is not None and optimizer.__class__.__name__ == "HybridDeviceOptimizer"


def _iter_hybrid_device_sub_optimizers(optimizer):
    sub_optimizers = getattr(optimizer, "sub_optimizers", None)
    if sub_optimizers is not None:
        return list(sub_optimizers)

    sub_optimizers = list(getattr(optimizer, "cpu_optimizers", []))
    gpu_optimizer = getattr(optimizer, "gpu_optimizer", None)
    if gpu_optimizer is not None:
        sub_optimizers.append(gpu_optimizer)
    return sub_optimizers


def _init_hybrid_device_optimizer_adam_state(optimizer, config=None) -> None:
    if not _is_hybrid_device_optimizer(optimizer):
        raise TypeError(
            "ROCm HybridDeviceOptimizer checkpoint restore initializer received "
            f"{optimizer.__class__.__name__}"
        )

    sync_param_groups = getattr(optimizer, "_sync_hdo_param_groups_to_sub_optimizers", None)
    if sync_param_groups is None:
        raise RuntimeError("HybridDeviceOptimizer is missing _sync_hdo_param_groups_to_sub_optimizers")
    sync_param_groups()

    initialized_states = 0
    for sub_optimizer in _iter_hybrid_device_sub_optimizers(optimizer):
        for group in sub_optimizer.param_groups:
            for param in group["params"]:
                state = sub_optimizer.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param.data)
                    initialized_states += 1
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

    sync_state = getattr(optimizer, "_sync_sub_optimizers_state_to_hdo", None)
    if sync_state is None:
        raise RuntimeError("HybridDeviceOptimizer is missing _sync_sub_optimizers_state_to_hdo")
    sync_state()

    if initialized_states:
        logger.info(
            "Initialized %s HybridDeviceOptimizer Adam states for ROCm checkpoint restore",
            initialized_states,
        )


def _build_inner_param_groups(optimizer):
    inner_param_groups = []
    param_to_inner_param = getattr(optimizer, "param_to_inner_param", {})
    for group in optimizer.param_groups:
        inner_group = group.copy()
        inner_group["params"] = [param_to_inner_param.get(param, param) for param in group["params"]]
        inner_param_groups.append(inner_group)
    return inner_param_groups


def _load_hybrid_device_optimizer_state_dict_on_inner_params(optimizer, state_dict: dict) -> None:
    original_param_groups = [group.copy() for group in optimizer.param_groups]
    inner_param_groups = _build_inner_param_groups(optimizer)
    pre_hooks = optimizer._optimizer_load_state_dict_pre_hooks.copy()
    post_hooks = optimizer._optimizer_load_state_dict_post_hooks.copy()

    optimizer._optimizer_load_state_dict_pre_hooks.clear()
    optimizer._optimizer_load_state_dict_post_hooks.clear()
    optimizer.param_groups = inner_param_groups
    try:
        torch.optim.Optimizer.load_state_dict(optimizer, state_dict)
        loaded_inner_state = optimizer.state
        loaded_inner_param_groups = [group.copy() for group in optimizer.param_groups]
    finally:
        optimizer._optimizer_load_state_dict_pre_hooks.update(pre_hooks)
        optimizer._optimizer_load_state_dict_post_hooks.update(post_hooks)

    restored_param_groups = []
    assert len(original_param_groups) == len(loaded_inner_param_groups)
    for original_group, loaded_group in zip(original_param_groups, loaded_inner_param_groups):
        restored_group = loaded_group.copy()
        restored_group["params"] = original_group["params"]
        restored_param_groups.append(restored_group)
    optimizer.param_groups = restored_param_groups
    optimizer._sync_hdo_param_groups_to_sub_optimizers()

    for sub_optimizer in _iter_hybrid_device_sub_optimizers(optimizer):
        sub_state = defaultdict(dict)
        for group in sub_optimizer.param_groups:
            for param in group["params"]:
                if param in loaded_inner_state:
                    sub_state[param] = loaded_inner_state[param]
        sub_optimizer.state = sub_state
    optimizer._sync_sub_optimizers_state_to_hdo()


def _load_fp32_optimizer_with_hybrid_device_optimizer_state(self, state_dict: dict) -> None:
    if "common_step" in state_dict["state"]:
        common_step = state_dict["state"].pop("common_step")
        self._restore_common_per_param_step(state_dict, common_step)

    state_dict["param_groups"] = self._filter_and_reorder_param_groups(
        self.optimizer.param_groups,
        state_dict["param_groups"],
    )
    _load_hybrid_device_optimizer_state_dict_on_inner_params(self.optimizer, state_dict)


def install_hybrid_device_optimizer_init_state_fn(optimizer, args: Namespace, role: str) -> bool:
    """Install the missing Megatron load initializer for ROCm HDO restore."""

    if role != "actor" or _get_data_parallel_size(args) != 1:
        return False
    if torch.version.hip is None or not getattr(args, "optimizer_cpu_offload", False):
        return False

    changed = False
    for wrapped_optimizer in _iter_wrapped_megatron_optimizers(optimizer):
        inner_optimizer = getattr(wrapped_optimizer, "optimizer", None)
        if not _is_hybrid_device_optimizer(inner_optimizer):
            continue

        config = getattr(wrapped_optimizer, "config", None)
        optimizer_kind = getattr(config, "optimizer", getattr(args, "optimizer", None))
        if optimizer_kind != "adam":
            raise RuntimeError(
                "ROCm HybridDeviceOptimizer checkpoint restore only supports Adam state "
                f"initialization, got optimizer={optimizer_kind!r}"
            )

        if getattr(wrapped_optimizer, "init_state_fn", None) is None:
            wrapped_optimizer.init_state_fn = _init_hybrid_device_optimizer_adam_state
            changed = True
        if (
            wrapped_optimizer.__class__.__name__ == "FP32Optimizer"
            and not getattr(wrapped_optimizer, "_relax_rocm_hdo_load_state_dict", False)
        ):
            wrapped_optimizer.load_state_dict = MethodType(
                _load_fp32_optimizer_with_hybrid_device_optimizer_state,
                wrapped_optimizer,
            )
            wrapped_optimizer._relax_rocm_hdo_load_state_dict = True
            changed = True

    if changed:
        logger.info(
            "Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore"
        )
    return changed


def refresh_hybrid_device_optimizer_param_groups(optimizer) -> bool:
    """Refresh HybridDeviceOptimizer mappings after main-param wrapping."""

    refreshed = False
    for wrapped_optimizer in _iter_wrapped_megatron_optimizers(optimizer):
        if wrapped_optimizer.__class__.__name__ == "FP32Optimizer":
            continue
        inner_optimizer = getattr(wrapped_optimizer, "optimizer", None)
        if not _is_hybrid_device_optimizer(inner_optimizer):
            continue
        inner_optimizer._init_sub_optimizers()
        inner_optimizer._sync_hdo_param_groups_to_sub_optimizers()
        refreshed = True

    if refreshed:
        logger.info("Refreshed HybridDeviceOptimizer param-group mappings after Megatron main-param wrapping")
    return refreshed
