# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from argparse import Namespace

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_ROCM_TORCH_OPTIMIZER_PATCHED = False
_ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED = False
_ORIGINAL_TORCH_ADAM = torch.optim.Adam
_ORIGINAL_TORCH_ADAMW = torch.optim.AdamW


def _force_single_tensor_adam_kwargs(kwargs: dict) -> dict:
    patched_kwargs = dict(kwargs)
    patched_kwargs["foreach"] = False
    patched_kwargs["fused"] = False
    return patched_kwargs


class _RelaxRocmSingleTensorAdam(_ORIGINAL_TORCH_ADAM):
    """Adam wrapper that avoids ROCm foreach/fused kernels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **_force_single_tensor_adam_kwargs(kwargs))


class _RelaxRocmSingleTensorAdamW(_ORIGINAL_TORCH_ADAMW):
    """AdamW wrapper that avoids ROCm foreach/fused kernels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **_force_single_tensor_adam_kwargs(kwargs))


def _get_data_parallel_size(args: Namespace) -> int | None:
    data_parallel_size = getattr(args, "data_parallel_size", None)
    if data_parallel_size is None:
        data_parallel_size = getattr(args, "world_size", None)
    return data_parallel_size


def normalize_single_rank_optimizer_args(args: Namespace, role: str) -> bool:
    """Disable optimizer features that only make sense with multiple DP
    ranks."""

    if _get_data_parallel_size(args) != 1:
        return False

    disabled_flags = []
    for field_name in (
        "use_distributed_optimizer",
        "use_precision_aware_optimizer",
        "overlap_grad_reduce",
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


def reject_rocm_actor_cpu_offload_optimizer(args: Namespace, role: str) -> None:
    """Fail loudly for the ROCm CPU-offload optimizer path.

    The MI210 bring-up repeatedly reached real training and then crashed inside
    Megatron's HybridDeviceOptimizer grad-sync path. Treating CPU optimizer
    offload as a fallback hides that failure. The supported ROCm path is the
    GPU optimizer path.
    """

    if torch.version.hip is None or not getattr(args, "optimizer_cpu_offload", False):
        return

    raise RuntimeError(
        f"ROCm {role} optimizer CPU offload is disabled because Megatron "
        "HybridDeviceOptimizer crashes during grad sync on this path. Remove "
        "--optimizer-cpu-offload and use the GPU optimizer path."
    )


def patch_rocm_torch_optimizer_for_megatron(args: Namespace, role: str) -> bool:
    """Force Megatron's torch optimizer path for the supported ROCm DP1 setup.

    ROCm TransformerEngine can import successfully while its ``FusedAdam``
    optimizer still fails at runtime on MI210. Megatron decides whether to use
    TE/Apex or torch optimizers through module globals initialized at import
    time, so Relax patches those globals before constructing the optimizer.
    """

    global _ROCM_TORCH_OPTIMIZER_PATCHED
    if _ROCM_TORCH_OPTIMIZER_PATCHED or torch.version.hip is None:
        return False
    if os.environ.get("RELAX_ROCM_TORCH_OPTIMIZER_PATCH", "1") != "1":
        return False
    if _get_data_parallel_size(args) != 1:
        return False
    if getattr(args, "optimizer_cpu_offload", False):
        return False
    if getattr(args, "use_precision_aware_optimizer", False) or getattr(args, "fp16", False):
        raise RuntimeError(
            f"ROCm {role} torch optimizer fallback is incompatible with precision-aware "
            "optimizer mode. Disable --use-precision-aware-optimizer and fp16 for this path."
        )

    import megatron.core.optimizer as megatron_optimizer

    if not hasattr(megatron_optimizer, "_relax_rocm_original_optimizer_globals"):
        megatron_optimizer._relax_rocm_original_optimizer_globals = {
            "Adam": megatron_optimizer.Adam,
            "SGD": megatron_optimizer.SGD,
            "USING_PYTORCH_OPTIMIZER": megatron_optimizer.USING_PYTORCH_OPTIMIZER,
            "torch_Adam": torch.optim.Adam,
            "torch_AdamW": torch.optim.AdamW,
        }

    torch.optim.Adam = _RelaxRocmSingleTensorAdam
    torch.optim.AdamW = _RelaxRocmSingleTensorAdamW
    megatron_optimizer.Adam = _RelaxRocmSingleTensorAdamW
    megatron_optimizer.SGD = torch.optim.SGD
    megatron_optimizer.USING_PYTORCH_OPTIMIZER = True
    _ROCM_TORCH_OPTIMIZER_PATCHED = True
    logger.info(
        "Patched Megatron optimizer globals to use ROCm single-tensor torch Adam optimizers for %s DP1 path",
        role,
    )
    return True


def patch_rocm_optimizer_cuda_graph_health_check(args: Namespace, role: str) -> bool:
    """Skip PyTorch optimizer CUDA-graph checks on the non-graph ROCm path."""

    global _ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED
    if _ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED or torch.version.hip is None:
        return False
    if os.environ.get("RELAX_ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCH", "1") != "1":
        return False
    if getattr(args, "enable_cuda_graph", False) or getattr(args, "external_cuda_graph", False):
        raise RuntimeError(
            f"ROCm {role} optimizer CUDA-graph health-check patch requires training CUDA graphs to be disabled."
        )
    if getattr(args, "cuda_graph_impl", "none") not in (None, "none"):
        raise RuntimeError(
            f"ROCm {role} optimizer CUDA-graph health-check patch requires --cuda-graph-impl none."
        )

    from torch.optim.optimizer import Optimizer

    if getattr(Optimizer._cuda_graph_capture_health_check, "_relax_rocm_noop_patch", False):
        _ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED = True
        return False

    original_health_check = Optimizer._cuda_graph_capture_health_check

    def _relax_rocm_cuda_graph_capture_health_check(self) -> None:
        del self

    _relax_rocm_cuda_graph_capture_health_check._relax_rocm_noop_patch = True
    _relax_rocm_cuda_graph_capture_health_check._relax_rocm_original_health_check = original_health_check
    Optimizer._cuda_graph_capture_health_check = _relax_rocm_cuda_graph_capture_health_check
    _ROCM_OPTIMIZER_CUDA_GRAPH_HEALTH_CHECK_PATCHED = True
    logger.info("Patched PyTorch optimizer CUDA-graph health check for ROCm %s non-graph path", role)
    return True
