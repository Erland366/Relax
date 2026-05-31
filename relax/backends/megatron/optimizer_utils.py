# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


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
