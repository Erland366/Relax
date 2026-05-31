# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import torch


def _get_data_parallel_size(args: Namespace) -> int | None:
    data_parallel_size = getattr(args, "data_parallel_size", None)
    if data_parallel_size is None:
        data_parallel_size = getattr(args, "world_size", None)
    return data_parallel_size


def should_disable_pinned_host_weight_backups(args: Namespace, role: str) -> bool:
    """Avoid pinned host memory for actor weight snapshots on single-rank
    ROCm."""

    return role == "actor" and _get_data_parallel_size(args) == 1 and torch.version.hip is not None
