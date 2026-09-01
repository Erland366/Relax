# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate CPU topology before launching CPU-vision capacity experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple


MIN_SCHEDULER_AFFINITY_CPUS = 16
MIN_DISTINCT_PHYSICAL_CORES = 8
MAX_RESERVED_VIT_CPUS = 8
CPU_TOPOLOGY_ROOT = Path("/sys/devices/system/cpu")

PhysicalCoreKey = Tuple[int, int]


def validate_cpu_capacity(
    *,
    affinity_cpu_ids: Iterable[int],
    physical_core_keys: Iterable[PhysicalCoreKey],
    reserved_vit_cpus: int,
    min_affinity_cpus: int = MIN_SCHEDULER_AFFINITY_CPUS,
    min_physical_cores: int = MIN_DISTINCT_PHYSICAL_CORES,
) -> None:
    """Validate the scheduler allocation and CPU reservation for a capacity run."""
    affinity_cpu_count = len(set(affinity_cpu_ids))
    if affinity_cpu_count < min_affinity_cpus:
        raise ValueError(
            f"CPU-vision capacity mode requires at least {min_affinity_cpus} "
            f"scheduler-affinity CPUs; found {affinity_cpu_count}"
        )

    physical_core_count = len(set(physical_core_keys))
    if physical_core_count < min_physical_cores:
        raise ValueError(
            f"CPU-vision capacity mode requires at least {min_physical_cores} "
            f"distinct physical cores; found {physical_core_count}"
        )

    if reserved_vit_cpus > MAX_RESERVED_VIT_CPUS:
        raise ValueError(
            f"CPU-vision capacity mode permits at most {MAX_RESERVED_VIT_CPUS} reserved ViT CPUs; "
            f"got {reserved_vit_cpus}"
        )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _physical_core_key(cpu_id: int) -> PhysicalCoreKey:
    topology_dir = CPU_TOPOLOGY_ROOT / f"cpu{cpu_id}" / "topology"
    package_id = int((topology_dir / "physical_package_id").read_text())
    core_id = int((topology_dir / "core_id").read_text())
    return package_id, core_id


def _physical_core_keys(affinity_cpu_ids: Iterable[int]) -> list[PhysicalCoreKey]:
    return [_physical_core_key(cpu_id) for cpu_id in affinity_cpu_ids]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-encoder-num-replicas", required=True, type=_positive_integer)
    parser.add_argument("--vision-encoder-num-cpus", required=True, type=_positive_integer)
    parser.add_argument("--min-affinity-cpus", type=_positive_integer, default=MIN_SCHEDULER_AFFINITY_CPUS)
    parser.add_argument("--min-physical-cores", type=_positive_integer, default=MIN_DISTINCT_PHYSICAL_CORES)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    affinity_cpu_ids = sorted(os.sched_getaffinity(0))
    reserved_vit_cpus = args.vision_encoder_num_replicas * args.vision_encoder_num_cpus
    try:
        physical_core_keys = _physical_core_keys(affinity_cpu_ids)
        validate_cpu_capacity(
            affinity_cpu_ids=affinity_cpu_ids,
            physical_core_keys=physical_core_keys,
            reserved_vit_cpus=reserved_vit_cpus,
            min_affinity_cpus=args.min_affinity_cpus,
            min_physical_cores=args.min_physical_cores,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        "CPU-vision capacity preflight passed: "
        f"affinity_cpus={len(affinity_cpu_ids)}, "
        f"physical_cores={len(set(physical_core_keys))}, "
        f"reserved_vit_cpus={reserved_vit_cpus}"
    )


if __name__ == "__main__":
    main()
