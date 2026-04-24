# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
Distributed Checkpoint Service (DCS) - A high-performance checkpoint engine.

This package provides a distributed checkpoint engine with:
- Control plane / Data plane separation
- Dynamic role-aware networking
- Dual communication backends (DeviceDirect + CpuOffload)
- Elastic scaling and resharding support
- Production-grade fault tolerance
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from relax.distributed.checkpoint_service.backends import CommBackend, DeviceDirectBackend
    from relax.distributed.checkpoint_service.client import CheckpointEngineClient
    from relax.distributed.checkpoint_service.config import BackendType, DCSConfig, RoleInfo
    from relax.distributed.checkpoint_service.coordinator import DCSCoordinator
    from relax.distributed.checkpoint_service.metrics import MetricsCollector


__version__ = "0.1.0"

__all__ = [
    "DCSConfig",
    "RoleInfo",
    "BackendType",
    "CommBackend",
    "DeviceDirectBackend",
    "DCSCoordinator",
    "CheckpointEngineClient",
    "MetricsCollector",
]

_EXPORT_TO_MODULE = {
    "DCSConfig": "relax.distributed.checkpoint_service.config",
    "RoleInfo": "relax.distributed.checkpoint_service.config",
    "BackendType": "relax.distributed.checkpoint_service.config",
    "CommBackend": "relax.distributed.checkpoint_service.backends",
    "DeviceDirectBackend": "relax.distributed.checkpoint_service.backends",
    "DCSCoordinator": "relax.distributed.checkpoint_service.coordinator",
    "CheckpointEngineClient": "relax.distributed.checkpoint_service.client",
    "MetricsCollector": "relax.distributed.checkpoint_service.metrics",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
