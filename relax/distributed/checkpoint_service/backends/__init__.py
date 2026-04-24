# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Communication backends package."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from relax.distributed.checkpoint_service.backends.base import CommBackend
    from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


__all__ = [
    "CommBackend",
    "DeviceDirectBackend",
]

_EXPORT_TO_MODULE = {
    "CommBackend": "relax.distributed.checkpoint_service.backends.base",
    "DeviceDirectBackend": "relax.distributed.checkpoint_service.backends.device_direct",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
