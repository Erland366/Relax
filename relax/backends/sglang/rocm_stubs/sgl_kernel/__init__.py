# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.abc
import importlib.machinery
import os
import sys


_UNAVAILABLE_MESSAGE = os.environ.get(
    "RELAX_SGL_KERNEL_STUB_MESSAGE",
    "sgl_kernel is unavailable on this ROCm runtime. Relax only stubs it so SGLang can import "
    "optional CUDA-only LoRA/MoE modules for dense transformers serving.",
)
_FINDER_MARKER = "_relax_rocm_sgl_kernel_stub_finder"


class _UnavailableSglKernelSymbol:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_UNAVAILABLE_MESSAGE)

    def __bool__(self):
        return False

    def __repr__(self):
        return "<unavailable ROCm sgl_kernel symbol>"


class _SglKernelSubmoduleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []
        module.__getattr__ = lambda name: _UnavailableSglKernelSymbol()


class _SglKernelSubmoduleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("sgl_kernel."):
            return importlib.machinery.ModuleSpec(fullname, _SglKernelSubmoduleLoader(), is_package=True)
        return None


def _install_submodule_finder() -> None:
    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return

    finder = _SglKernelSubmoduleFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


def __getattr__(name):
    return _UnavailableSglKernelSymbol()


_install_submodule_finder()
