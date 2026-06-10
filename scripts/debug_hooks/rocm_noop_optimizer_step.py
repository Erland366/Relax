# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Diagnostic hook for isolating ROCm optimizer-step failures.

Pass this file through ``--custom-megatron-before-train-step-hook-path`` to
prove which part of the first optimizer update corrupts the next backward pass.
This is not a training mode; it is an explicit debugging probe.
"""

import os
from types import MethodType

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _patch_scheduler(opt_param_scheduler) -> None:
    if getattr(opt_param_scheduler, "_relax_rocm_noop_step_patched", False):
        return

    def _noop_scheduler_step(self, increment):
        del self, increment
        logger.warning("ROCm diagnostic hook: skipping optimizer scheduler step")

    opt_param_scheduler.step = MethodType(_noop_scheduler_step, opt_param_scheduler)
    opt_param_scheduler._relax_rocm_noop_step_patched = True
    logger.warning("ROCm diagnostic hook: installed no-op optimizer scheduler step")


def hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
    del args, rollout_id, step_id, model

    mode = os.environ.get("RELAX_ROCM_OPTIMIZER_DIAGNOSTIC_MODE", "noop_optimizer_step")
    if mode == "noop_optimizer_step":
        if not getattr(optimizer, "_relax_rocm_noop_step_patched", False):

            def _noop_step(self):
                del self
                logger.warning("ROCm diagnostic hook: skipping optimizer.step()")
                return True, 0.0, None

            optimizer.step = MethodType(_noop_step, optimizer)
            optimizer._relax_rocm_noop_step_patched = True
            logger.warning("ROCm diagnostic hook: installed no-op optimizer.step()")
        _patch_scheduler(opt_param_scheduler)
        return

    if mode == "skip_step_with_ready_grads":
        if not getattr(optimizer, "_relax_rocm_skip_ready_step_patched", False):

            def _skip_step_with_ready_grads(self):
                del self
                logger.warning("ROCm diagnostic hook: skipping step_with_ready_grads()")
                return True

            optimizer.step_with_ready_grads = MethodType(_skip_step_with_ready_grads, optimizer)
            optimizer._relax_rocm_skip_ready_step_patched = True
            logger.warning("ROCm diagnostic hook: installed no-op step_with_ready_grads()")
        _patch_scheduler(opt_param_scheduler)
        return

    if mode == "prepare_only":
        if not getattr(optimizer, "_relax_rocm_prepare_only_patched", False):

            def _skip_step_with_ready_grads(self):
                del self
                logger.warning("ROCm diagnostic hook: prepare_only skipping step_with_ready_grads()")
                return True

            def _noop_clip_grad_norm(self, clip_grad):
                del self, clip_grad
                logger.warning("ROCm diagnostic hook: prepare_only skipping clip_grad_norm()")
                return 0.0

            def _noop_count_zeros(self):
                del self
                logger.warning("ROCm diagnostic hook: prepare_only skipping count_zeros()")
                return 0

            optimizer.step_with_ready_grads = MethodType(_skip_step_with_ready_grads, optimizer)
            optimizer.clip_grad_norm = MethodType(_noop_clip_grad_norm, optimizer)
            optimizer.count_zeros = MethodType(_noop_count_zeros, optimizer)
            optimizer._relax_rocm_prepare_only_patched = True
            logger.warning("ROCm diagnostic hook: installed prepare_only optimizer patches")
        _patch_scheduler(opt_param_scheduler)
        return

    if mode == "skip_inner_optimizer_step":
        base_optimizer = getattr(optimizer, "optimizer", None)
        if base_optimizer is None:
            raise RuntimeError("ROCm diagnostic hook expected Megatron optimizer to expose .optimizer")
        if not getattr(base_optimizer, "_relax_rocm_noop_inner_step_patched", False):

            def _noop_inner_step(self, closure=None):
                del self, closure
                logger.warning("ROCm diagnostic hook: skipping inner torch optimizer.step()")

            base_optimizer.step = MethodType(_noop_inner_step, base_optimizer)
            base_optimizer._relax_rocm_noop_inner_step_patched = True
            logger.warning("ROCm diagnostic hook: installed no-op inner torch optimizer.step()")
        _patch_scheduler(opt_param_scheduler)
        return

    if mode == "skip_main_to_model_copy":
        if not hasattr(optimizer, "_copy_main_params_to_model_params"):
            raise RuntimeError("ROCm diagnostic hook expected mixed-precision optimizer copy method")
        if not getattr(optimizer, "_relax_rocm_skip_main_to_model_copy_patched", False):

            def _skip_main_to_model_copy(self):
                del self
                logger.warning("ROCm diagnostic hook: skipping fp32 main-param to model-param copy")

            optimizer._copy_main_params_to_model_params = MethodType(_skip_main_to_model_copy, optimizer)
            optimizer._relax_rocm_skip_main_to_model_copy_patched = True
            logger.warning("ROCm diagnostic hook: installed no-op main-to-model copy")
        return

    raise RuntimeError(f"Unknown RELAX_ROCM_OPTIMIZER_DIAGNOSTIC_MODE={mode!r}")
