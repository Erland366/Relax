# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.optimizer_utils import (
    TorchCPUAdamW,
    apply_single_rank_rocm_cpu_offload_optimizer_config,
    apply_single_rank_rocm_cpu_offload_safety_args,
    install_hybrid_device_optimizer_init_state_fn,
    normalize_single_rank_optimizer_args,
    patch_megatron_cpu_offload_optimizer_for_rocm,
    refresh_hybrid_device_optimizer_param_groups,
    should_disable_pinned_host_weight_backups,
)


def test_normalize_single_rank_optimizer_args_disables_distributed_optimizer_flags():
    args = Namespace(
        data_parallel_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.use_precision_aware_optimizer is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_uses_world_size_fallback():
    args = Namespace(
        world_size=1,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="critic")

    assert changed is True
    assert args.use_distributed_optimizer is False
    assert args.overlap_cpu_optimizer_d2h_h2d is False


def test_normalize_single_rank_optimizer_args_keeps_multi_rank_flags():
    args = Namespace(
        data_parallel_size=2,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
        overlap_cpu_optimizer_d2h_h2d=True,
    )

    changed = normalize_single_rank_optimizer_args(args, role="actor")

    assert changed is False
    assert args.use_distributed_optimizer is True
    assert args.use_precision_aware_optimizer is True
    assert args.overlap_param_gather is True
    assert args.overlap_param_gather_with_optimizer_step is True
    assert args.overlap_cpu_optimizer_d2h_h2d is True


def test_apply_single_rank_rocm_cpu_offload_safety_args(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(
        data_parallel_size=1,
        optimizer_cpu_offload=True,
        use_torch_optimizer_for_cpu_offload=False,
        pin_cpu_grads=True,
        pin_cpu_params=True,
    )

    changed = apply_single_rank_rocm_cpu_offload_safety_args(args, role="actor")

    assert changed is True
    assert args.use_torch_optimizer_for_cpu_offload is True
    assert args.pin_cpu_grads is False
    assert args.pin_cpu_params is False


def test_apply_single_rank_rocm_cpu_offload_safety_args_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", None)
    args = Namespace(
        data_parallel_size=1,
        optimizer_cpu_offload=True,
        use_torch_optimizer_for_cpu_offload=False,
        pin_cpu_grads=True,
        pin_cpu_params=True,
    )

    changed = apply_single_rank_rocm_cpu_offload_safety_args(args, role="actor")

    assert changed is False
    assert args.use_torch_optimizer_for_cpu_offload is False
    assert args.pin_cpu_grads is True
    assert args.pin_cpu_params is True


def test_apply_single_rank_rocm_cpu_offload_optimizer_config_bypasses_main_param_wrapper(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True)
    kwargs = {
        "bf16": True,
        "fp16": False,
        "use_precision_aware_optimizer": True,
    }

    changed = apply_single_rank_rocm_cpu_offload_optimizer_config(args, role="actor", kwargs=kwargs)

    assert changed is True
    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is False
    assert kwargs["use_precision_aware_optimizer"] is False


def test_apply_single_rank_rocm_cpu_offload_optimizer_config_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", None)
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True)
    kwargs = {
        "bf16": True,
        "fp16": False,
        "use_precision_aware_optimizer": True,
    }

    changed = apply_single_rank_rocm_cpu_offload_optimizer_config(args, role="actor", kwargs=kwargs)

    assert changed is False
    assert kwargs["bf16"] is True
    assert kwargs["use_precision_aware_optimizer"] is True


def test_patch_megatron_cpu_offload_optimizer_for_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    fake_module = SimpleNamespace(CPUAdam=object())
    args = Namespace(
        data_parallel_size=1,
        optimizer_cpu_offload=True,
        use_torch_optimizer_for_cpu_offload=True,
    )

    changed = patch_megatron_cpu_offload_optimizer_for_rocm(args, role="actor", module=fake_module)

    assert changed is True
    assert fake_module.CPUAdam is TorchCPUAdamW


def test_torch_cpu_adamw_forces_non_foreach_path(monkeypatch):
    captured = {}

    def fake_adamw_init(self, params, **kwargs):
        captured["params"] = list(params)
        captured["kwargs"] = kwargs
        self.param_groups = [{"params": captured["params"], "amsgrad": False}]
        self.state = defaultdict(dict)

    monkeypatch.setattr(torch.optim.AdamW, "__init__", fake_adamw_init)

    params = [torch.nn.Parameter(torch.zeros(1))]
    TorchCPUAdamW(params, lr=1e-3, fused=True, bias_correction=True)

    assert captured["params"] == params
    assert captured["kwargs"]["lr"] == 1e-3
    assert captured["kwargs"]["foreach"] is False
    assert "fused" not in captured["kwargs"]
    assert "bias_correction" not in captured["kwargs"]


def test_torch_cpu_adamw_keeps_state_lazy():
    param = torch.nn.Parameter(torch.zeros(2, 3))

    optimizer = TorchCPUAdamW([param], lr=1e-3)

    assert optimizer.state[param] == {}


def test_should_disable_pinned_host_weight_backups_for_single_rank_rocm_actor(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    args = Namespace(data_parallel_size=1, offload_train=False)

    assert should_disable_pinned_host_weight_backups(args, role="actor") is True


def test_should_disable_pinned_host_weight_backups_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", None)
    args = Namespace(data_parallel_size=1, offload_train=True)

    assert should_disable_pinned_host_weight_backups(args, role="actor") is False


class _HybridDeviceOptimizerDouble:
    def __init__(self):
        self.calls = []

    def _init_sub_optimizers(self):
        self.calls.append("_init_sub_optimizers")

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        self.calls.append("_sync_hdo_param_groups_to_sub_optimizers")


_FakeHybridDeviceOptimizer = type("HybridDeviceOptimizer", (_HybridDeviceOptimizerDouble,), {})


def test_refresh_hybrid_device_optimizer_param_groups_rebuilds_wrapped_hdo():
    inner_optimizer = _FakeHybridDeviceOptimizer()
    wrapper = Namespace(optimizer=inner_optimizer)

    changed = refresh_hybrid_device_optimizer_param_groups(wrapper)

    assert changed is True
    assert inner_optimizer.calls == [
        "_init_sub_optimizers",
        "_sync_hdo_param_groups_to_sub_optimizers",
    ]


def test_refresh_hybrid_device_optimizer_param_groups_handles_chained_optimizers():
    inner_optimizer = _FakeHybridDeviceOptimizer()
    wrapper = Namespace(chained_optimizers=[Namespace(optimizer=inner_optimizer)])

    changed = refresh_hybrid_device_optimizer_param_groups(wrapper)

    assert changed is True
    assert inner_optimizer.calls == [
        "_init_sub_optimizers",
        "_sync_hdo_param_groups_to_sub_optimizers",
    ]


def test_refresh_hybrid_device_optimizer_param_groups_skips_other_optimizers():
    wrapper = Namespace(optimizer=Namespace())

    changed = refresh_hybrid_device_optimizer_param_groups(wrapper)

    assert changed is False


_FakeFP32Optimizer = type(
    "FP32Optimizer",
    (),
    {"__init__": lambda self, optimizer: setattr(self, "optimizer", optimizer)},
)


def test_refresh_hybrid_device_optimizer_param_groups_skips_fp32_wrapper():
    inner_optimizer = _FakeHybridDeviceOptimizer()
    wrapper = _FakeFP32Optimizer(inner_optimizer)

    changed = refresh_hybrid_device_optimizer_param_groups(wrapper)

    assert changed is False
    assert inner_optimizer.calls == []


class _SubOptimizerDouble:
    def __init__(self, params):
        self.param_groups = [{"params": params}]
        self.state = defaultdict(dict)


class _HybridDeviceOptimizerStateDouble:
    def __init__(self):
        self.calls = []
        self.orig_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.cpu_param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.cpu_optimizers = [_SubOptimizerDouble([self.cpu_param])]
        self.gpu_optimizer = None
        self.state = defaultdict(dict)
        self.inner_param_to_orig_param = {self.cpu_param: self.orig_param}

    @property
    def sub_optimizers(self):
        return self.cpu_optimizers

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        self.calls.append("_sync_hdo_param_groups_to_sub_optimizers")

    def _sync_sub_optimizers_state_to_hdo(self):
        self.calls.append("_sync_sub_optimizers_state_to_hdo")
        self.state = defaultdict(dict)
        for sub_optimizer in self.sub_optimizers:
            for param, state in sub_optimizer.state.items():
                self.state[self.inner_param_to_orig_param[param]] = state


_FakeStatefulHybridDeviceOptimizer = type(
    "HybridDeviceOptimizer",
    (_HybridDeviceOptimizerStateDouble,),
    {},
)


class _HybridDeviceOptimizerLoadStateDouble(torch.optim.Optimizer):
    def __init__(self):
        self.orig_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.cpu_param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        super().__init__([{"params": [self.orig_param], "lr": 0.1}], defaults={})
        self.cpu_optimizers = [_SubOptimizerDouble([self.cpu_param])]
        self.gpu_optimizer = None
        self.param_to_inner_param = {self.orig_param: self.cpu_param}
        self.inner_param_to_orig_param = {self.cpu_param: self.orig_param}

    @property
    def sub_optimizers(self):
        return self.cpu_optimizers

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        public_group = self.param_groups[0].copy()
        public_group.pop("params")
        for sub_optimizer in self.sub_optimizers:
            for group in sub_optimizer.param_groups:
                group.update(public_group)

    def _sync_sub_optimizers_state_to_hdo(self):
        self.state = defaultdict(dict)
        for sub_optimizer in self.sub_optimizers:
            for param, state in sub_optimizer.state.items():
                self.state[self.inner_param_to_orig_param[param]] = state


_FakeLoadStateHybridDeviceOptimizer = type(
    "HybridDeviceOptimizer",
    (_HybridDeviceOptimizerLoadStateDouble,),
    {},
)


class _FP32OptimizerLoadStateDouble:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.init_state_fn = None
        self.config = Namespace(optimizer="adam")

    def _filter_and_reorder_param_groups(self, current_param_groups, saved_param_groups):
        return saved_param_groups

    def _restore_common_per_param_step(self, state_dict, step):
        for param_state in state_dict["state"].values():
            param_state["step"] = step


_FakeLoadStateFP32Optimizer = type(
    "FP32Optimizer",
    (_FP32OptimizerLoadStateDouble,),
    {},
)


def test_install_hybrid_device_optimizer_init_state_fn_initializes_cpu_adam_state(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    inner_optimizer = _FakeStatefulHybridDeviceOptimizer()
    wrapper = Namespace(
        optimizer=inner_optimizer,
        init_state_fn=None,
        config=Namespace(optimizer="adam"),
    )
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True, optimizer="adam")

    changed = install_hybrid_device_optimizer_init_state_fn(wrapper, args, role="actor")

    assert changed is True
    assert wrapper.init_state_fn is not None

    wrapper.init_state_fn(inner_optimizer, wrapper.config)

    cpu_state = inner_optimizer.cpu_optimizers[0].state[inner_optimizer.cpu_param]
    assert cpu_state["exp_avg"].shape == inner_optimizer.cpu_param.shape
    assert cpu_state["exp_avg"].dtype == inner_optimizer.cpu_param.dtype
    assert cpu_state["exp_avg"].device == inner_optimizer.cpu_param.device
    assert cpu_state["exp_avg_sq"].shape == inner_optimizer.cpu_param.shape
    assert inner_optimizer.state[inner_optimizer.orig_param] is cpu_state
    assert inner_optimizer.calls == [
        "_sync_hdo_param_groups_to_sub_optimizers",
        "_sync_sub_optimizers_state_to_hdo",
    ]


def test_install_hybrid_device_optimizer_init_state_fn_handles_chained_wrapper(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    inner_optimizer = _FakeStatefulHybridDeviceOptimizer()
    fp32_wrapper = Namespace(
        optimizer=inner_optimizer,
        init_state_fn=None,
        config=Namespace(optimizer="adam"),
    )
    chained_wrapper = Namespace(chained_optimizers=[fp32_wrapper])
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True, optimizer="adam")

    changed = install_hybrid_device_optimizer_init_state_fn(chained_wrapper, args, role="actor")

    assert changed is True
    assert fp32_wrapper.init_state_fn is not None


def test_install_hybrid_device_optimizer_load_state_dict_loads_state_on_inner_params(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    inner_optimizer = _FakeLoadStateHybridDeviceOptimizer()
    wrapper = _FakeLoadStateFP32Optimizer(inner_optimizer)
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True, optimizer="adam")
    state_dict = {
        "state": {
            0: {
                "exp_avg": torch.full((2,), 2.0, dtype=torch.bfloat16),
                "exp_avg_sq": torch.full((2,), 3.0, dtype=torch.bfloat16),
            },
            "common_step": torch.tensor(4.0),
        },
        "param_groups": [{"params": [0], "lr": 0.2}],
    }

    changed = install_hybrid_device_optimizer_init_state_fn(wrapper, args, role="actor")
    wrapper.load_state_dict(state_dict)

    assert changed is True
    assert inner_optimizer.param_groups[0]["params"] == [inner_optimizer.orig_param]
    assert inner_optimizer.param_groups[0]["lr"] == 0.2
    cpu_state = inner_optimizer.cpu_optimizers[0].state[inner_optimizer.cpu_param]
    assert torch.equal(cpu_state["exp_avg"], torch.full((2,), 2.0, dtype=torch.bfloat16))
    assert torch.equal(cpu_state["exp_avg_sq"], torch.full((2,), 3.0, dtype=torch.bfloat16))
    assert torch.equal(cpu_state["step"], torch.tensor(4.0))
    assert inner_optimizer.state[inner_optimizer.orig_param] is cpu_state
    assert inner_optimizer.orig_param not in inner_optimizer.cpu_optimizers[0].state


def test_install_hybrid_device_optimizer_init_state_fn_skips_non_rocm(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", None)
    inner_optimizer = _FakeStatefulHybridDeviceOptimizer()
    wrapper = Namespace(
        optimizer=inner_optimizer,
        init_state_fn=None,
        config=Namespace(optimizer="adam"),
    )
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True, optimizer="adam")

    changed = install_hybrid_device_optimizer_init_state_fn(wrapper, args, role="actor")

    assert changed is False
    assert wrapper.init_state_fn is None


def test_install_hybrid_device_optimizer_init_state_fn_fails_loudly_for_non_adam(monkeypatch):
    monkeypatch.setattr("relax.backends.megatron.optimizer_utils.torch.version.hip", "6.3.0")
    inner_optimizer = _FakeStatefulHybridDeviceOptimizer()
    wrapper = Namespace(
        optimizer=inner_optimizer,
        init_state_fn=None,
        config=Namespace(optimizer="sgd"),
    )
    args = Namespace(data_parallel_size=1, optimizer_cpu_offload=True, optimizer="sgd")

    with pytest.raises(RuntimeError, match="only supports Adam"):
        install_hybrid_device_optimizer_init_state_fn(wrapper, args, role="actor")


def test_local_hybrid_device_optimizer_uses_single_param_chunks_for_rocm_safe_path(monkeypatch):
    hybrid_optimizer_path = (
        Path(__file__).resolve().parents[3]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location("local_hybrid_optimizer", hybrid_optimizer_path)
    assert spec is not None
    assert spec.loader is not None
    hybrid_optimizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_optimizer_module)
    hybrid_optimizer_cls = hybrid_optimizer_module.HybridDeviceOptimizer

    class DummyCPUOptimizer:
        def __init__(self, param_groups):
            self.param_groups = param_groups

    cpu_params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    cpu_param_groups = [{"params": cpu_params, "lr": 1e-3}]

    def fake_get_sub_optimizer_param_groups(self, offload_fraction):
        return cpu_param_groups, [], {}, {}, {}

    monkeypatch.setattr(
        hybrid_optimizer_cls,
        "_get_sub_optimizer_param_groups",
        fake_get_sub_optimizer_param_groups,
    )
    monkeypatch.setattr(
        "torch.cuda.current_stream",
        lambda: SimpleNamespace(wait_stream=lambda *_: None),
    )

    optimizer = hybrid_optimizer_cls(
        [{"params": cpu_params}],
        cpu_optimizer_cls=DummyCPUOptimizer,
        gpu_optimizer_cls=None,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        overlap_cpu_optimizer_d2h_h2d=False,
    )

    assert len(optimizer.cpu_optimizers) == 3
    assert all(len(sub_optimizer.param_groups[0]["params"]) == 1 for sub_optimizer in optimizer.cpu_optimizers)


def test_local_hybrid_device_optimizer_keeps_bf16_cpu_params_on_rocm_safe_path(monkeypatch):
    hybrid_optimizer_path = (
        Path(__file__).resolve().parents[3]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location("local_hybrid_optimizer", hybrid_optimizer_path)
    assert spec is not None
    assert spec.loader is not None
    hybrid_optimizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_optimizer_module)
    hybrid_optimizer_cls = hybrid_optimizer_module.HybridDeviceOptimizer

    class DummyCPUOptimizer:
        def __init__(self, param_groups):
            self.param_groups = param_groups

    cpu_params = [torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)) for _ in range(2)]
    cpu_param_groups = [{"params": cpu_params, "lr": 1e-3}]

    def fake_get_sub_optimizer_param_groups(self, offload_fraction):
        return cpu_param_groups, [], {}, {}, {}

    monkeypatch.setattr(
        hybrid_optimizer_cls,
        "_get_sub_optimizer_param_groups",
        fake_get_sub_optimizer_param_groups,
    )
    monkeypatch.setattr(
        "torch.cuda.current_stream",
        lambda: SimpleNamespace(wait_stream=lambda *_: None),
    )
    monkeypatch.setattr("torch.version.hip", "6.3.0")

    optimizer = hybrid_optimizer_cls(
        [{"params": cpu_params}],
        cpu_optimizer_cls=DummyCPUOptimizer,
        gpu_optimizer_cls=None,
        param_update_in_fp32=True,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        overlap_cpu_optimizer_d2h_h2d=False,
    )

    cpu_optimizer_param = optimizer.cpu_optimizers[0].param_groups[0]["params"][0]
    assert cpu_optimizer_param.dtype == torch.bfloat16
    assert optimizer.param_update_in_fp32 is False
