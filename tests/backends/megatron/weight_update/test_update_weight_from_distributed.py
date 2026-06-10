# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import torch

from relax.backends.megatron.weight_update import update_weight_from_distributed as update_module


class _RemoteMethod:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def remote(self):
        self.events.append(self.name)
        return f"{self.name}-ref"


class _FakeEngine:
    def __init__(self, events: list[str]) -> None:
        self.pause_generation = _RemoteMethod("pause", events)
        self.flush_cache = _RemoteMethod("flush", events)
        self.continue_generation = _RemoteMethod("continue", events)


def _make_updater(events: list[str], model_update_group):
    args = SimpleNamespace(update_weight_buffer_size=1, distributed_timeout_minutes=30)
    engine = _FakeEngine(events)
    updater = update_module.UpdateWeightFromDistributed(
        args,
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
    )
    updater.rollout_engines = [engine]
    updater.rollout_engine_lock = object()
    updater._engine_gpu_counts = [1]
    updater._is_pp_src_rank = True
    updater._group_name = "slime-pp_0"
    updater._model_update_groups = model_update_group
    return updater


def _patch_update_runtime(monkeypatch, accelerator_type):
    monkeypatch.setattr(update_module.ray, "get", lambda refs, *args, **kwargs: refs)
    monkeypatch.setattr(update_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(update_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(update_module, "named_params_and_buffers", lambda args, model: [])
    monkeypatch.setattr(update_module.device_utils, "get_accelerator_type", lambda: accelerator_type)


def test_rocm_update_tears_down_model_update_group_before_resuming_generation(monkeypatch):
    events = []
    old_group = object()
    updater = _make_updater(events, old_group)
    _patch_update_runtime(monkeypatch, update_module.device_utils.AcceleratorType.ROCM)

    disconnect_calls = []

    def fake_disconnect(args, group_name, model_update_groups, rollout_engines):
        disconnect_calls.append((args, group_name, model_update_groups, rollout_engines))
        events.append("disconnect")

    monkeypatch.setattr(update_module, "disconnect_rollout_engines_from_distributed", fake_disconnect)

    updater.update_weights()

    assert updater._model_update_groups is None
    assert disconnect_calls == [(updater.args, "slime-pp_0", old_group, updater.rollout_engines)]
    assert events[-2:] == ["disconnect", "continue"]


def test_rocm_update_reconnects_missing_model_update_group(monkeypatch):
    events = []
    new_group = object()
    updater = _make_updater(events, None)
    _patch_update_runtime(monkeypatch, update_module.device_utils.AcceleratorType.ROCM)

    connect_calls = []

    def fake_connect(args, group_name, rollout_engines, engine_gpu_counts=None):
        connect_calls.append((args, group_name, rollout_engines, engine_gpu_counts))
        events.append("connect")
        return new_group

    def fake_disconnect(args, group_name, model_update_groups, rollout_engines):
        events.append("disconnect")

    monkeypatch.setattr(update_module, "connect_rollout_engines_from_distributed", fake_connect)
    monkeypatch.setattr(update_module, "disconnect_rollout_engines_from_distributed", fake_disconnect)

    updater.update_weights()

    assert connect_calls == [(updater.args, "slime-pp_0", updater.rollout_engines, [1])]
    assert updater._model_update_groups is None
    assert events.index("connect") < events.index("disconnect") < events.index("continue")


def test_non_rocm_update_keeps_persistent_model_update_group(monkeypatch):
    events = []
    old_group = object()
    updater = _make_updater(events, old_group)
    _patch_update_runtime(monkeypatch, update_module.device_utils.AcceleratorType.CUDA)

    disconnect_calls = []
    monkeypatch.setattr(
        update_module,
        "disconnect_rollout_engines_from_distributed",
        lambda *args, **kwargs: disconnect_calls.append((args, kwargs)),
    )

    updater.update_weights()

    assert updater._model_update_groups is old_group
    assert disconnect_calls == []
    assert events[-1] == "continue"


def test_rocm_broadcast_uses_detached_contiguous_clone(monkeypatch):
    monkeypatch.setattr(
        update_module.device_utils,
        "get_accelerator_type",
        lambda: update_module.device_utils.AcceleratorType.ROCM,
    )
    original = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]

    prepared = update_module.prepare_broadcast_named_tensors([("weight", original)])

    assert prepared[0][0] == "weight"
    assert prepared[0][1].is_contiguous()
    assert prepared[0][1].data_ptr() != original.data_ptr()
    assert torch.equal(prepared[0][1], original)


def test_non_rocm_broadcast_uses_original_tensor(monkeypatch):
    monkeypatch.setattr(
        update_module.device_utils,
        "get_accelerator_type",
        lambda: update_module.device_utils.AcceleratorType.CUDA,
    )
    original = torch.arange(4, dtype=torch.float32)
    converted_named_tensors = [("weight", original)]

    prepared = update_module.prepare_broadcast_named_tensors(converted_named_tensors)

    assert prepared is converted_named_tensors
    assert prepared[0][1].data_ptr() == original.data_ptr()


def test_rocm_broadcast_uses_blocking_collectives(monkeypatch):
    monkeypatch.setattr(
        update_module.device_utils,
        "get_accelerator_type",
        lambda: update_module.device_utils.AcceleratorType.ROCM,
    )
    calls = []

    def fake_broadcast(tensor, src, group=None, async_op=False):
        calls.append((tensor.data_ptr(), src, group, async_op))

    monkeypatch.setattr(update_module.dist, "broadcast", fake_broadcast)

    tensor_a = torch.tensor([1.0])
    tensor_b = torch.tensor([2.0])
    update_module.broadcast_tensors_to_rollout("group", [("a", tensor_a), ("b", tensor_b)])

    assert calls == [(tensor_a.data_ptr(), 0, "group", False), (tensor_b.data_ptr(), 0, "group", False)]


def test_non_rocm_broadcast_uses_async_collectives(monkeypatch):
    monkeypatch.setattr(
        update_module.device_utils,
        "get_accelerator_type",
        lambda: update_module.device_utils.AcceleratorType.CUDA,
    )
    calls = []
    waited = []

    class FakeHandle:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait(self) -> None:
            waited.append(self.name)

    def fake_broadcast(tensor, src, group=None, async_op=False):
        calls.append((tensor.data_ptr(), src, group, async_op))
        return FakeHandle(f"handle-{len(calls)}")

    monkeypatch.setattr(update_module.dist, "broadcast", fake_broadcast)

    tensor = torch.tensor([1.0])
    update_module.broadcast_tensors_to_rollout("group", [("a", tensor)])

    assert calls == [(tensor.data_ptr(), 0, "group", True)]
    assert waited == ["handle-1"]
