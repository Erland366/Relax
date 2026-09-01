# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import threading
from argparse import Namespace

import pytest

from tests.core.test_vision_encoder_integration import _install_lightweight_controller_stubs


@pytest.fixture
def controller_module(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    _install_lightweight_controller_stubs(monkeypatch)
    logging_utils = importlib.import_module("relax.utils.logging_utils")
    monkeypatch.setattr(logging_utils.LazyConfiguredLogger, "_configured", True)
    monkeypatch.delitem(sys.modules, "relax.core.controller", raising=False)
    module = importlib.import_module("relax.core.controller")
    yield module
    sys.modules.pop("relax.core.controller", None)


class _RemoteCall:
    def __init__(self, function):
        self._function = function

    def remote(self):
        return self._function()


class _FakeRolloutManager:
    def __init__(self, preload):
        self.preload_vision_features = _RemoteCall(preload)


class _FakeService:
    def __init__(self, role, events, *, rollout_manager=None):
        self.role = role
        self._events = events
        self._rollout_manager = rollout_manager

    async def get_rollout_manager(self):
        return self._rollout_manager

    async def set_rollout_manager(self, rollout_manager):
        self._events.append("set_rollout_manager")

    async def update_weights_fully_async(self):
        self._events.append("initial_weight_sync")

    async def get_step(self):
        return 0

    async def set_step(self, step):
        self._events.append(f"set_step:{self.role}:{step}")

    def run(self):
        async def _run():
            self._events.append(f"run:{self.role}")

        return _run()


def _controller_for_preload(controller_module, events, preload):
    rollout_manager = _FakeRolloutManager(preload)
    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        fully_async=True,
        hybrid=False,
        preload_vision_features=True,
    )
    controller.serve_dict = {
        controller_module.ROLES.actor: _FakeService(controller_module.ROLES.actor, events),
        controller_module.ROLES.rollout: _FakeService(
            controller_module.ROLES.rollout,
            events,
            rollout_manager=rollout_manager,
        ),
    }
    controller._pending_task_refs = []
    controller._pending_task_refs_lock = threading.Lock()
    controller._metrics_service_enabled = False
    controller._restarting = False
    return controller


def test_training_loop_awaits_feature_preload_after_weight_sync_before_service_run(controller_module, caplog):
    caplog.set_level("INFO", logger=controller_module.__name__)
    events = []
    metrics = {
        "schema_version": 2,
        "dataset_samples": 64,
        "unique_features": 64,
        "duplicate_features": 0,
    }

    async def preload():
        events.append("preload")
        return metrics

    controller = _controller_for_preload(controller_module, events, preload)

    controller.training_loop()

    first_run = min(index for index, event in enumerate(events) if event.startswith("run:"))
    assert "preload" in events, f"controller skipped the enabled preload barrier: {events}"
    assert events.index("initial_weight_sync") < events.index("preload") < first_run
    assert any(
        "VISION_FEATURE_PRELOAD" in message and repr(metrics) in message
        for message in caplog.messages
    )


def test_training_loop_does_not_start_any_service_when_feature_preload_fails(controller_module):
    events = []

    async def preload():
        events.append("preload")
        raise RuntimeError("SGLang cache failed for feature-b")

    controller = _controller_for_preload(controller_module, events, preload)

    with pytest.raises(RuntimeError, match="SGLang cache failed for feature-b"):
        controller.training_loop()

    assert "initial_weight_sync" in events
    assert "preload" in events
    assert not any(event.startswith("run:") for event in events)
