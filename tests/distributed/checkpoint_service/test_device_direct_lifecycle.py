# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for DeviceDirectBackend rollout-engine lifecycle ordering."""

from types import SimpleNamespace

from relax.distributed.checkpoint_service.backends import device_direct


class _RemoteMethod:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def remote(self):
        self._events.append(f"{self._name}.remote")
        return f"{self._name}-ref"


class _RolloutEngine:
    def __init__(self, events: list[str]) -> None:
        self.flush_cache = _RemoteMethod(events, "flush")


def test_update_weights_waits_for_resume_before_cleanup(monkeypatch):
    events: list[str] = []
    backend = device_direct.DeviceDirectBackend.__new__(device_direct.DeviceDirectBackend)
    backend.weight_version = 0
    backend.args = SimpleNamespace()
    backend.model = []
    backend._is_pp_src_rank = False
    backend.rollout_engines = {0: _RolloutEngine(events)}

    def batch_request(endpoint, payload=None, get_rank=False):
        del payload, get_rank
        events.append(f"batch:{endpoint}")
        return [f"{endpoint}-ref"]

    def ray_get(refs):
        events.append(f"ray.get:{refs}")

    backend._batch_request = batch_request
    backend._cleanup_rollout_engines = lambda: events.append("cleanup")

    monkeypatch.setattr(device_direct.ray, "get", ray_get)
    monkeypatch.setattr(device_direct.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(device_direct.dist, "barrier", lambda group=None: events.append("barrier"))
    monkeypatch.setattr(device_direct, "get_gloo_group", lambda: None)
    monkeypatch.setattr(device_direct, "_get_weight_update_common", lambda: (None, lambda args, model: []))
    monkeypatch.setattr(device_direct.device_utils, "empty_cache", lambda: None)

    backend.update_weights_for_rollout(rollout_only=True)

    resume_wait = "ray.get:['/continue_generation-ref']"
    assert resume_wait in events
    assert events.index(resume_wait) < events.index("cleanup")
