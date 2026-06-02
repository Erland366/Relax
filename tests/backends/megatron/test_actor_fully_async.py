# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.backends.megatron import actor as actor_module


class _RemoteMethod:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def remote(self) -> str:
        self.events.append(f"{self.name}.remote")
        return f"{self.name}-ref"


class _FakeWeightSyncLock:
    def __init__(self, events: list[str]) -> None:
        self.acquire = _RemoteMethod(events, "acquire")
        self.release = _RemoteMethod(events, "release")


class _FakeCheckpointEngineClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def init_process_groups_for_actor_fwd_ref(self, rollout_id: int) -> str:
        self.events.append(f"init_actor_fwd_ref:{rollout_id}")
        return "init-actor-fwd-ref"

    def update_weights_for_rollout(self, rollout_only: bool, actor_fwd_only: bool) -> str:
        self.events.append(f"update_rollout:{rollout_only}:{actor_fwd_only}")
        return "update-rollout"


class _MegatronActorHarness:
    update_weights_fully_async = actor_module.MegatronTrainRayActor.update_weights_fully_async


def _install_update_weights_fakes(monkeypatch, events: list[str]) -> None:
    monkeypatch.setattr(actor_module.dist, "barrier", lambda group=None: events.append("barrier"))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: "gloo-group")
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: events.append("print_memory"))
    monkeypatch.setattr(actor_module, "run", lambda value: events.append(f"run:{value}"))

    def fake_ray_get(ref):
        events.append(f"ray.get:{ref}")
        if ref == "acquire-ref":
            return True
        return None

    monkeypatch.setattr(actor_module.ray, "get", fake_ray_get)


def test_update_weights_fully_async_initializes_actor_fwd_before_rollout_lock(monkeypatch):
    events: list[str] = []
    _install_update_weights_fakes(monkeypatch, events)

    actor = _MegatronActorHarness()
    actor._weight_sync_lock = _FakeWeightSyncLock(events)
    actor.checkpoint_engine_client = _FakeCheckpointEngineClient(events)

    actor.update_weights_fully_async(rollout_id=7)

    assert events == [
        "barrier",
        "print_memory",
        "init_actor_fwd_ref:7",
        "run:init-actor-fwd-ref",
        "acquire.remote",
        "ray.get:acquire-ref",
        "update_rollout:False:False",
        "run:update-rollout",
        "release.remote",
        "ray.get:release-ref",
    ]


def test_update_weights_fully_async_skips_rollout_lock_for_actor_fwd_only(monkeypatch):
    events: list[str] = []
    _install_update_weights_fakes(monkeypatch, events)

    actor = _MegatronActorHarness()
    actor._weight_sync_lock = _FakeWeightSyncLock(events)
    actor.checkpoint_engine_client = _FakeCheckpointEngineClient(events)

    actor.update_weights_fully_async(rollout_id=3, actor_fwd_only=True)

    assert "acquire.remote" not in events
    assert "release.remote" not in events
    assert events == [
        "barrier",
        "print_memory",
        "init_actor_fwd_ref:3",
        "run:init-actor-fwd-ref",
        "update_rollout:False:True",
        "run:update-rollout",
    ]
