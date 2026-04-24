# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from unittest.mock import MagicMock

import ray

from relax.distributed.ray.train_actor import TrainRayActor, should_sleep_train_actor_after_init


class _DummyTrainActor(TrainRayActor):
    def sleep(self, tags=None):
        return None

    def wake_up(self, tags=None):
        return None

    def train(self, rollout_id, rollout_data_ref=None):
        return None

    def save_model(self, rollout_id, force_sync=False):
        return None

    def update_weights(self):
        return None

    def _get_parallel_config(self):
        return {}


def test_set_rollout_manager_skips_async_only_setup_for_sync_training(monkeypatch):
    actor = _DummyTrainActor.__new__(_DummyTrainActor)
    actor.args = Namespace(fully_async=False, debug_rollout_only=False, rank=0, offload_train=False)
    actor.train_parallel_config = {"dp_size": 1}

    rollout_manager = MagicMock()

    actor.set_rollout_manager(rollout_manager)

    assert actor.rollout_manager is rollout_manager
    assert not hasattr(actor, "_weight_sync_lock")
    rollout_manager.set_train_parallel_config.remote.assert_not_called()
    rollout_manager.get_weight_sync_lock.remote.assert_not_called()


def test_set_rollout_manager_keeps_async_setup_for_fully_async_training(monkeypatch):
    actor = _DummyTrainActor.__new__(_DummyTrainActor)
    actor.args = Namespace(fully_async=True, debug_rollout_only=False, rank=0, offload_train=False)
    actor.train_parallel_config = {"dp_size": 1}

    rollout_manager = MagicMock()
    rollout_manager.set_train_parallel_config.remote.return_value = "set-config-ref"
    rollout_manager.get_weight_sync_lock.remote.return_value = "lock-ref"

    def fake_ray_get(ref):
        if ref == "set-config-ref":
            return None
        if ref == "lock-ref":
            return "lock-handle"
        raise AssertionError(f"unexpected ref: {ref!r}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    actor.set_rollout_manager(rollout_manager)

    rollout_manager.set_train_parallel_config.remote.assert_called_once_with(actor.train_parallel_config)
    rollout_manager.get_weight_sync_lock.remote.assert_called_once_with()
    assert actor._weight_sync_lock == "lock-handle"


def test_set_rollout_manager_wakes_offloaded_actor_before_setup(monkeypatch):
    actor = _DummyTrainActor.__new__(_DummyTrainActor)
    actor.args = Namespace(fully_async=False, debug_rollout_only=False, rank=0, offload_train=True)
    actor._is_sleeping = True
    actor.train_parallel_config = {"dp_size": 1}
    actor.wake_up = MagicMock()

    rollout_manager = MagicMock()

    actor.set_rollout_manager(rollout_manager)

    actor.wake_up.assert_called_once_with()
    assert actor.rollout_manager is rollout_manager
    rollout_manager.set_train_parallel_config.remote.assert_not_called()
    rollout_manager.get_weight_sync_lock.remote.assert_not_called()


def test_set_rollout_manager_skips_wake_for_resident_offloaded_actor(monkeypatch):
    actor = _DummyTrainActor.__new__(_DummyTrainActor)
    actor.args = Namespace(fully_async=False, debug_rollout_only=False, rank=0, offload_train=True)
    actor._is_sleeping = False
    actor.train_parallel_config = {"dp_size": 1}
    actor.wake_up = MagicMock()

    rollout_manager = MagicMock()

    actor.set_rollout_manager(rollout_manager)

    actor.wake_up.assert_not_called()
    assert actor.rollout_manager is rollout_manager
    rollout_manager.set_train_parallel_config.remote.assert_not_called()
    rollout_manager.get_weight_sync_lock.remote.assert_not_called()


def test_should_sleep_train_actor_after_init_requires_offload():
    args = Namespace(offload_train=False, fully_async=False, colocate=True)

    assert should_sleep_train_actor_after_init(args) is False


def test_should_sleep_train_actor_after_init_sleeps_for_fully_async():
    args = Namespace(offload_train=True, fully_async=True, colocate=False)

    assert should_sleep_train_actor_after_init(args) is True


def test_should_sleep_train_actor_after_init_sleeps_for_sync_colocate():
    args = Namespace(offload_train=True, fully_async=False, colocate=True)

    assert should_sleep_train_actor_after_init(args) is True


def test_should_sleep_train_actor_after_init_keeps_non_colocated_sync_resident():
    args = Namespace(offload_train=True, fully_async=False, colocate=False)

    assert should_sleep_train_actor_after_init(args) is False
