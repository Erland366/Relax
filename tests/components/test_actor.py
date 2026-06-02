# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from unittest.mock import MagicMock

import ray

from relax.components.actor import Actor


ActorClass = Actor.func_or_class


class _ActorHarness:
    _execute_training = ActorClass._execute_training
    _maybe_save_model = ActorClass._maybe_save_model


def test_execute_training_lets_backend_handle_hybrid_checkpoint(monkeypatch):
    actor = _ActorHarness()
    actor.config = Namespace(
        fully_async=True,
        hybrid=True,
        num_critic_only_steps=0,
        num_rollout=4,
        rotate_ckpt=False,
        save="/tmp/checkpoints",
        save_interval=2,
    )
    actor.step = 1
    actor.actor_model = MagicMock()
    actor.actor_model.train_hybrid.return_value = ["train-ref"]

    monkeypatch.setattr(ray, "get", lambda refs: None)

    actor._execute_training()

    actor.actor_model.train_hybrid.assert_called_once_with(1)
    actor.actor_model.save_model.assert_not_called()


def test_execute_training_saves_checkpoint_for_fully_async(monkeypatch):
    actor = _ActorHarness()
    actor.config = Namespace(
        fully_async=True,
        hybrid=False,
        num_critic_only_steps=0,
        num_rollout=2,
        rotate_ckpt=False,
        save="/tmp/checkpoints",
        save_interval=10,
    )
    actor.step = 1
    actor.actor_model = MagicMock()
    actor.actor_model.train_fully_async.return_value = ["train-ref"]

    monkeypatch.setattr(ray, "get", lambda refs: None)

    actor._execute_training()

    actor.actor_model.train_fully_async.assert_called_once_with(1)
    actor.actor_model.save_model.assert_called_once_with(1, force_sync=True)
