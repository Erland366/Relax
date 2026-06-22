# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.utils.metrics.metric_utils import compute_rollout_primary_reward_metrics
from relax.utils.types import Sample


def test_compute_rollout_primary_reward_metrics_from_scalar_rewards():
    args = SimpleNamespace(reward_key=None)
    samples = [
        Sample(reward=-4.0),
        Sample(reward=-2.0),
        Sample(reward=-1.0),
    ]

    assert compute_rollout_primary_reward_metrics(args, samples) == {
        "reward/mean": -7.0 / 3.0,
        "reward/median": -2.0,
        "reward/max": -1.0,
        "reward/min": -4.0,
    }


def test_compute_rollout_primary_reward_metrics_from_reward_key():
    args = SimpleNamespace(reward_key="score")
    samples = [
        Sample(reward={"score": -3.0, "acc": 0.0}),
        Sample(reward={"score": -1.0, "acc": 1.0}),
    ]

    assert compute_rollout_primary_reward_metrics(args, samples) == {
        "reward/mean": -2.0,
        "reward/median": -2.0,
        "reward/max": -1.0,
        "reward/min": -3.0,
    }
