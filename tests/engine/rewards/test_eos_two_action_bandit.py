# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
from relax.engine.rewards.eos_two_action_bandit import get_eos_two_action_bandit_reward
from relax.utils.metrics.metric_utils import compute_rollout_explicit_reward_metrics
from relax.utils.types import Sample


def _make_args(**overrides) -> SimpleNamespace:
    defaults = {
        "rm_type": "eos_two_action_bandit",
        "custom_rm_path": None,
        "reward_max_concurrency": 8,
        "reward_num_workers": 2,
        "reward_key": "score",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(task: str, response: str, response_length: int) -> Sample:
    return Sample(
        response=response,
        response_length=response_length,
        status=Sample.Status.COMPLETED,
        metadata={"task": task},
    )


@pytest.fixture(autouse=True)
def _reset_reward_executor():
    RewardExecutor._instance = None
    yield
    RewardExecutor._instance = None


def test_mixed_reward_scores_immediate_and_longer_eos_responses():
    immediate = get_eos_two_action_bandit_reward(_make_sample("eos", "", 1))
    longer = get_eos_two_action_bandit_reward(_make_sample("eos", "OK", 2))

    assert immediate == {
        "score": -1.0,
        "eos_reward": -1.0,
        "eos_response_length": 1.0,
        "immediate_eos": 1.0,
        "task_eos": 1.0,
        "task_bandit": 0.0,
    }
    assert longer["score"] == -2.0
    assert longer["immediate_eos"] == 0.0


@pytest.mark.parametrize(
    ("response", "expected_score", "metric"),
    [("A", 1.0, "action_a"), ("B", 0.0, "action_b"), ("C", -1.0, None)],
)
def test_mixed_reward_scores_bandit_actions(response, expected_score, metric):
    reward = get_eos_two_action_bandit_reward(_make_sample("two_action_bandit", response, 2))

    assert reward["score"] == expected_score
    assert reward["bandit_score"] == expected_score
    assert reward["task_eos"] == 0.0
    assert reward["task_bandit"] == 1.0
    assert "eos_reward" not in reward
    if metric is not None:
        assert reward[metric] == 1.0


@pytest.mark.parametrize("metadata", [{}, {"task": "unknown"}, None])
def test_mixed_reward_rejects_missing_or_unknown_task(metadata):
    sample = Sample(response="", response_length=1, metadata=metadata)

    with pytest.raises(ValueError, match=r"metadata\['task'\].*one of"):
        get_eos_two_action_bandit_reward(sample)


@pytest.mark.asyncio
async def test_async_rm_routes_mixed_reward_locally(monkeypatch):
    monkeypatch.setattr(
        RewardExecutor,
        "_ensure_workers",
        lambda self: (_ for _ in ()).throw(AssertionError("mixed reward should stay local")),
    )

    reward = await async_rm(_make_args(), _make_sample("two_action_bandit", "A", 2))

    assert reward["score"] == 1.0


@pytest.mark.asyncio
async def test_mixed_batch_has_one_numeric_primary_reward_contract():
    args = _make_args()
    samples = [
        _make_sample("eos", "", 1),
        _make_sample("eos", "OK", 2),
        _make_sample("two_action_bandit", "A", 2),
        _make_sample("two_action_bandit", "B", 2),
    ]

    rewards = await batched_async_rm(args, samples)
    for sample, reward in zip(samples, rewards, strict=True):
        sample.reward = reward

    assert [sample.get_reward_value(args) for sample in samples] == [-1.0, -2.0, 1.0, 0.0]


def test_mixed_reward_metrics_keep_task_means_separate():
    args = _make_args()
    samples = [
        _make_sample("eos", "", 1),
        _make_sample("eos", "OK", 2),
        _make_sample("two_action_bandit", "A", 2),
        _make_sample("two_action_bandit", "B", 2),
    ]
    for sample in samples:
        sample.reward = get_eos_two_action_bandit_reward(sample)

    metrics = compute_rollout_explicit_reward_metrics(args, samples)

    assert metrics["task_eos/mean"] == 0.5
    assert metrics["task_bandit/mean"] == 0.5
    assert metrics["eos_reward/mean"] == -1.5
    assert metrics["bandit_score/mean"] == 0.5
    assert "score/mean" not in metrics
