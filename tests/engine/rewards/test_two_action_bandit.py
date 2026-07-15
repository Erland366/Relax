# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
from relax.engine.rewards.two_action_bandit import get_two_action_bandit_reward
from relax.utils.types import Sample


def _make_args(**overrides) -> SimpleNamespace:
    defaults = {
        "rm_type": "two_action_bandit",
        "custom_rm_path": None,
        "reward_max_concurrency": 8,
        "reward_num_workers": 2,
        "reward_key": "score",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _reset_reward_executor():
    RewardExecutor._instance = None
    yield
    RewardExecutor._instance = None


@pytest.mark.parametrize("response", ["A", " A", "A\n", "A<|im_end|>"])
def test_two_action_bandit_rewards_exact_action_a(response):
    assert get_two_action_bandit_reward(Sample(response=response, response_length=2)) == {
        "score": 1.0,
        "action_a": 1.0,
        "action_b": 0.0,
        "valid_action": 1.0,
    }


@pytest.mark.parametrize("response", ["B", " B", "B\n", "B<|im_end|>"])
def test_two_action_bandit_scores_exact_action_b_as_baseline(response):
    assert get_two_action_bandit_reward(Sample(response=response, response_length=2)) == {
        "score": 0.0,
        "action_a": 0.0,
        "action_b": 1.0,
        "valid_action": 1.0,
    }


@pytest.mark.parametrize(
    "response",
    ["", "AB", "A B", "A.", "C", "answer: A", "AB<|im_end|>", "A<|im_end|>extra"],
)
def test_two_action_bandit_penalizes_invalid_responses(response):
    assert get_two_action_bandit_reward(Sample(response=response, response_length=2))["score"] == -1.0


@pytest.mark.parametrize("response_length", [0, 1, 3])
def test_two_action_bandit_requires_one_action_token_followed_by_eos(response_length):
    reward = get_two_action_bandit_reward(Sample(response="A", response_length=response_length))

    assert reward == {"score": -1.0, "action_a": 0.0, "action_b": 0.0, "valid_action": 0.0}


@pytest.mark.asyncio
async def test_async_rm_routes_two_action_bandit_locally(monkeypatch):
    args = _make_args()

    monkeypatch.setattr(
        RewardExecutor,
        "_ensure_workers",
        lambda self: (_ for _ in ()).throw(AssertionError("two_action_bandit should stay local")),
    )

    reward = await async_rm(args, Sample(response="A", response_length=2))

    assert reward["score"] == 1.0


@pytest.mark.asyncio
async def test_batched_async_rm_scores_two_action_bandit_responses():
    args = _make_args()
    samples = [
        Sample(response="A", response_length=2),
        Sample(response="B", response_length=2),
        Sample(response="anything else", response_length=2),
    ]

    rewards = await batched_async_rm(args, samples)

    assert [reward["score"] for reward in rewards] == [1.0, 0.0, -1.0]
