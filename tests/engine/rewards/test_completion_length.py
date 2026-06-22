# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
from relax.engine.rewards.completion_length import get_completion_length_reward
from relax.utils.types import Sample


def _make_args(**overrides) -> SimpleNamespace:
    defaults = {
        "rm_type": "completion_length",
        "custom_rm_path": None,
        "reward_max_concurrency": 8,
        "reward_num_workers": 2,
        "reward_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _reset_reward_executor():
    RewardExecutor._instance = None
    yield
    RewardExecutor._instance = None


def test_completion_length_reward_scores_immediate_eos_as_minus_one():
    sample = Sample(response="", response_length=1)

    assert get_completion_length_reward(sample) == -1.0


def test_completion_length_reward_penalizes_longer_completions():
    short = Sample(response="a", response_length=2)
    long = Sample(response="longer", response_length=8)

    assert get_completion_length_reward(short) == -2.0
    assert get_completion_length_reward(long) == -8.0
    assert get_completion_length_reward(short) > get_completion_length_reward(long)


def test_completion_length_reward_rejects_negative_response_length():
    sample = Sample(response="", response_length=-1)

    with pytest.raises(AssertionError, match="response_length must be non-negative"):
        get_completion_length_reward(sample)


@pytest.mark.asyncio
async def test_async_rm_routes_completion_length_locally(monkeypatch):
    args = _make_args()
    sample = Sample(response="", response_length=1)

    monkeypatch.setattr(
        RewardExecutor,
        "_ensure_workers",
        lambda self: (_ for _ in ()).throw(AssertionError("completion_length should stay local")),
    )

    assert await async_rm(args, sample) == -1.0


@pytest.mark.asyncio
async def test_batched_async_rm_returns_negative_response_lengths():
    args = _make_args()
    samples = [
        Sample(response="", response_length=1),
        Sample(response="abc", response_length=3),
        Sample(response="abcdef", response_length=6),
    ]

    assert await batched_async_rm(args, samples) == [-1.0, -3.0, -6.0]


@pytest.mark.asyncio
async def test_sample_metadata_can_override_to_completion_length():
    args = _make_args(rm_type="random")
    sample = Sample(response="", response_length=1, metadata={"rm_type": "completion_length"})

    assert await async_rm(args, sample) == -1.0
