from types import SimpleNamespace

import pytest

from relax.engine.rewards import RewardExecutor, async_rm
from relax.engine.rewards.visual_xor import get_visual_xor_reward
from relax.utils.metrics.metric_utils import compute_rollout_explicit_reward_metrics
from relax.utils.types import Sample


def _sample(
    response: str,
    *,
    label: str = "A",
    combination: str = "00",
    response_length: int = 2,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    left_bit, right_bit = map(int, combination)
    return Sample(
        response=response,
        response_length=response_length,
        status=status,
        label=label,
        metadata={"left_bit": left_bit, "right_bit": right_bit, "combination": combination},
    )


@pytest.mark.parametrize("response", ["A", " A", "A\n", "A<|im_end|>"])
def test_visual_xor_rewards_exact_correct_action(response):
    reward = get_visual_xor_reward(_sample(response))

    assert reward["score"] == 1.0
    assert reward["correct_action"] == 1.0
    assert reward["valid_action"] == 1.0
    assert reward["combo_00_accuracy"] == 1.0


def test_visual_xor_scores_valid_incorrect_action_as_zero():
    reward = get_visual_xor_reward(_sample("B"))

    assert reward["score"] == 0.0
    assert reward["correct_action"] == 0.0
    assert reward["valid_action"] == 1.0
    assert reward["action_b"] == 1.0


@pytest.mark.parametrize(
    ("response", "response_length", "status"),
    [
        ("", 1, Sample.Status.COMPLETED),
        ("A.", 2, Sample.Status.COMPLETED),
        ("answer: A", 2, Sample.Status.COMPLETED),
        ("AB", 2, Sample.Status.COMPLETED),
        ("A", 3, Sample.Status.COMPLETED),
        ("A", 2, Sample.Status.TRUNCATED),
    ],
)
def test_visual_xor_penalizes_malformed_or_truncated_responses(response, response_length, status):
    reward = get_visual_xor_reward(_sample(response, response_length=response_length, status=status))

    assert reward["score"] == -1.0
    assert reward["valid_action"] == 0.0


@pytest.mark.parametrize(
    ("label", "metadata", "message"),
    [
        (None, {"left_bit": 0, "right_bit": 0, "combination": "00"}, "label"),
        ("C", {"left_bit": 0, "right_bit": 0, "combination": "00"}, "label"),
        ("A", {}, "metadata"),
        ("A", {"left_bit": 0, "right_bit": 1, "combination": "00"}, "inconsistent"),
        ("A", {"left_bit": 0, "right_bit": 1, "combination": "01"}, "inconsistent"),
    ],
)
def test_visual_xor_rejects_invalid_private_answer_data(label, metadata, message):
    sample = Sample(response="A", response_length=2, label=label, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        get_visual_xor_reward(sample)


@pytest.mark.asyncio
async def test_visual_xor_routes_in_process(monkeypatch):
    RewardExecutor._instance = None
    monkeypatch.setattr(
        RewardExecutor,
        "_ensure_workers",
        lambda self: (_ for _ in ()).throw(AssertionError("visual XOR should stay local")),
    )
    args = SimpleNamespace(
        rm_type="visual_xor",
        custom_rm_path=None,
        reward_max_concurrency=8,
        reward_num_workers=2,
        reward_key="score",
    )

    reward = await async_rm(args, _sample("A"))

    assert reward["score"] == 1.0
    RewardExecutor._instance = None


def test_visual_xor_metrics_are_conditioned_by_combination():
    args = SimpleNamespace(reward_key="score")
    samples = [_sample("A", combination="00", label="A"), _sample("A", combination="01", label="B")]
    for sample in samples:
        sample.reward = get_visual_xor_reward(sample)

    metrics = compute_rollout_explicit_reward_metrics(args, samples)

    assert metrics["combo_00_accuracy/mean"] == 1.0
    assert metrics["combo_01_accuracy/mean"] == 0.0
    assert metrics["target_a_accuracy/mean"] == 1.0
    assert metrics["target_b_accuracy/mean"] == 0.0
