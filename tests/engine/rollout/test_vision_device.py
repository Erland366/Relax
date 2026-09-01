# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from asyncio import gather, run, sleep
from threading import Lock
from types import SimpleNamespace

import pytest

from relax.engine.rollout.vision_device import load_vision_device_plan


def _write_plan(tmp_path, **overrides):
    payload = {
        "schema_version": 2,
        "initial_device": "gpu",
        "minimum_gap": 0.05,
        "time_models": {
            "gpu": {"seconds_per_visual_token": 0.001},
            "cpu": {"seconds_per_visual_token": 0.002},
        },
        "cycles": {
            "0": {
                "image_count": 1,
                "visual_tokens": 100,
                "feature_bytes": 800,
                "overlap_seconds": 0.15,
            },
            "1": {
                "image_count": 1,
                "visual_tokens": 100,
                "feature_bytes": 800,
                "overlap_seconds": 0.0,
            },
            "2": {
                "image_count": 1,
                "visual_tokens": 100,
                "feature_bytes": 800,
                "overlap_seconds": 0.096,
            },
        },
    }
    payload.update(overrides)
    path = tmp_path / "vision-device-plan.json"
    path.write_text(json.dumps(payload))
    return path


def test_vision_device_plan_selects_cpu_gpu_and_keeps_previous_device(tmp_path):
    plan = load_vision_device_plan(_write_plan(tmp_path))

    cpu = plan.choose(0)
    gpu = plan.choose(1)
    unchanged = plan.choose(2)

    assert cpu.device == "cpu"
    assert cpu.reason == "cpu_predicted_faster"
    assert cpu.predicted_gpu_seconds == pytest.approx(0.1)
    assert cpu.predicted_cpu_seconds == pytest.approx(0.2)
    assert cpu.predicted_cpu_seconds_after_overlap == pytest.approx(0.05)
    assert cpu.predicted_gap == pytest.approx(0.5)
    assert gpu.device == "gpu"
    assert gpu.reason == "gpu_predicted_faster"
    assert unchanged.device == "gpu"
    assert unchanged.reason == "keep_previous_device"
    assert unchanged.to_metrics()["vision_device/gpu"] == 1


def test_vision_device_choices_are_independent_of_request_order(tmp_path):
    plan = load_vision_device_plan(_write_plan(tmp_path))

    unchanged = plan.choose(2)
    cpu = plan.choose(0)
    gpu = plan.choose(1)

    assert unchanged.device == "gpu"
    assert unchanged.reason == "keep_previous_device"
    assert cpu.device == "cpu"
    assert gpu.device == "gpu"


def test_overlapping_rollout_contexts_keep_distinct_vision_devices(tmp_path):
    from relax.engine.rollout.sglang_rollout import (
        _choose_vision_device_for_cycle,
        _collect_vision_device_metrics,
        _uses_cpu_vision,
    )

    state = SimpleNamespace(
        vision_encoder=object(),
        vision_device_plan=load_vision_device_plan(_write_plan(tmp_path)),
        vision_device_choices={},
        vision_device_lock=Lock(),
    )

    async def select(rollout_id, delay):
        _choose_vision_device_for_cycle(state, rollout_id)
        await sleep(delay)
        return _uses_cpu_vision(state), _collect_vision_device_metrics(state)

    async def overlap():
        return await gather(select(0, 0.01), select(1, 0.0))

    (uses_cpu, cpu_metrics), (uses_gpu, gpu_metrics) = run(overlap())

    assert uses_cpu is True
    assert cpu_metrics["vision_device/cpu"] == 1
    assert uses_gpu is False
    assert gpu_metrics["vision_device/gpu"] == 1


def test_vision_device_plan_fails_when_cycle_is_missing(tmp_path):
    plan = load_vision_device_plan(_write_plan(tmp_path))

    with pytest.raises(KeyError, match="rollout_id=9.*available rollout IDs"):
        plan.choose(9)


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"schema_version": 1}, "schema_version"),
        ({"cycles": {}}, "at least one rollout cycle"),
        ({"initial_device": "automatic"}, "initial_device"),
        (
            {
                "cycles": {
                    "0": {
                        "image_count": 1,
                        "visual_tokens": -1,
                        "feature_bytes": 800,
                        "overlap_seconds": 0.0,
                    }
                }
            },
            "visual_tokens.*non-negative integer",
        ),
    ],
)
def test_vision_device_plan_rejects_invalid_input(tmp_path, overrides, expected_error):
    path = _write_plan(tmp_path, **overrides)

    with pytest.raises(ValueError, match=expected_error):
        load_vision_device_plan(path)


def test_cli_minimum_gap_overrides_plan_value(tmp_path):
    plan = load_vision_device_plan(_write_plan(tmp_path), minimum_gap=0.01)

    choice = plan.choose(2)

    assert choice.device == "gpu"
    assert choice.reason == "gpu_predicted_faster"
    assert choice.minimum_gap == 0.01
