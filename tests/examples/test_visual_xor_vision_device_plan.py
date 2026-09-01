# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.visual_xor.build_vision_device_plan import build_plan
from relax.engine.rollout.vision_device import load_vision_device_plan


def _write_inputs(tmp_path):
    measurements = tmp_path / "measurements.jsonl"
    rows = []
    for device, intercept, per_image, per_token, per_byte in (
        ("gpu", 0.2, 0.01, 0.001, 0.000001),
        ("cpu", 0.3, 0.02, 0.002, 0.000002),
    ):
        for image_count, visual_tokens, feature_bytes in (
            (1, 100, 800),
            (2, 100, 800),
            (1, 200, 800),
            (1, 100, 1600),
            (4, 400, 3200),
        ):
            seconds = intercept + per_image * image_count + per_token * visual_tokens + per_byte * feature_bytes
            rows.append(
                json.dumps(
                    {
                        "device": device,
                        "image_count": image_count,
                        "visual_tokens": visual_tokens,
                        "feature_bytes": feature_bytes,
                        "seconds": seconds,
                    }
                )
            )
    measurements.write_text("\n".join(rows) + "\n")
    cycles = tmp_path / "cycles.json"
    cycles.write_text(
        json.dumps(
            {
                "0": {
                    "image_count": 1,
                    "visual_tokens": 100,
                    "feature_bytes": 800,
                    "overlap_seconds": 0.4,
                }
            }
        )
    )
    return measurements, cycles


def test_build_plan_recovers_nonnegative_times_and_cycle_workload(tmp_path):
    measurements, cycles = _write_inputs(tmp_path)

    plan_payload = build_plan(
        measurements=measurements,
        cycles=cycles,
        minimum_gap=0.05,
        initial_device="gpu",
    )

    assert plan_payload["schema_version"] == 2
    assert plan_payload["time_models"]["gpu"]["intercept_seconds"] == pytest.approx(0.2)
    assert plan_payload["time_models"]["cpu"]["seconds_per_visual_token"] == pytest.approx(0.002)
    assert plan_payload["timing_model_fit"]["gpu"]["r_squared"] == pytest.approx(1.0)
    assert plan_payload["cycles"]["0"]["overlap_seconds"] == 0.4
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_payload))
    assert load_vision_device_plan(path).choose(0).device == "cpu"


def test_build_plan_requires_enough_rows_for_each_device(tmp_path):
    measurements, cycles = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in measurements.read_text().splitlines()]
    measurements.write_text("\n".join(json.dumps(row) for row in rows if row["device"] == "gpu") + "\n")

    with pytest.raises(ValueError, match="Device 'cpu' requires at least"):
        build_plan(measurements=measurements, cycles=cycles, minimum_gap=0.05, initial_device="gpu")


def test_build_plan_drops_collinear_feature_bytes(tmp_path):
    measurements = tmp_path / "measurements.jsonl"
    rows = []
    for device, scale in (("gpu", 1.0), ("cpu", 1.5)):
        for image_count, visual_tokens in ((1, 64), (2, 128), (3, 192), (2, 256)):
            feature_bytes = 16 * visual_tokens + 12 * image_count
            rows.append(
                {
                    "device": device,
                    "image_count": image_count,
                    "visual_tokens": visual_tokens,
                    "feature_bytes": feature_bytes,
                    "seconds": scale * (0.1 + 0.01 * image_count + 0.001 * visual_tokens),
                }
            )
    measurements.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cycles = tmp_path / "cycles.json"
    cycles.write_text(
        json.dumps(
            {
                "0": {
                    "image_count": 1,
                    "visual_tokens": 64,
                    "feature_bytes": 1036,
                    "overlap_seconds": 0.0,
                }
            }
        )
    )

    plan = build_plan(measurements=measurements, cycles=cycles, minimum_gap=0.05, initial_device="gpu")

    assert plan["timing_model_fit"]["gpu"]["design_rank"] == 3
    assert plan["timing_model_fit"]["gpu"]["dropped_collinear_predictors"] == ["seconds_per_feature_byte"]
    assert plan["time_models"]["gpu"]["seconds_per_feature_byte"] == 0.0
    assert plan["timing_model_fit"]["gpu"]["r_squared"] == pytest.approx(1.0)
