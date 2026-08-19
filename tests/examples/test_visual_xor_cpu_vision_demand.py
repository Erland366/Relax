# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest


def _write_cycle(log_path, cycle, *, unique_images, requests, step_time, wait_ratio):
    with log_path.open("a") as output:
        output.write(
            f"CPU vision metrics rollout_{cycle}: "
            + repr(
                {
                    "vision_encoder/features/unique_interval": unique_images,
                    "vision_encoder/requests_interval": requests,
                    "vision_encoder/backend/encoded_images_interval": requests,
                }
            )
            + "\n"
        )
        output.write(
            f"relax.utils.training.train_metric_utils:47 perf {cycle}: "
            + repr(
                {
                    "perf/step_time": step_time,
                    "perf/wait_time_ratio": wait_ratio,
                }
            )
            + "\n"
        )


def test_analyze_e1_demand_uses_paired_steady_cycles_and_reports_accounting(tmp_path):
    from examples.visual_xor.analyze_cpu_vision_demand import analyze_e1_demand, write_demand_artifact

    d2_log = tmp_path / "e1_d2.console.log"
    d3_log = tmp_path / "e1_d3.console.log"
    for cycle in range(4):
        _write_cycle(
            d2_log,
            cycle,
            unique_images=32,
            requests=32,
            step_time=8.0 if cycle >= 2 else 80.0,
            wait_ratio=0.04,
        )
        _write_cycle(
            d3_log,
            cycle,
            unique_images=64,
            requests=65 if cycle == 3 else 64,
            step_time=10.0,
            wait_ratio=0.01,
        )

    result = analyze_e1_demand(
        {"D2": d2_log, "D3": d3_log},
        first_steady_cycle=2,
        expected_steady_cycles=2,
        saturation_profile="D2",
        actor_wait_threshold=0.05,
    )

    assert result["peak_unique_images_per_second"] == pytest.approx(6.4)
    assert result["capacity_target_images_per_second"] == pytest.approx(8.0)
    assert result["saturation_profile_reached_actor_wait_gate"] is False
    assert result["profiles"]["D2"]["peak_unique_images_per_second"] == pytest.approx(4.0)
    assert result["profiles"]["D2"]["exact_request_accounting"] is True
    assert result["profiles"]["D3"]["exact_request_accounting"] is False
    assert result["profiles"]["D3"]["extra_requests"] == 1

    output_path = tmp_path / "demand.json"
    artifact = write_demand_artifact(output_path, result)
    assert artifact["schema_version"] == 1
    assert json.loads(output_path.read_text()) == artifact


def test_analyze_e1_demand_fails_when_a_steady_actor_cycle_is_missing(tmp_path):
    from examples.visual_xor.analyze_cpu_vision_demand import analyze_e1_demand

    log_path = tmp_path / "e1_d2.console.log"
    _write_cycle(log_path, 2, unique_images=32, requests=32, step_time=10.0, wait_ratio=0.01)
    with log_path.open("a") as output:
        output.write(
            "CPU vision metrics rollout_3: "
            + repr(
                {
                    "vision_encoder/features/unique_interval": 32,
                    "vision_encoder/requests_interval": 32,
                    "vision_encoder/backend/encoded_images_interval": 32,
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="paired steady cycles"):
        analyze_e1_demand(
            {"D2": log_path},
            first_steady_cycle=2,
            expected_steady_cycles=2,
            saturation_profile="D2",
            actor_wait_threshold=0.05,
        )
