# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.visual_xor.compare_vision_device_choices import compare_vision_device_choices


def _write_log(path, *, run, training_cycle_seconds):
    lines = []
    devices = ("cpu", "gpu")
    for cycle, seconds in enumerate(training_cycle_seconds):
        rollout_metrics = {
            "perf_detail/rollout/sglang_parallel_samples/total": 64,
            "perf_detail/rollout/sglang_generation_requests/total": 32,
            "rollout/valid_action/mean": 1.0,
            "perf/rollout_time": seconds / 2,
            "rollout/reward/mean": 0.75,
        }
        if run == "automatic":
            device = devices[cycle % 2]
            rollout_metrics[f"vision_device/{device}"] = 1
            rollout_metrics[f"vision_device/{'gpu' if device == 'cpu' else 'cpu'}"] = 0
            choice = {
                "schema_version": 2,
                "rollout_id": cycle,
                "device": device,
                "reason": "test",
                "workload": {
                    "image_count": 1,
                    "visual_tokens": 100,
                    "feature_bytes": 800,
                    "overlap_seconds": 0.1,
                },
                "predicted_gpu_seconds": 1.0,
                "predicted_cpu_seconds": 1.0,
                "predicted_cpu_seconds_after_overlap": 0.9,
                "predicted_gap": 0.1,
                "minimum_gap": 0.05,
            }
            lines.append(f"VISION_DEVICE_CHOICE {json.dumps(choice)}")
        lines.append(f"relax.distributed.ray.rollout:1 perf {cycle}: {rollout_metrics!r}")
        lines.append(
            f"relax.utils.training.train_metric_utils:1 perf {cycle}: "
            f"{{'perf/step_time': {seconds!r}}}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(tmp_path, *, plan_log_modifier=None):
    repeats = []
    for repeat_index in range(3):
        gpu = tmp_path / f"repeat_{repeat_index}_gpu.log"
        cpu = tmp_path / f"repeat_{repeat_index}_cpu.log"
        plan = tmp_path / f"repeat_{repeat_index}_plan.log"
        _write_log(gpu, run="gpu", training_cycle_seconds=[10.0, 8.0])
        _write_log(cpu, run="cpu", training_cycle_seconds=[8.0, 10.0])
        _write_log(plan, run="automatic", training_cycle_seconds=[8.1, 8.1])
        if plan_log_modifier is not None:
            plan_log_modifier(plan)
        repeats.append(
            {
                "name": f"repeat_{repeat_index}",
                "runs": {"gpu": str(gpu), "cpu": str(cpu), "automatic": str(plan)},
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "measured_cycles": [0, 1],
                "expected_responses_per_cycle": 64,
                "expected_generation_requests_per_cycle": 32,
                "require_both_devices": True,
                "repeats": repeats,
            }
        )
    )
    return manifest


def test_comparison_reports_matches_and_extra_time(tmp_path):
    result = compare_vision_device_choices(_write_manifest(tmp_path))

    assert result["summary"]["match_rate"] == 1.0
    assert result["summary"]["total_cycles"] == 6
    assert result["summary"]["cycles_over_minimum_gap"] == 6
    assert result["summary"]["device_matches"] == 6
    assert result["summary"]["mean_extra_time"] == pytest.approx(0.0125)
    assert result["summary"]["acceptance"] == {
        "match_rate": True,
        "extra_time": True,
        "coverage": True,
    }


def test_comparison_rejects_missing_device_choice(tmp_path):
    def remove_first_choice(path):
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")

    with pytest.raises(ValueError, match=r"choice mismatch.*missing=\[0\]"):
        compare_vision_device_choices(_write_manifest(tmp_path, plan_log_modifier=remove_first_choice))
