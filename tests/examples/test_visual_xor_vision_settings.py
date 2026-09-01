# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.visual_xor.compare_vision_settings import (
    CPU,
    CPU_AND_SGLANG_CACHE,
    CPU_CACHE,
    GPU,
    compare_vision_settings,
    write_vision_settings,
)


def _write_cycle(path, cycle, *, settings, step_time, responses=64, valid_action=1.0):
    with path.open("a") as output:
        if settings.device == "cpu":
            cache_hits = 32 if settings.cpu_cache else 0
            cache_misses = 0 if settings.cpu_cache else 32
            output.write(
                f"CPU vision metrics rollout_{cycle}: "
                + repr(
                    {
                        "vision_encoder/cache/hits_interval": cache_hits,
                        "vision_encoder/cache/misses_interval": cache_misses,
                        "vision_encoder/cache/evictions_interval": 0,
                        "vision_encoder/requests_interval": 32,
                        "vision_encoder/backend/encode_requests_interval": cache_misses,
                        "vision_encoder/backend/encoded_images_interval": cache_misses,
                        "vision_encoder/backend/encode_seconds_interval": 0.5 if cache_misses else 0.0,
                        "vision_encoder/features/unique_interval": 32,
                        "vision_encoder/backend/duplicate_encode_ratio_interval": 1.0 if cache_misses else 0.0,
                        "vision_encoder/backend/batch_size_mean_interval": 1.0 if cache_misses else 0.0,
                        "vision_encoder/backend/batch_size_p95_interval": 1.0 if cache_misses else 0.0,
                        "vision_encoder/backend/batch_size_max_interval": 1 if cache_misses else 0,
                        "vision_encoder/replicas/expected": 1,
                        "vision_encoder/replicas/observed": 1,
                        "vision_encoder/replicas/active": 1,
                        "vision_encoder/replicas/unobserved": 0,
                        "vision_encoder/replica/test/requests_interval": 32,
                        "vision_encoder/replica/test/process_cpu_seconds_total": 12.0 + cycle,
                        "vision_encoder/replica/test/process_cpu_utilization_percent_interval": 90.0,
                        "vision_encoder/replica/test/rss_bytes": 2_000_000_000,
                    }
                )
                + "\n"
            )
        output.write(
            f"relax.distributed.ray.rollout:47 perf {cycle}: "
            + repr(
                {
                    "perf/rollout_time": 2.0,
                    "perf_detail/rollout/sglang_parallel_samples/total": responses,
                    "perf_detail/rollout/http_request_body_bytes/total": 64_000,
                    "perf_detail/rollout/precomputed_to_list_time/total": 0.2,
                    "perf_detail/rollout/vision_service_round_trip_time/mean": (
                        0.3 if settings.device == "cpu" else 0.0
                    ),
                    "perf_detail/rollout/vision_service_round_trip_time/p95": 0.4 if settings.device == "cpu" else 0.0,
                    "perf_detail/rollout/sglang_vision_cache_id_only_hits/total": (
                        32 if settings.sglang_cache else 0
                    ),
                    "perf_detail/rollout/sglang_vision_cache_id_only_misses/total": 0,
                    "perf_detail/rollout/sglang_vision_cache_republish/total": 0,
                    "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total": 0,
                    "rollout/response_len/mean": 2.0,
                    "rollout/reward/mean": 0.75,
                    "rollout/action_a/mean": 0.5,
                    "rollout/action_b/mean": 0.5,
                    "rollout/valid_action/mean": valid_action,
                }
            )
            + "\n"
        )
        output.write(
            f"relax.backends.megatron.data:47 rollout {cycle}: "
            + repr({"rollout/response_lengths": 2.0, "rollout/total_lengths": 116.0})
            + "\n"
        )
        output.write(
            f"relax.utils.training.train_metric_utils:47 perf {cycle}: "
            + repr(
                {
                    "perf/train_wait_time": 0.1,
                    "perf/train_time": step_time - 0.1,
                    "perf/actor_train_time": 4.0,
                    "perf/update_weights_fully_async_time": step_time - 4.1,
                    "perf/step_time": step_time,
                    "perf/wait_time_ratio": 0.1 / step_time,
                }
            )
            + "\n"
        )


def _repeated_logs(tmp_path, *, cycles=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings_order = (GPU, CPU, CPU_CACHE, CPU_AND_SGLANG_CACHE)
    times = {GPU: 10.0, CPU: 9.0, CPU_CACHE: 8.0, CPU_AND_SGLANG_CACHE: 7.0}
    runs = {}
    for repeat_index in range(4):
        repeat = f"repeat_{repeat_index + 1}"
        runs[repeat] = {}
        for settings in settings_order:
            slug = f"{settings.device}-{int(settings.cpu_cache)}-{int(settings.sglang_cache)}"
            path = tmp_path / f"{repeat}-{slug}.log"
            for cycle in range(cycles):
                _write_cycle(path, cycle, settings=settings, step_time=times[settings])
            runs[repeat][settings] = path
    return runs


def _comparison(result, baseline, candidate):
    return next(
        comparison
        for comparison in result["comparisons"]
        if comparison["baseline"] == baseline and comparison["candidate"] == candidate
    )


def test_comparison_uses_repeated_runs_as_statistical_units(tmp_path):
    result = compare_vision_settings(
        _repeated_logs(tmp_path), first_steady_cycle=2, expected_steady_cycles=2
    )

    assert result["repeat_count"] == 4
    first_repeat = result["repeats"]["repeat_1"]["runs"]
    gpu = next(run["metrics"] for run in first_repeat if run["settings"]["device"] == "gpu")
    assert gpu["training_cycles_per_hour"] == pytest.approx(360.0)
    assert gpu["optimizer_steps_per_hour"] == pytest.approx(720.0)
    gpu_to_cpu = _comparison(
        result,
        {"device": "gpu", "cpu_cache": False, "sglang_cache": False},
        {"device": "cpu", "cpu_cache": False, "sglang_cache": False},
    )
    assert gpu_to_cpu["speedup"]["mean"] == pytest.approx(0.1)
    assert gpu_to_cpu["speedup"]["repeat_count"] == 4


def test_comparison_requires_vision_metrics_only_for_cpu_runs(tmp_path):
    runs = _repeated_logs(tmp_path)
    assert compare_vision_settings(runs, first_steady_cycle=2, expected_steady_cycles=2)
    cpu_log = runs["repeat_1"][CPU]
    cpu_log.write_text(
        "\n".join(line for line in cpu_log.read_text().splitlines() if "CPU vision metrics" not in line)
    )

    with pytest.raises(ValueError, match="missing_vision"):
        compare_vision_settings(runs, first_steady_cycle=2, expected_steady_cycles=2)


def test_comparison_rejects_response_or_valid_action_mismatch(tmp_path):
    runs = _repeated_logs(tmp_path)
    path = runs["repeat_3"][CPU_AND_SGLANG_CACHE]
    path.write_text(
        path.read_text().replace(
            "'perf_detail/rollout/sglang_parallel_samples/total': 64",
            "'perf_detail/rollout/sglang_parallel_samples/total': 63",
            1,
        )
    )
    with pytest.raises(ValueError, match="expected 64 responses"):
        compare_vision_settings(runs, first_steady_cycle=0, expected_steady_cycles=4)

    runs = _repeated_logs(tmp_path / "valid")
    path = runs["repeat_2"][GPU]
    path.write_text(
        path.read_text().replace("'rollout/valid_action/mean': 1.0", "'rollout/valid_action/mean': 0.9", 1)
    )
    with pytest.raises(ValueError, match="valid-action rate"):
        compare_vision_settings(runs, first_steady_cycle=0, expected_steady_cycles=4)


def test_comparison_artifact_is_versioned(tmp_path):
    result = compare_vision_settings(
        _repeated_logs(tmp_path), first_steady_cycle=2, expected_steady_cycles=2
    )
    output = tmp_path / "vision-settings.json"
    artifact = write_vision_settings(output, result)

    assert artifact["schema_version"] == 2
    assert json.loads(output.read_text()) == artifact
