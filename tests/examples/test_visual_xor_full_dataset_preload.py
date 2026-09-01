# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.visual_xor.measure_full_dataset_preload import (
    measure_full_dataset_preload,
    write_full_dataset_preload_artifact,
)


def _write_cycle(path, cycle, *, preload, step_time, overrides=None):
    preloaded = preload == "full_dataset"
    metrics = {
        "requests": 32,
        "hits": 32 if preloaded else 0,
        "misses": 0 if preloaded else 32,
        "evictions": 0,
        "encodes": 0 if preloaded else 32,
        "encoded_images": 0 if preloaded else 32,
        "id_hits": 32 if preloaded else 0,
        "id_misses": 0,
        "inline_publishes": 0,
        "republishes": 0,
    }
    if overrides:
        metrics.update(overrides)

    with path.open("a") as output:
        output.write(
            f"CPU vision metrics rollout_{cycle}: "
            + repr(
                {
                    "vision_encoder/cache/hits_interval": metrics["hits"],
                    "vision_encoder/cache/misses_interval": metrics["misses"],
                    "vision_encoder/cache/evictions_interval": metrics["evictions"],
                    "vision_encoder/requests_interval": metrics["requests"],
                    "vision_encoder/backend/encode_requests_interval": metrics["encodes"],
                    "vision_encoder/backend/encoded_images_interval": metrics["encoded_images"],
                    "vision_encoder/backend/encode_seconds_interval": 0.0 if preloaded else 0.5,
                    "vision_encoder/features/unique_interval": 32,
                    "vision_encoder/backend/duplicate_encode_ratio_interval": 0.0 if preloaded else 1.0,
                    "vision_encoder/backend/batch_size_mean_interval": 0.0 if preloaded else 1.0,
                    "vision_encoder/backend/batch_size_p95_interval": 0.0 if preloaded else 1.0,
                    "vision_encoder/backend/batch_size_max_interval": 0 if preloaded else 1,
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
                    "perf_detail/rollout/sglang_parallel_samples/total": 64,
                    "perf_detail/rollout/http_request_body_bytes/total": 64_000,
                    "perf_detail/rollout/precomputed_to_list_time/total": 0.2,
                    "perf_detail/rollout/vision_service_round_trip_time/mean": 0.3,
                    "perf_detail/rollout/vision_service_round_trip_time/p95": 0.4,
                    "perf_detail/rollout/sglang_vision_cache_id_only_hits/total": metrics["id_hits"],
                    "perf_detail/rollout/sglang_vision_cache_id_only_misses/total": metrics["id_misses"],
                    "perf_detail/rollout/sglang_vision_cache_republish/total": metrics["republishes"],
                    "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total": metrics[
                        "inline_publishes"
                    ],
                    "rollout/response_len/mean": 2.0,
                    "rollout/reward/mean": 0.75,
                    "rollout/action_a/mean": 0.5,
                    "rollout/action_b/mean": 0.5,
                    "rollout/valid_action/mean": 1.0,
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
        for optimizer_step in range(2):
            output.write(
                f"train_one_step rollout={cycle} step={optimizer_step}: "
                "finished optimizer.step (update_successful=True)\n"
            )


def _write_completion(path):
    with path.open("a") as output:
        output.write("Actor training completed step 21/22\n")
        output.write("All training steps finished\n")
        output.write("All rollouts finished\n")
        output.write("Service task completed successfully\n")


def _write_preload(path, *, wall_seconds=44.0, schema_version=2):
    feature_ids = [f"feature-{index}" for index in range(64)]
    dataset_state = {
        "sample_offset": 0,
        "epoch_id": 0,
        "sample_group_index": 0,
        "sample_index": 0,
        "dataset_size": 64,
        "dataset_fingerprint": "a" * 64,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "wall_seconds": wall_seconds,
                "dataset_samples": 64,
                "unique_features": 64,
                "duplicate_features": 0,
                "feature_ids": feature_ids,
                "features": [
                    {
                        "feature_id": feature_id,
                        "vision_revision": "qwen3-vl-vision-revision",
                        "feature_schema_version": 1,
                        "image_grid_thw": [[1, 2, 2]],
                        "image_tokens": 4,
                        "feature_bytes": 16_384,
                    }
                    for feature_id in feature_ids
                ],
                "work_counters": {
                    "generation_requests": 0,
                    "generated_samples": 0,
                    "training_samples": 0,
                },
                "dataset_state": {
                    "before": dataset_state,
                    "after": dict(dataset_state),
                    "unchanged": True,
                },
                "cpu_cache": {"entries": 64, "evictions": 0},
                "sglang_cache": {"entries": 64, "stores": 64, "evictions": 0},
            }
        )
        + "\n"
    )


def _study(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    orders = {
        "repeat_1": ("none", "full_dataset"),
        "repeat_2": ("full_dataset", "none"),
        "repeat_3": ("none", "full_dataset"),
        "repeat_4": ("full_dataset", "none"),
    }
    runs = {}
    preload_artifacts = {}
    for repeat, settings in orders.items():
        runs[repeat] = {}
        for preload in settings:
            path = tmp_path / f"{repeat}-{preload}.log"
            for cycle in range(22):
                _write_cycle(
                    path,
                    cycle,
                    preload=preload,
                    step_time=10.0 if preload == "none" else 8.0,
                )
            _write_completion(path)
            runs[repeat][preload] = path
        preload_path = tmp_path / f"{repeat}-preload.json"
        _write_preload(preload_path)
        preload_artifacts[repeat] = preload_path
    return runs, preload_artifacts


def test_measurement_reports_training_only_and_preload_inclusive_speedup(tmp_path):
    runs, preload_artifacts = _study(tmp_path)
    result = measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)

    assert result["repeat_count"] == 4
    first = result["repeats"]["repeat_1"]
    assert first["runs"]["full_dataset"]["measured_training_seconds"] == pytest.approx(176.0)
    assert first["runs"]["full_dataset"]["preload_seconds"] == pytest.approx(44.0)
    assert first["runs"]["full_dataset"]["training_cycle_seconds_including_preload"] == pytest.approx(10.0)
    assert first["training_only_speedup"] == pytest.approx(0.2)
    assert first["speedup_including_preload"] == pytest.approx(0.0)
    assert first["cycles_to_recover_preload_time"] == 22

    output = tmp_path / "full-dataset-preload.json"
    artifact = write_full_dataset_preload_artifact(output, result)
    assert artifact["schema_version"] == 2
    assert json.loads(output.read_text()) == artifact


@pytest.mark.parametrize(
    ("preload", "cycle", "overrides", "error_pattern"),
    [
        ("full_dataset", 0, {"hits": 0, "misses": 32, "encodes": 32}, r"cycle 0.*32 CPU cache hits"),
        ("full_dataset", 17, {"id_hits": 31, "id_misses": 1}, r"cycle 17.*32 SGLang cache ID hits"),
        ("none", 9, {"requests": 31, "misses": 31, "encodes": 31}, r"cycle 9.*32 CPU cache requests"),
    ],
)
def test_measurement_requires_exact_cache_accounting(tmp_path, preload, cycle, overrides, error_pattern):
    runs, preload_artifacts = _study(tmp_path)
    broken = runs["repeat_1"][preload]
    broken.unlink()
    for current_cycle in range(22):
        _write_cycle(
            broken,
            current_cycle,
            preload=preload,
            step_time=10.0 if preload == "none" else 8.0,
            overrides=overrides if current_cycle == cycle else None,
        )
    _write_completion(broken)

    with pytest.raises(ValueError, match=error_pattern):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "error_pattern"),
    [
        (("dataset_samples",), 63, r"dataset_samples.*64"),
        (("cpu_cache", "entries"), 63, r"CPU cache entries=64"),
        (("cpu_cache", "evictions"), 1, r"CPU cache evictions.*zero"),
        (("sglang_cache", "entries"), 63, r"SGLang cache entries=64"),
        (("sglang_cache", "stores"), 63, r"SGLang cache stores=64"),
    ],
)
def test_measurement_rejects_incomplete_preload(tmp_path, field_path, invalid_value, error_pattern):
    runs, preload_artifacts = _study(tmp_path)
    artifact = json.loads(preload_artifacts["repeat_1"].read_text())
    target = artifact
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = invalid_value
    preload_artifacts["repeat_1"].write_text(json.dumps(artifact) + "\n")

    with pytest.raises(ValueError, match=error_pattern):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)


def test_measurement_requires_unchanged_dataset_state_and_no_work(tmp_path):
    runs, preload_artifacts = _study(tmp_path)
    artifact = json.loads(preload_artifacts["repeat_1"].read_text())
    artifact["dataset_state"]["after"]["sample_index"] = 1
    preload_artifacts["repeat_1"].write_text(json.dumps(artifact) + "\n")
    with pytest.raises(ValueError, match=r"dataset_state.*unchanged.*before.*after"):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)

    runs, preload_artifacts = _study(tmp_path / "work")
    artifact = json.loads(preload_artifacts["repeat_1"].read_text())
    artifact["work_counters"]["generated_samples"] = 1
    preload_artifacts["repeat_1"].write_text(json.dumps(artifact) + "\n")
    with pytest.raises(ValueError, match=r"work_counters generated_samples.*zero"):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)


def test_measurement_requires_complete_lifecycle(tmp_path):
    runs, preload_artifacts = _study(tmp_path)
    broken = runs["repeat_2"]["full_dataset"]
    broken.write_text(
        "\n".join(
            line for line in broken.read_text().splitlines() if line != "Service task completed successfully"
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="service completion"):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)


def test_measurement_requires_four_complete_repeats_and_schema_two(tmp_path):
    runs, preload_artifacts = _study(tmp_path)
    del runs["repeat_4"]
    del preload_artifacts["repeat_4"]
    with pytest.raises(ValueError, match="exactly four complete repeats"):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)

    runs, preload_artifacts = _study(tmp_path / "schema")
    _write_preload(preload_artifacts["repeat_4"], schema_version=1)
    with pytest.raises(ValueError, match=r"repeat_4.*schema_version 2"):
        measure_full_dataset_preload(runs, preload_artifacts, expected_cycles=22)
