# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest


def _write_cycle(
    log_path,
    cycle,
    *,
    cache_hits,
    cache_misses,
    encoded_images,
    request_bytes,
    id_hits,
    inline_requests,
    rollout_time,
    rtt_p95,
    wait_ratio,
):
    with log_path.open("a") as output:
        output.write(
            f"CPU vision metrics rollout_{cycle}: "
            + repr(
                {
                    "vision_encoder/cache/hits_interval": cache_hits,
                    "vision_encoder/cache/misses_interval": cache_misses,
                    "vision_encoder/requests_interval": cache_hits + cache_misses,
                    "vision_encoder/backend/encoded_images_interval": encoded_images,
                    "vision_encoder/features/unique_interval": cache_hits + cache_misses,
                    "vision_encoder/backend/duplicate_encode_ratio_interval": 1.0,
                }
            )
            + "\n"
        )
        output.write(
            f"relax.distributed.ray.rollout:47 perf {cycle}: "
            + repr(
                {
                    "perf/rollout_time": rollout_time,
                    "perf_detail/rollout/http_request_body_bytes/total": request_bytes,
                    "perf_detail/rollout/precomputed_to_list_time/total": 0.2,
                    "perf_detail/rollout/vision_service_round_trip_time/mean": 0.3,
                    "perf_detail/rollout/vision_service_round_trip_time/p95": rtt_p95,
                    "perf_detail/rollout/sglang_vision_cache_id_only_hits/total": id_hits,
                    "perf_detail/rollout/sglang_vision_cache_id_only_misses/total": 0,
                    "perf_detail/rollout/sglang_vision_cache_republish/total": 0,
                    "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total": inline_requests,
                    "rollout/valid_action/mean": 1.0,
                }
            )
            + "\n"
        )
        output.write(
            f"relax.utils.training.train_metric_utils:47 perf {cycle}: "
            + repr({"perf/step_time": rollout_time + 5.0, "perf/wait_time_ratio": wait_ratio})
            + "\n"
        )


def test_measure_vision_cache_reuse_pairs_steady_cycles_and_compares_cache_settings(tmp_path):
    from examples.visual_xor.measure_vision_cache_reuse import (
        measure_vision_cache_reuse,
        write_vision_cache_reuse_artifact,
    )

    cpu_cache_log = tmp_path / "cpu_cache.log"
    cpu_and_sglang_cache_log = tmp_path / "cpu_and_sglang_cache.log"
    for cycle in range(4):
        steady = cycle >= 2
        _write_cycle(
            cpu_cache_log,
            cycle,
            cache_hits=32 if steady else 0,
            cache_misses=0 if steady else 32,
            encoded_images=0 if steady else 32,
            request_bytes=22_400_000,
            id_hits=0,
            inline_requests=0,
            rollout_time=3.0,
            rtt_p95=0.5,
            wait_ratio=0.01,
        )
        _write_cycle(
            cpu_and_sglang_cache_log,
            cycle,
            cache_hits=32 if steady else 0,
            cache_misses=0 if steady else 32,
            encoded_images=0 if steady else 32,
            request_bytes=64_000,
            id_hits=32 if steady else 0,
            inline_requests=0 if steady else 32,
            rollout_time=2.8,
            rtt_p95=0.52,
            wait_ratio=0.01,
        )

    result = measure_vision_cache_reuse(
        {"cpu_cache_only": cpu_cache_log, "cpu_and_sglang_cache": cpu_and_sglang_cache_log},
        first_steady_cycle=2,
        expected_steady_cycles=2,
    )

    assert result["profiles"]["cpu_cache_only"]["cpu_cache_hit_rate"] == 1.0
    assert result["profiles"]["cpu_and_sglang_cache"]["sglang_id_hit_rate"] == 1.0
    assert result["request_body_byte_reduction"] == pytest.approx(1 - 64_000 / 22_400_000)
    assert result["rollout_time_reduction"] == pytest.approx(1 - 2.8 / 3.0)
    assert result["p95_service_rtt_regression"] == pytest.approx(0.04)
    assert result["all_valid_actions"] is True
    assert result["no_republish"] is True

    output_path = tmp_path / "cache.json"
    artifact = write_vision_cache_reuse_artifact(output_path, result)
    assert artifact["schema_version"] == 1
    assert json.loads(output_path.read_text()) == artifact


def test_measure_vision_cache_reuse_rejects_missing_rollout_cycle(tmp_path):
    from examples.visual_xor.measure_vision_cache_reuse import measure_vision_cache_reuse

    log_path = tmp_path / "incomplete.log"
    _write_cycle(
        log_path,
        2,
        cache_hits=32,
        cache_misses=0,
        encoded_images=0,
        request_bytes=64_000,
        id_hits=32,
        inline_requests=0,
        rollout_time=2.8,
        rtt_p95=0.52,
        wait_ratio=0.01,
    )

    with pytest.raises(ValueError, match="paired steady cycles"):
        measure_vision_cache_reuse(
            {"cpu_cache_only": log_path, "cpu_and_sglang_cache": log_path},
            first_steady_cycle=2,
            expected_steady_cycles=2,
        )
