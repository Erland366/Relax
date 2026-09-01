# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze the live CPU-vision CPU cache and SGLang transport-cache ablation."""

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_VISION_METRICS = re.compile(r"CPU vision metrics rollout_(\d+): (\{.*\})$")
_ROLLOUT_METRICS = re.compile(r"relax\.distributed\.ray\.rollout:\d+ perf (\d+): (\{.*\})$")
_TRAINING_METRICS = re.compile(r"relax\.utils\.training\.train_metric_utils:\d+ perf (\d+): (\{.*\})$")
_REQUIRED_PROFILES = ("cpu_cache_only", "cpu_and_sglang_cache")


def _parse_metrics_log(log_path: str | Path) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"vision cache reuse console log does not exist: {path}")

    vision_cycles: dict[int, dict] = {}
    rollout_cycles: dict[int, dict] = {}
    training_cycles: dict[int, dict] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        if match := _VISION_METRICS.search(line):
            vision_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
        elif match := _ROLLOUT_METRICS.search(line):
            rollout_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
        elif match := _TRAINING_METRICS.search(line):
            training_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
    return vision_cycles, rollout_cycles, training_cycles


def _finite_nonnegative(metrics: Mapping[str, object], key: str) -> float:
    value = float(metrics[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"metric {key} must be finite and non-negative, got {value}")
    return value


def _analyze_profile(
    log_path: str | Path,
    *,
    first_steady_cycle: int,
    expected_steady_cycles: int,
) -> dict[str, object]:
    vision_cycles, rollout_cycles, training_cycles = _parse_metrics_log(log_path)
    expected_cycles = list(range(first_steady_cycle, first_steady_cycle + expected_steady_cycles))
    paired_cycles = [
        cycle
        for cycle in expected_cycles
        if cycle in vision_cycles and cycle in rollout_cycles and cycle in training_cycles
    ]
    if paired_cycles != expected_cycles:
        raise ValueError(
            f"vision cache reuse log {log_path} does not contain all paired steady cycles; "
            f"missing_vision={sorted(set(expected_cycles) - vision_cycles.keys())}, "
            f"missing_rollout={sorted(set(expected_cycles) - rollout_cycles.keys())}, "
            f"missing_training={sorted(set(expected_cycles) - training_cycles.keys())}"
        )

    results = []
    for cycle in expected_cycles:
        vision = vision_cycles[cycle]
        rollout = rollout_cycles[cycle]
        training = training_cycles[cycle]
        hits = int(vision["vision_encoder/cache/hits_interval"])
        misses = int(vision["vision_encoder/cache/misses_interval"])
        requests = int(vision["vision_encoder/requests_interval"])
        encoded_images = int(vision["vision_encoder/backend/encoded_images_interval"])
        if min(hits, misses, requests, encoded_images) < 0 or hits + misses != requests:
            raise ValueError(f"vision cache reuse cycle {cycle} in {log_path} has inconsistent CPU cache counters")
        results.append(
            {
                "cycle": cycle,
                "cpu_cache_hits": hits,
                "cpu_cache_misses": misses,
                "cpu_cache_requests": requests,
                "encoded_images": encoded_images,
                "request_body_bytes": int(
                    _finite_nonnegative(rollout, "perf_detail/rollout/http_request_body_bytes/total")
                ),
                "serialization_seconds": _finite_nonnegative(
                    rollout, "perf_detail/rollout/precomputed_to_list_time/total"
                ),
                "service_rtt_mean_seconds": _finite_nonnegative(
                    rollout, "perf_detail/rollout/vision_service_round_trip_time/mean"
                ),
                "service_rtt_p95_seconds": _finite_nonnegative(
                    rollout, "perf_detail/rollout/vision_service_round_trip_time/p95"
                ),
                "rollout_seconds": _finite_nonnegative(rollout, "perf/rollout_time"),
                "actor_wait_ratio": _finite_nonnegative(training, "perf/wait_time_ratio"),
                "valid_action_rate": _finite_nonnegative(rollout, "rollout/valid_action/mean"),
                "sglang_id_hits": int(
                    rollout.get("perf_detail/rollout/sglang_vision_cache_id_only_hits/total", 0)
                ),
                "sglang_id_misses": int(
                    rollout.get("perf_detail/rollout/sglang_vision_cache_id_only_misses/total", 0)
                ),
                "sglang_republishes": int(
                    rollout.get("perf_detail/rollout/sglang_vision_cache_republish/total", 0)
                ),
                "sglang_inline_publishes": int(
                    rollout.get(
                        "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total", 0
                    )
                ),
            }
        )

    def total(key: str) -> float:
        return sum(float(result[key]) for result in results)

    cpu_cache_lookups = total("cpu_cache_hits") + total("cpu_cache_misses")
    id_lookups = total("sglang_id_hits") + total("sglang_id_misses")
    return {
        "log_path": str(Path(log_path)),
        "steady_cycles": results,
        "cpu_cache_hit_rate": total("cpu_cache_hits") / cpu_cache_lookups if cpu_cache_lookups else 0.0,
        "sglang_id_hit_rate": total("sglang_id_hits") / id_lookups if id_lookups else 0.0,
        "total_request_body_bytes": int(total("request_body_bytes")),
        "total_serialization_seconds": total("serialization_seconds"),
        "mean_service_rtt_seconds": total("service_rtt_mean_seconds") / len(results),
        "mean_service_rtt_p95_seconds": total("service_rtt_p95_seconds") / len(results),
        "mean_rollout_seconds": total("rollout_seconds") / len(results),
        "mean_actor_wait_ratio": total("actor_wait_ratio") / len(results),
        "total_backend_encoded_images": int(total("encoded_images")),
        "total_sglang_id_hits": int(total("sglang_id_hits")),
        "total_sglang_id_misses": int(total("sglang_id_misses")),
        "total_sglang_republishes": int(total("sglang_republishes")),
        "total_sglang_inline_publishes": int(total("sglang_inline_publishes")),
        "all_valid_actions": all(result["valid_action_rate"] == 1.0 for result in results),
    }


def measure_vision_cache_reuse(
    profile_logs: Mapping[str, str | Path],
    *,
    first_steady_cycle: int = 2,
    expected_steady_cycles: int = 10,
) -> dict[str, object]:
    """Compare CPU-cache-only transport with CPU and SGLang caches."""
    missing_profiles = sorted(set(_REQUIRED_PROFILES) - profile_logs.keys())
    if missing_profiles:
        raise ValueError(f"cache ablation is missing required profiles: {missing_profiles}")
    if first_steady_cycle < 0 or expected_steady_cycles <= 0:
        raise ValueError("steady-cycle bounds must be non-negative with at least one expected cycle")

    profiles = {
        profile: _analyze_profile(
            profile_logs[profile],
            first_steady_cycle=first_steady_cycle,
            expected_steady_cycles=expected_steady_cycles,
        )
        for profile in _REQUIRED_PROFILES
    }
    cpu_cache = profiles["cpu_cache_only"]
    cpu_and_sglang_cache = profiles["cpu_and_sglang_cache"]

    def reduction(baseline: float, candidate: float) -> float:
        if baseline <= 0:
            raise ValueError(f"cache ablation baseline must be positive, got {baseline}")
        return 1 - candidate / baseline

    result = {
        "first_steady_cycle": first_steady_cycle,
        "expected_steady_cycles": expected_steady_cycles,
        "profiles": profiles,
        "request_body_byte_reduction": reduction(
            cpu_cache["total_request_body_bytes"], cpu_and_sglang_cache["total_request_body_bytes"]
        ),
        "serialization_wall_reduction": reduction(
            cpu_cache["total_serialization_seconds"], cpu_and_sglang_cache["total_serialization_seconds"]
        ),
        "rollout_time_reduction": reduction(
            cpu_cache["mean_rollout_seconds"], cpu_and_sglang_cache["mean_rollout_seconds"]
        ),
        "p95_service_rtt_regression": (
            cpu_and_sglang_cache["mean_service_rtt_p95_seconds"] / cpu_cache["mean_service_rtt_p95_seconds"] - 1
        ),
        "all_valid_actions": all(profile["all_valid_actions"] for profile in profiles.values()),
        "no_republish": cpu_and_sglang_cache["total_sglang_republishes"] == 0,
    }
    result["performance_target_met"] = bool(
        result["all_valid_actions"]
        and result["no_republish"]
        and result["p95_service_rtt_regression"] <= 0.10
        and result["rollout_time_reduction"] >= 0.10
    )
    return result


def write_vision_cache_reuse_artifact(output_path: str | Path, analysis: Mapping[str, object]) -> dict[str, object]:
    artifact = {"schema_version": 1, **analysis}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _parse_profile(value: str) -> tuple[str, Path]:
    profile, separator, raw_path = value.partition("=")
    if not separator or not profile or not raw_path:
        raise argparse.ArgumentTypeError("profile logs must use NAME=PATH")
    return profile, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, type=_parse_profile)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-steady-cycle", type=int, default=2)
    parser.add_argument("--expected-steady-cycles", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    profile_logs = dict(args.profile)
    if len(profile_logs) != len(args.profile):
        raise ValueError("profile names must be unique")
    analysis = measure_vision_cache_reuse(
        profile_logs,
        first_steady_cycle=args.first_steady_cycle,
        expected_steady_cycles=args.expected_steady_cycles,
    )
    return write_vision_cache_reuse_artifact(args.output, analysis)


if __name__ == "__main__":
    main()
