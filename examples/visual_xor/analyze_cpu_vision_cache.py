# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze the live CPU-vision producer and SGLang transport-cache ablation."""

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
_ACTOR_METRICS = re.compile(r"relax\.utils\.training\.train_metric_utils:\d+ perf (\d+): (\{.*\})$")
_REQUIRED_PROFILES = ("tier1_only", "tier1_tier2")


def _parse_metrics_log(log_path: str | Path) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"E4 cache console log does not exist: {path}")

    vision_cycles: dict[int, dict] = {}
    rollout_cycles: dict[int, dict] = {}
    actor_cycles: dict[int, dict] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        if match := _VISION_METRICS.search(line):
            vision_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
        elif match := _ROLLOUT_METRICS.search(line):
            rollout_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
        elif match := _ACTOR_METRICS.search(line):
            actor_cycles[int(match.group(1))] = ast.literal_eval(match.group(2))
    return vision_cycles, rollout_cycles, actor_cycles


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
    vision_cycles, rollout_cycles, actor_cycles = _parse_metrics_log(log_path)
    expected_cycles = list(range(first_steady_cycle, first_steady_cycle + expected_steady_cycles))
    paired_cycles = [
        cycle
        for cycle in expected_cycles
        if cycle in vision_cycles and cycle in rollout_cycles and cycle in actor_cycles
    ]
    if paired_cycles != expected_cycles:
        raise ValueError(
            f"E4 cache log {log_path} does not contain all paired steady cycles; "
            f"missing_vision={sorted(set(expected_cycles) - vision_cycles.keys())}, "
            f"missing_rollout={sorted(set(expected_cycles) - rollout_cycles.keys())}, "
            f"missing_actor={sorted(set(expected_cycles) - actor_cycles.keys())}"
        )

    results = []
    for cycle in expected_cycles:
        vision = vision_cycles[cycle]
        rollout = rollout_cycles[cycle]
        actor = actor_cycles[cycle]
        hits = int(vision["vision_encoder/cache/hits_interval"])
        misses = int(vision["vision_encoder/cache/misses_interval"])
        requests = int(vision["vision_encoder/requests_interval"])
        encoded_images = int(vision["vision_encoder/backend/encoded_images_interval"])
        if min(hits, misses, requests, encoded_images) < 0 or hits + misses != requests:
            raise ValueError(f"E4 cache cycle {cycle} in {log_path} has inconsistent producer counters")
        results.append(
            {
                "cycle": cycle,
                "producer_hits": hits,
                "producer_misses": misses,
                "producer_requests": requests,
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
                "rollout_wall_seconds": _finite_nonnegative(rollout, "perf/rollout_time"),
                "actor_wait_ratio": _finite_nonnegative(actor, "perf/wait_time_ratio"),
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

    producer_lookups = total("producer_hits") + total("producer_misses")
    id_lookups = total("sglang_id_hits") + total("sglang_id_misses")
    return {
        "log_path": str(Path(log_path)),
        "steady_cycles": results,
        "producer_cache_hit_rate": total("producer_hits") / producer_lookups if producer_lookups else 0.0,
        "sglang_id_hit_rate": total("sglang_id_hits") / id_lookups if id_lookups else 0.0,
        "total_request_body_bytes": int(total("request_body_bytes")),
        "total_serialization_seconds": total("serialization_seconds"),
        "mean_service_rtt_seconds": total("service_rtt_mean_seconds") / len(results),
        "mean_service_rtt_p95_seconds": total("service_rtt_p95_seconds") / len(results),
        "mean_rollout_wall_seconds": total("rollout_wall_seconds") / len(results),
        "mean_actor_wait_ratio": total("actor_wait_ratio") / len(results),
        "total_backend_encoded_images": int(total("encoded_images")),
        "total_sglang_id_hits": int(total("sglang_id_hits")),
        "total_sglang_id_misses": int(total("sglang_id_misses")),
        "total_sglang_republishes": int(total("sglang_republishes")),
        "total_sglang_inline_publishes": int(total("sglang_inline_publishes")),
        "all_valid_actions": all(result["valid_action_rate"] == 1.0 for result in results),
    }


def analyze_cache_ablation(
    profile_logs: Mapping[str, str | Path],
    *,
    first_steady_cycle: int = 2,
    expected_steady_cycles: int = 10,
) -> dict[str, object]:
    """Compare Tier 1-only transport with Tier 1 plus SGLang identity reuse."""
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
    tier1 = profiles["tier1_only"]
    tier2 = profiles["tier1_tier2"]

    def reduction(baseline: float, candidate: float) -> float:
        if baseline <= 0:
            raise ValueError(f"cache ablation baseline must be positive, got {baseline}")
        return 1 - candidate / baseline

    result = {
        "first_steady_cycle": first_steady_cycle,
        "expected_steady_cycles": expected_steady_cycles,
        "profiles": profiles,
        "request_body_byte_reduction": reduction(
            tier1["total_request_body_bytes"], tier2["total_request_body_bytes"]
        ),
        "serialization_wall_reduction": reduction(
            tier1["total_serialization_seconds"], tier2["total_serialization_seconds"]
        ),
        "rollout_wall_reduction": reduction(
            tier1["mean_rollout_wall_seconds"], tier2["mean_rollout_wall_seconds"]
        ),
        "p95_service_rtt_regression": (
            tier2["mean_service_rtt_p95_seconds"] / tier1["mean_service_rtt_p95_seconds"] - 1
        ),
        "all_valid_actions": all(profile["all_valid_actions"] for profile in profiles.values()),
        "no_republish": tier2["total_sglang_republishes"] == 0,
    }
    result["adoption_gate_passed"] = bool(
        result["all_valid_actions"]
        and result["no_republish"]
        and result["p95_service_rtt_regression"] <= 0.10
        and result["rollout_wall_reduction"] >= 0.10
    )
    return result


def write_cache_artifact(output_path: str | Path, analysis: Mapping[str, object]) -> dict[str, object]:
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
    analysis = analyze_cache_ablation(
        profile_logs,
        first_steady_cycle=args.first_steady_cycle,
        expected_steady_cycles=args.expected_steady_cycles,
    )
    return write_cache_artifact(args.output, analysis)


if __name__ == "__main__":
    main()
