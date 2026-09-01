# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze paired CPU-vision and training-cycle metrics from CPU vision demand console logs."""

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence


CAPACITY_HEADROOM_RATIO = 1.25
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_VISION_METRICS = re.compile(r"CPU vision metrics rollout_(\d+): (\{.*\})$")
_TRAINING_METRICS = re.compile(r"relax\.utils\.training\.train_metric_utils:\d+ perf (\d+): (\{.*\})$")


def _parse_metrics_log(log_path: str | Path) -> tuple[dict[int, dict], dict[int, dict]]:
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"CPU vision demand console log does not exist: {path}")

    vision_cycles: dict[int, dict] = {}
    training_cycles: dict[int, dict] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        vision_match = _VISION_METRICS.search(line)
        if vision_match:
            vision_cycles[int(vision_match.group(1))] = ast.literal_eval(vision_match.group(2))
            continue
        training_match = _TRAINING_METRICS.search(line)
        if training_match:
            training_cycles[int(training_match.group(1))] = ast.literal_eval(training_match.group(2))
    return vision_cycles, training_cycles


def _analyze_profile(
    log_path: str | Path,
    *,
    first_steady_cycle: int,
    expected_steady_cycles: int,
) -> dict[str, object]:
    vision_cycles, training_cycles = _parse_metrics_log(log_path)
    expected_cycles = list(range(first_steady_cycle, first_steady_cycle + expected_steady_cycles))
    paired_cycles = [
        cycle for cycle in expected_cycles if cycle in vision_cycles and cycle in training_cycles
    ]
    if paired_cycles != expected_cycles:
        missing_vision = sorted(set(expected_cycles) - vision_cycles.keys())
        missing_training = sorted(set(expected_cycles) - training_cycles.keys())
        raise ValueError(
            f"CPU vision demand log {log_path} does not contain all paired steady cycles; "
            f"missing_vision={missing_vision}, missing_training={missing_training}"
        )

    cycle_results = []
    for cycle in expected_cycles:
        vision = vision_cycles[cycle]
        training = training_cycles[cycle]
        unique_images = int(vision["vision_encoder/features/unique_interval"])
        requests = int(vision["vision_encoder/requests_interval"])
        encoded_images = int(vision["vision_encoder/backend/encoded_images_interval"])
        step_time = float(training["perf/step_time"])
        wait_ratio = float(training["perf/wait_time_ratio"])
        if unique_images <= 0 or requests <= 0 or encoded_images <= 0:
            raise ValueError(
                f"CPU vision demand cycle {cycle} in {log_path} contains non-positive image/request counts"
            )
        if not math.isfinite(step_time) or step_time <= 0:
            raise ValueError(
                f"CPU vision demand cycle {cycle} in {log_path} contains invalid training-cycle time {step_time}"
            )
        if not math.isfinite(wait_ratio) or wait_ratio < 0:
            raise ValueError(
                f"CPU vision demand cycle {cycle} in {log_path} contains invalid training wait ratio {wait_ratio}"
            )
        cycle_results.append(
            {
                "cycle": cycle,
                "unique_images": unique_images,
                "requests": requests,
                "encoded_images": encoded_images,
                "training_cycle_seconds": step_time,
                "wait_ratio": wait_ratio,
                "unique_images_per_second": unique_images / step_time,
            }
        )

    exact_request_accounting = all(
        cycle["requests"] == cycle["unique_images"] == cycle["encoded_images"]
        for cycle in cycle_results
    )
    return {
        "log_path": str(Path(log_path)),
        "steady_cycles": cycle_results,
        "peak_unique_images_per_second": max(
            cycle["unique_images_per_second"] for cycle in cycle_results
        ),
        "mean_wait_ratio": sum(cycle["wait_ratio"] for cycle in cycle_results)
        / len(cycle_results),
        "max_wait_ratio": max(cycle["wait_ratio"] for cycle in cycle_results),
        "exact_request_accounting": exact_request_accounting,
        "extra_requests": sum(
            max(cycle["requests"] - cycle["unique_images"], 0) for cycle in cycle_results
        ),
    }


def measure_cpu_vision_demand(
    profile_logs: Mapping[str, str | Path],
    *,
    first_steady_cycle: int = 2,
    expected_steady_cycles: int = 10,
    required_high_demand_setting: str = "prompts32_samples2",
    actor_wait_threshold: float = 0.05,
) -> dict[str, object]:
    """Compute CPU vision demand consumer demand without substituting backend producer throughput."""
    if not profile_logs:
        raise ValueError("profile_logs must not be empty")
    if first_steady_cycle < 0 or expected_steady_cycles <= 0:
        raise ValueError("steady-cycle bounds must be non-negative with at least one expected cycle")
    if not math.isfinite(actor_wait_threshold) or not 0 <= actor_wait_threshold <= 1:
        raise ValueError("actor_wait_threshold must be finite and between zero and one")
    if required_high_demand_setting not in profile_logs:
        raise ValueError(f"high-demand setting {required_high_demand_setting!r} is absent from profile_logs")

    profiles = {
        profile: _analyze_profile(
            log_path,
            first_steady_cycle=first_steady_cycle,
            expected_steady_cycles=expected_steady_cycles,
        )
        for profile, log_path in profile_logs.items()
    }
    peak_demand = max(profile["peak_unique_images_per_second"] for profile in profiles.values())
    saturation_result = profiles[required_high_demand_setting]
    return {
        "first_steady_cycle": first_steady_cycle,
        "expected_steady_cycles": expected_steady_cycles,
        "actor_wait_threshold": actor_wait_threshold,
        "required_high_demand_setting": required_high_demand_setting,
        "required_high_demand_setting_reached_actor_wait_gate": saturation_result["max_wait_ratio"]
        >= actor_wait_threshold,
        "peak_unique_images_per_second": peak_demand,
        "capacity_headroom_ratio": CAPACITY_HEADROOM_RATIO,
        "capacity_target_images_per_second": peak_demand * CAPACITY_HEADROOM_RATIO,
        "profiles": profiles,
    }


def write_demand_artifact(output_path: str | Path, analysis: Mapping[str, object]) -> dict[str, object]:
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
    parser.add_argument("--saturation-profile", default="prompts32_samples2")
    parser.add_argument("--actor-wait-threshold", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    profile_logs = dict(args.profile)
    if len(profile_logs) != len(args.profile):
        raise ValueError("profile names must be unique")
    analysis = measure_cpu_vision_demand(
        profile_logs,
        first_steady_cycle=args.first_steady_cycle,
        expected_steady_cycles=args.expected_steady_cycles,
        required_high_demand_setting=args.required_high_demand_setting,
        actor_wait_threshold=args.actor_wait_threshold,
    )
    return write_demand_artifact(args.output, analysis)


if __name__ == "__main__":
    main()
