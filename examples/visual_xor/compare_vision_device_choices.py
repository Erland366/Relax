# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare planned vision-device choices with fixed CPU and GPU runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from examples.visual_xor.compare_vision_settings import (
    _ANSI_ESCAPE,
    _finite,
    _parse_metrics_log,
    _summarize_comparison,
)


_CHOICE = re.compile(r"VISION_DEVICE_CHOICE (\{.*\})$")
_REQUIRED_RUNS = frozenset(("gpu", "cpu", "automatic"))


def _parse_choices(path: str | Path) -> dict[int, dict[str, Any]]:
    log_path = Path(path)
    choices = {}
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        match = _CHOICE.search(line)
        if match is None:
            continue
        choice = json.loads(match.group(1))
        if choice.get("schema_version") != 2:
            raise ValueError(f"Device-plan log {log_path} contains an unsupported choice schema")
        rollout_id = int(choice["rollout_id"])
        if rollout_id in choices:
            raise ValueError(f"Device-plan log {log_path} contains duplicate choice for cycle {rollout_id}")
        if choice.get("device") not in {"cpu", "gpu"}:
            raise ValueError(
                f"Device-plan log {log_path} cycle {rollout_id} has invalid device {choice.get('device')!r}"
            )
        for field in (
            "predicted_gpu_seconds",
            "predicted_cpu_seconds",
            "predicted_cpu_seconds_after_overlap",
            "minimum_gap",
        ):
            value = float(choice[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Device-plan log {log_path} field {field} must be finite and non-negative")
        predicted_gap = float(choice["predicted_gap"])
        if not math.isfinite(predicted_gap):
            raise ValueError(f"Device-plan log {log_path} predicted_gap must be finite")
        if float(choice["minimum_gap"]) >= 1:
            raise ValueError(f"Device-plan log {log_path} minimum_gap must be below 1")
        workload = choice.get("workload")
        if not isinstance(workload, dict) or set(workload) != {
            "image_count",
            "visual_tokens",
            "feature_bytes",
            "overlap_seconds",
        }:
            raise ValueError(f"Device-plan log {log_path} contains an invalid workload")
        choices[rollout_id] = choice
    return choices


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Vision-device comparison manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision-device comparison manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("Vision-device comparison manifest requires schema_version 2")
    cycles = manifest.get("measured_cycles")
    if not isinstance(cycles, list) or not cycles or any(not isinstance(cycle, int) for cycle in cycles):
        raise ValueError("Manifest requires a non-empty measured_cycles integer list")
    if len(set(cycles)) != len(cycles):
        raise ValueError("Manifest measured_cycles contains duplicates")
    repeats = manifest.get("repeats")
    if not isinstance(repeats, list) or len(repeats) < 3:
        raise ValueError("Vision-device comparison requires at least three repeated runs")
    return manifest


def _analyze_run(
    path: str | Path,
    *,
    run: str,
    cycles: Sequence[int],
    expected_responses: int,
    expected_generation_requests: int,
) -> dict[str, Any]:
    _, rollout, _, actor = _parse_metrics_log(path)
    missing_rollout = sorted(set(cycles) - rollout.keys())
    missing_actor = sorted(set(cycles) - actor.keys())
    if missing_rollout or missing_actor:
        raise ValueError(
            f"Vision-device run {path} is incomplete: "
            f"missing_rollout={missing_rollout}, missing_actor={missing_actor}"
        )
    choices = _parse_choices(path) if run == "automatic" else {}
    if run == "automatic":
        missing_choices = sorted(set(cycles) - choices.keys())
        unexpected_choices = sorted(choices.keys() - set(cycles))
        if missing_choices or unexpected_choices:
            raise ValueError(
                f"Device-choice mismatch: missing={missing_choices}, unexpected={unexpected_choices}"
            )

    records = []
    for cycle in cycles:
        rollout_metrics = rollout[cycle]
        responses = int(_finite(rollout_metrics, "perf_detail/rollout/sglang_parallel_samples/total"))
        if responses != expected_responses:
            raise ValueError(f"Run {path} cycle {cycle} expected {expected_responses} responses, got {responses}")
        generation_requests = int(
            _finite(rollout_metrics, "perf_detail/rollout/sglang_generation_requests/total")
        )
        if generation_requests != expected_generation_requests:
            raise ValueError(
                f"Run {path} cycle {cycle} expected {expected_generation_requests} generation requests, "
                f"got {generation_requests}"
            )
        valid_action_rate = _finite(rollout_metrics, "rollout/valid_action/mean")
        if valid_action_rate != 1.0:
            raise ValueError(f"Run {path} cycle {cycle} valid-action rate must be 1.0, got {valid_action_rate}")
        record = {
            "cycle": cycle,
            "training_cycle_seconds": _finite(actor[cycle], "perf/step_time"),
            "rollout_seconds": _finite(rollout_metrics, "perf/rollout_time"),
            "reward_mean": _finite(rollout_metrics, "rollout/reward/mean"),
            "responses": responses,
            "generation_requests": generation_requests,
        }
        if run == "automatic":
            choice = choices[cycle]
            cpu = int(_finite(rollout_metrics, "vision_device/cpu"))
            gpu = int(_finite(rollout_metrics, "vision_device/gpu"))
            if cpu not in {0, 1} or gpu not in {0, 1} or cpu + gpu != 1:
                raise ValueError(
                    f"Device-plan run {path} cycle {cycle} requires complementary CPU/GPU metrics; "
                    f"got cpu={cpu}, gpu={gpu}"
                )
            metric_device = "cpu" if cpu else "gpu"
            if metric_device != choice["device"]:
                raise ValueError(
                    f"Run {path} cycle {cycle} metric device {metric_device!r} "
                    f"does not match logged choice {choice['device']!r}"
                )
            record["choice"] = choice
        records.append(record)
    times = [record["training_cycle_seconds"] for record in records]
    return {
        "path": str(Path(path)),
        "cycles": records,
        "mean_training_cycle_seconds": statistics.fmean(times),
        "training_cycles_per_hour": 3600.0 / statistics.fmean(times),
        "mean_reward": statistics.fmean(record["reward_mean"] for record in records),
    }


def _analyze_repeat(
    repeat: Mapping[str, Any],
    *,
    cycles: Sequence[int],
    expected_responses: int,
    expected_generation_requests: int,
    require_both_devices: bool,
) -> dict[str, Any]:
    name = repeat.get("name")
    runs = repeat.get("runs")
    if not isinstance(name, str) or not name:
        raise ValueError("Each repeated run requires a non-empty name")
    if not isinstance(runs, dict) or set(runs) != _REQUIRED_RUNS:
        raise ValueError(f"Repeat {name} runs must be exactly {sorted(_REQUIRED_RUNS)}")
    analyzed = {
        run: _analyze_run(
            runs[run],
            run=run,
            cycles=cycles,
            expected_responses=expected_responses,
            expected_generation_requests=expected_generation_requests,
        )
        for run in sorted(_REQUIRED_RUNS)
    }

    cycle_results = []
    cycles_over_minimum_gap = 0
    device_matches = 0
    selected_devices = set()
    for cycle_index, cycle in enumerate(cycles):
        gpu_seconds = analyzed["gpu"]["cycles"][cycle_index]["training_cycle_seconds"]
        cpu_seconds = analyzed["cpu"]["cycles"][cycle_index]["training_cycle_seconds"]
        automatic_record = analyzed["automatic"]["cycles"][cycle_index]
        automatic_seconds = automatic_record["training_cycle_seconds"]
        selected_device = automatic_record["choice"]["device"]
        selected_devices.add(selected_device)
        faster_device = "cpu" if cpu_seconds < gpu_seconds else "gpu"
        faster_seconds = min(gpu_seconds, cpu_seconds)
        device_time_gap = abs(gpu_seconds - cpu_seconds) / max(gpu_seconds, cpu_seconds, 1e-12)
        minimum_gap = float(automatic_record["choice"]["minimum_gap"])
        over_minimum_gap = device_time_gap > minimum_gap
        device_match = selected_device == faster_device
        if over_minimum_gap:
            cycles_over_minimum_gap += 1
            device_matches += int(device_match)
        cycle_results.append(
            {
                "cycle": cycle,
                "selected_device": selected_device,
                "faster_device": faster_device,
                "device_time_gap": device_time_gap,
                "minimum_gap": minimum_gap,
                "over_minimum_gap": over_minimum_gap,
                "device_match": device_match,
                "gpu_seconds": gpu_seconds,
                "cpu_seconds": cpu_seconds,
                "automatic_seconds": automatic_seconds,
                "faster_seconds": faster_seconds,
                "extra_time": (automatic_seconds - faster_seconds) / faster_seconds,
            }
        )
    if require_both_devices and selected_devices != {"cpu", "gpu"}:
        raise ValueError(f"Repeat {name} must select both devices; got {sorted(selected_devices)}")

    gpu_mean = analyzed["gpu"]["mean_training_cycle_seconds"]
    cpu_mean = analyzed["cpu"]["mean_training_cycle_seconds"]
    automatic_mean = analyzed["automatic"]["mean_training_cycle_seconds"]
    return {
        "name": name,
        "runs": analyzed,
        "cycles": cycle_results,
        "cycles_over_minimum_gap": cycles_over_minimum_gap,
        "device_matches": device_matches,
        "match_rate": (
            device_matches / cycles_over_minimum_gap
            if cycles_over_minimum_gap
            else None
        ),
        "mean_extra_time": statistics.fmean(result["extra_time"] for result in cycle_results),
        "speedup_vs_gpu": 1.0 - automatic_mean / gpu_mean,
        "speedup_vs_cpu": 1.0 - automatic_mean / cpu_mean,
    }


def compare_vision_device_choices(path: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(path)
    cycles = manifest["measured_cycles"]
    expected_responses = int(manifest["expected_responses_per_cycle"])
    expected_generation_requests = int(manifest["expected_generation_requests_per_cycle"])
    if expected_responses <= 0 or expected_generation_requests <= 0:
        raise ValueError("expected response and generation-request counts must be positive")
    require_both_devices = bool(manifest.get("require_both_devices", False))
    repeats = [
        _analyze_repeat(
            repeat,
            cycles=cycles,
            expected_responses=expected_responses,
            expected_generation_requests=expected_generation_requests,
            require_both_devices=require_both_devices,
        )
        for repeat in manifest["repeats"]
    ]
    cycles_over_minimum_gap = sum(repeat["cycles_over_minimum_gap"] for repeat in repeats)
    device_matches = sum(repeat["device_matches"] for repeat in repeats)
    total_cycles = len(cycles) * len(repeats)
    mean_extra_time = statistics.fmean(repeat["mean_extra_time"] for repeat in repeats)
    match_rate = (
        device_matches / cycles_over_minimum_gap
        if cycles_over_minimum_gap
        else None
    )
    return {
        "schema_version": 2,
        "manifest": str(Path(path).resolve()),
        "repeats": repeats,
        "summary": {
            "repeat_count": len(repeats),
            "total_cycles": total_cycles,
            "cycles_over_minimum_gap": cycles_over_minimum_gap,
            "device_matches": device_matches,
            "match_rate": match_rate,
            "mean_extra_time": mean_extra_time,
            "speedup_vs_gpu": _summarize_comparison(
                [repeat["speedup_vs_gpu"] for repeat in repeats]
            ),
            "speedup_vs_cpu": _summarize_comparison(
                [repeat["speedup_vs_cpu"] for repeat in repeats]
            ),
            "acceptance": {
                "match_rate": match_rate is not None and match_rate >= 0.9,
                "extra_time": mean_extra_time < 0.05,
                "coverage": cycles_over_minimum_gap >= math.ceil(total_cycles / 2),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compare_vision_device_choices(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
