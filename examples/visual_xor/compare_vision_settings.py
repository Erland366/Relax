# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare CPU/GPU vision and feature-cache settings across repeated runs."""

import argparse
import ast
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_VISION_METRICS = re.compile(r"CPU vision metrics rollout_(\d+): (\{.*\})$")
_ROLLOUT_METRICS = re.compile(r"relax\.distributed\.ray\.rollout:\d+ perf (\d+): (\{.*\})$")
_TRAIN_DATA_METRICS = re.compile(r"relax\.backends\.megatron\.data:\d+ rollout (\d+): (\{.*\})$")
_ACTOR_METRICS = re.compile(r"relax\.utils\.training\.train_metric_utils:\d+ perf (\d+): (\{.*\})$")
_T_CRITICAL_95 = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582}


@dataclass(frozen=True)
class VisionSettings:
    device: str
    cpu_cache: bool
    sglang_cache: bool

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "gpu"}:
            raise ValueError(f"device must be 'cpu' or 'gpu', got {self.device!r}")
        if self.device == "gpu" and (self.cpu_cache or self.sglang_cache):
            raise ValueError("GPU vision cannot use the CPU or SGLang feature cache")
        if self.sglang_cache and not self.cpu_cache:
            raise ValueError("The SGLang feature cache requires the CPU feature cache")


GPU = VisionSettings("gpu", False, False)
CPU = VisionSettings("cpu", False, False)
CPU_CACHE = VisionSettings("cpu", True, False)
CPU_AND_SGLANG_CACHE = VisionSettings("cpu", True, True)
_SETTINGS = (GPU, CPU, CPU_CACHE, CPU_AND_SGLANG_CACHE)
_COMPARISONS = (
    (GPU, CPU),
    (CPU, CPU_CACHE),
    (CPU_CACHE, CPU_AND_SGLANG_CACHE),
    (CPU, CPU_AND_SGLANG_CACHE),
    (GPU, CPU_AND_SGLANG_CACHE),
)


def _store_unique(target: dict[int, dict], cycle: int, metrics: dict, *, kind: str, path: Path) -> None:
    if cycle in target:
        raise ValueError(f"matched log {path} contains duplicate {kind} cycle {cycle}")
    target[cycle] = metrics


def _parse_metrics_log(
    log_path: str | Path,
) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict], dict[int, dict]]:
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"matched CPU-vision log does not exist: {path}")

    vision: dict[int, dict] = {}
    rollout: dict[int, dict] = {}
    train_data: dict[int, dict] = {}
    actor: dict[int, dict] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line)
        if match := _VISION_METRICS.search(line):
            _store_unique(vision, int(match.group(1)), ast.literal_eval(match.group(2)), kind="vision", path=path)
        elif match := _ROLLOUT_METRICS.search(line):
            _store_unique(rollout, int(match.group(1)), ast.literal_eval(match.group(2)), kind="rollout", path=path)
        elif match := _TRAIN_DATA_METRICS.search(line):
            _store_unique(
                train_data,
                int(match.group(1)),
                ast.literal_eval(match.group(2)),
                kind="train-data",
                path=path,
            )
        elif match := _ACTOR_METRICS.search(line):
            _store_unique(actor, int(match.group(1)), ast.literal_eval(match.group(2)), kind="actor", path=path)
    return vision, rollout, train_data, actor


def _finite(metrics: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    if key not in metrics:
        if default is None:
            raise ValueError(f"required metric {key} is missing")
        return default
    value = float(metrics[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"metric {key} must be finite and non-negative, got {value}")
    return value


def _nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _nearest_rank_p95(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _sum_suffix(metrics: Mapping[str, object], suffix: str) -> float:
    return sum(
        float(value)
        for key, value in metrics.items()
        if key.startswith("vision_encoder/replica/") and key.endswith(suffix)
    )


def _max_suffix(metrics: Mapping[str, object], suffix: str) -> float:
    values = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("vision_encoder/replica/") and key.endswith(suffix)
    ]
    return max(values, default=0.0)


def _analyze_run(
    log_path: str | Path,
    *,
    settings: VisionSettings,
    first_steady_cycle: int,
    expected_steady_cycles: int,
    expected_responses: int,
    optimizer_steps_per_cycle: int,
) -> dict[str, object]:
    vision, rollout, train_data, actor = _parse_metrics_log(log_path)
    expected_cycles = list(range(first_steady_cycle, first_steady_cycle + expected_steady_cycles))
    missing = {
        "rollout": sorted(set(expected_cycles) - rollout.keys()),
        "train_data": sorted(set(expected_cycles) - train_data.keys()),
        "actor": sorted(set(expected_cycles) - actor.keys()),
    }
    if settings.device == "cpu":
        missing["vision"] = sorted(set(expected_cycles) - vision.keys())
    if any(missing.values()):
        details = ", ".join(f"missing_{kind}={cycles}" for kind, cycles in missing.items())
        raise ValueError(f"matched log {log_path} is incomplete: {details}")

    cycles: list[dict[str, object]] = []
    for cycle in expected_cycles:
        rollout_metrics = rollout[cycle]
        actor_metrics = actor[cycle]
        train_metrics = train_data[cycle]
        responses = int(_finite(rollout_metrics, "perf_detail/rollout/sglang_parallel_samples/total"))
        if responses != expected_responses:
            raise ValueError(
                f"matched log {log_path} cycle {cycle} expected {expected_responses} responses, got {responses}"
            )
        valid_action_rate = _finite(rollout_metrics, "rollout/valid_action/mean")
        if valid_action_rate != 1.0:
            raise ValueError(
                f"matched log {log_path} cycle {cycle} valid-action rate must be 1.0, "
                f"got {valid_action_rate}"
            )

        response_length = _finite(rollout_metrics, "rollout/response_len/mean")
        total_length = _finite(train_metrics, "rollout/total_lengths")
        actor_step = _finite(actor_metrics, "perf/step_time")
        train_wait = _finite(actor_metrics, "perf/train_wait_time")
        train_scope = _finite(actor_metrics, "perf/train_time")
        if not math.isclose(actor_step, train_wait + train_scope, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"matched log {log_path} cycle {cycle} has inconsistent actor boundary: "
                f"step={actor_step}, wait+train={train_wait + train_scope}"
            )

        record: dict[str, object] = {
            "cycle": cycle,
            "training_cycle_seconds": actor_step,
            "wait_seconds": train_wait,
            "training_seconds": train_scope,
            "model_update_seconds": _finite(actor_metrics, "perf/actor_train_time"),
            "weight_update_seconds": _finite(
                actor_metrics, "perf/update_weights_fully_async_time", default=0.0
            ),
            "wait_ratio": _finite(actor_metrics, "perf/wait_time_ratio"),
            "rollout_seconds": _finite(rollout_metrics, "perf/rollout_time"),
            "responses": responses,
            "response_tokens": responses * response_length,
            "training_tokens": responses * total_length,
            "valid_action_rate": valid_action_rate,
            "reward_mean": _finite(rollout_metrics, "rollout/reward/mean"),
            "action_a_mean": _finite(rollout_metrics, "rollout/action_a/mean"),
            "action_b_mean": _finite(rollout_metrics, "rollout/action_b/mean"),
            "request_body_bytes": int(
                _finite(rollout_metrics, "perf_detail/rollout/http_request_body_bytes/total", default=0.0)
            ),
            "serialization_seconds": _finite(
                rollout_metrics, "perf_detail/rollout/precomputed_to_list_time/total", default=0.0
            ),
            "service_rtt_mean_seconds": _finite(
                rollout_metrics, "perf_detail/rollout/vision_service_round_trip_time/mean", default=0.0
            ),
            "service_rtt_p95_seconds": _finite(
                rollout_metrics, "perf_detail/rollout/vision_service_round_trip_time/p95", default=0.0
            ),
            "sglang_id_hits": int(
                _finite(
                    rollout_metrics,
                    "perf_detail/rollout/sglang_vision_cache_id_only_hits/total",
                    default=0.0,
                )
            ),
            "sglang_id_misses": int(
                _finite(
                    rollout_metrics,
                    "perf_detail/rollout/sglang_vision_cache_id_only_misses/total",
                    default=0.0,
                )
            ),
            "sglang_republishes": int(
                _finite(
                    rollout_metrics,
                    "perf_detail/rollout/sglang_vision_cache_republish/total",
                    default=0.0,
                )
            ),
            "sglang_inline_publishes": int(
                _finite(
                    rollout_metrics,
                    "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total",
                    default=0.0,
                )
            ),
        }
        if settings.device == "cpu":
            vision_metrics = vision[cycle]
            hits = int(_finite(vision_metrics, "vision_encoder/cache/hits_interval"))
            misses = int(_finite(vision_metrics, "vision_encoder/cache/misses_interval"))
            requests = int(_finite(vision_metrics, "vision_encoder/requests_interval"))
            if hits + misses != requests:
                raise ValueError(
                    f"matched log {log_path} cycle {cycle} has inconsistent CPU-cache requests: "
                    f"hits={hits}, misses={misses}, requests={requests}"
                )
            if int(_finite(vision_metrics, "vision_encoder/replicas/unobserved")) != 0:
                raise ValueError(f"matched log {log_path} cycle {cycle} has an unobserved vision replica")
            record["vision"] = {
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_evictions": int(_finite(vision_metrics, "vision_encoder/cache/evictions_interval")),
                "requests": requests,
                "backend_forwards": int(
                    _finite(vision_metrics, "vision_encoder/backend/encode_requests_interval")
                ),
                "encoded_images": int(
                    _finite(vision_metrics, "vision_encoder/backend/encoded_images_interval")
                ),
                "encode_seconds": _finite(
                    vision_metrics, "vision_encoder/backend/encode_seconds_interval"
                ),
                "unique_features": int(
                    _finite(vision_metrics, "vision_encoder/features/unique_interval")
                ),
                "duplicate_encode_ratio": _finite(
                    vision_metrics, "vision_encoder/backend/duplicate_encode_ratio_interval"
                ),
                "backend_batch_size_mean": _finite(
                    vision_metrics, "vision_encoder/backend/batch_size_mean_interval"
                ),
                "backend_batch_size_p95": _finite(
                    vision_metrics, "vision_encoder/backend/batch_size_p95_interval"
                ),
                "backend_batch_size_max": _finite(
                    vision_metrics, "vision_encoder/backend/batch_size_max_interval"
                ),
                "replicas_expected": int(_finite(vision_metrics, "vision_encoder/replicas/expected")),
                "replicas_observed": int(_finite(vision_metrics, "vision_encoder/replicas/observed")),
                "replicas_active": int(_finite(vision_metrics, "vision_encoder/replicas/active")),
                "replica_requests": _sum_suffix(vision_metrics, "/requests_interval"),
                "replica_cpu_utilization_percent_max": _max_suffix(
                    vision_metrics, "/process_cpu_utilization_percent_interval"
                ),
                "replica_process_cpu_seconds_total_max": _max_suffix(
                    vision_metrics, "/process_cpu_seconds_total"
                ),
                "replica_rss_bytes_max": _max_suffix(vision_metrics, "/rss_bytes"),
            }
        cycles.append(record)

    if settings.device == "cpu" and not settings.cpu_cache:
        if any(int(cycle["vision"]["cache_hits"]) for cycle in cycles):
            raise ValueError(f"CPU run {log_path} unexpectedly contains CPU-cache hits")
    if settings.cpu_cache and not any(int(cycle["vision"]["cache_hits"]) for cycle in cycles):
        raise ValueError(f"CPU run {log_path} does not demonstrate CPU-cache reuse")
    if settings.sglang_cache and any(int(cycle["sglang_republishes"]) for cycle in cycles):
        raise ValueError(f"SGLang-cache run {log_path} contains an unexpected republish")

    actor_values = [float(cycle["training_cycle_seconds"]) for cycle in cycles]
    actor_mean = statistics.fmean(actor_values)
    summary = {
        "training_cycle_seconds": _describe(actor_values),
        "wait_seconds": _describe([float(cycle["wait_seconds"]) for cycle in cycles]),
        "training_seconds": _describe([float(cycle["training_seconds"]) for cycle in cycles]),
        "model_update_seconds": _describe([float(cycle["model_update_seconds"]) for cycle in cycles]),
        "weight_update_seconds": _describe([float(cycle["weight_update_seconds"]) for cycle in cycles]),
        "rollout_seconds": _describe([float(cycle["rollout_seconds"]) for cycle in cycles]),
        "service_rtt_mean_seconds": _describe(
            [float(cycle["service_rtt_mean_seconds"]) for cycle in cycles]
        ),
        "service_rtt_p95_seconds": _describe(
            [float(cycle["service_rtt_p95_seconds"]) for cycle in cycles]
        ),
        "training_cycles_per_hour": 3600.0 / actor_mean,
        "optimizer_steps_per_hour": optimizer_steps_per_cycle * 3600.0 / actor_mean,
        "total_responses": sum(int(cycle["responses"]) for cycle in cycles),
        "total_response_tokens": sum(float(cycle["response_tokens"]) for cycle in cycles),
        "total_training_tokens": sum(float(cycle["training_tokens"]) for cycle in cycles),
        "mean_wait_ratio": statistics.fmean(float(cycle["wait_ratio"]) for cycle in cycles),
        "mean_reward": statistics.fmean(float(cycle["reward_mean"]) for cycle in cycles),
        "mean_action_a": statistics.fmean(float(cycle["action_a_mean"]) for cycle in cycles),
        "mean_action_b": statistics.fmean(float(cycle["action_b_mean"]) for cycle in cycles),
        "total_request_body_bytes": sum(int(cycle["request_body_bytes"]) for cycle in cycles),
        "total_serialization_seconds": sum(float(cycle["serialization_seconds"]) for cycle in cycles),
        "total_sglang_id_hits": sum(int(cycle["sglang_id_hits"]) for cycle in cycles),
        "total_sglang_id_misses": sum(int(cycle["sglang_id_misses"]) for cycle in cycles),
        "total_sglang_republishes": sum(int(cycle["sglang_republishes"]) for cycle in cycles),
        "total_sglang_inline_publishes": sum(
            int(cycle["sglang_inline_publishes"]) for cycle in cycles
        ),
    }
    if settings.device == "cpu":
        summary["vision"] = {
            key: sum(float(cycle["vision"][key]) for cycle in cycles)
            for key in (
                "cache_hits",
                "cache_misses",
                "cache_evictions",
                "requests",
                "backend_forwards",
                "encoded_images",
                "encode_seconds",
                "unique_features",
            )
        }
        summary["vision"]["replica_cpu_utilization_percent"] = _describe(
            [float(cycle["vision"]["replica_cpu_utilization_percent_max"]) for cycle in cycles]
        )
        summary["vision"]["replica_process_cpu_seconds_total"] = _describe(
            [float(cycle["vision"]["replica_process_cpu_seconds_total_max"]) for cycle in cycles]
        )
        summary["vision"]["replica_rss_bytes"] = _describe(
            [float(cycle["vision"]["replica_rss_bytes_max"]) for cycle in cycles]
        )
    return {"log_path": str(Path(log_path)), "steady_cycles": cycles, **summary}


def _summarize_comparison(values: Sequence[float]) -> dict[str, object]:
    repeat_count = len(values)
    if repeat_count < 2:
        raise ValueError("a confidence interval requires at least two repeated runs")
    degrees_of_freedom = repeat_count - 1
    critical = _T_CRITICAL_95.get(degrees_of_freedom)
    if critical is None:
        raise ValueError(f"no checked 95% t critical value for {degrees_of_freedom} degrees of freedom")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = critical * standard_deviation / math.sqrt(repeat_count)
    return {
        "repeat_count": repeat_count,
        "values": list(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "degrees_of_freedom": degrees_of_freedom,
        "confidence_level": 0.95,
        "confidence_interval": [mean - half_width, mean + half_width],
    }


def compare_vision_settings(
    repeat_runs: Mapping[str, Mapping[VisionSettings, str | Path]],
    *,
    first_steady_cycle: int = 2,
    expected_steady_cycles: int = 20,
    expected_responses: int = 64,
    optimizer_steps_per_cycle: int = 2,
) -> dict[str, object]:
    """Compare four vision settings without treating cycles as independent repeats."""
    if first_steady_cycle < 0 or expected_steady_cycles <= 0:
        raise ValueError("steady-cycle bounds must be non-negative with at least one expected cycle")
    if expected_responses <= 0 or optimizer_steps_per_cycle <= 0:
        raise ValueError("response and optimizer-step counts must be positive")
    if len(repeat_runs) < 2:
        raise ValueError("vision-setting comparison requires at least two repeated runs")

    repeats: dict[str, object] = {}
    for repeat in sorted(repeat_runs):
        missing = [asdict(settings) for settings in _SETTINGS if settings not in repeat_runs[repeat]]
        extra = [asdict(settings) for settings in repeat_runs[repeat] if settings not in _SETTINGS]
        if missing or extra:
            raise ValueError(f"repeat {repeat} has missing={missing}, extra={extra}")
        repeats[repeat] = {
            "runs": [
                {
                    "settings": asdict(settings),
                    "metrics": _analyze_run(
                    repeat_runs[repeat][settings],
                    settings=settings,
                    first_steady_cycle=first_steady_cycle,
                    expected_steady_cycles=expected_steady_cycles,
                    expected_responses=expected_responses,
                    optimizer_steps_per_cycle=optimizer_steps_per_cycle,
                    ),
                }
                for settings in _SETTINGS
            ]
        }

    comparisons = []
    for baseline, candidate in _COMPARISONS:
        values = []
        for repeat in sorted(repeats):
            runs = {
                VisionSettings(**run["settings"]): run["metrics"]
                for run in repeats[repeat]["runs"]
            }
            baseline_time = float(runs[baseline]["training_cycle_seconds"]["mean"])
            candidate_time = float(runs[candidate]["training_cycle_seconds"]["mean"])
            values.append(1.0 - candidate_time / baseline_time)
        comparisons.append(
            {
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "speedup": _summarize_comparison(values),
            }
        )

    return {
        "repeat_count": len(repeats),
        "note": "Cycles are repeated measurements within a run, not independent repeats.",
        "first_steady_cycle": first_steady_cycle,
        "expected_steady_cycles": expected_steady_cycles,
        "expected_responses_per_cycle": expected_responses,
        "optimizer_steps_per_cycle": optimizer_steps_per_cycle,
        "repeats": repeats,
        "comparisons": comparisons,
    }


def write_vision_settings(output_path: str | Path, analysis: Mapping[str, object]) -> dict[str, object]:
    artifact = {"schema_version": 2, **analysis}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _parse_boolean(value: str, *, field: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError(f"{field} must be 0 or 1")
    return value == "1"


def _parse_run(value: str) -> tuple[str, VisionSettings, Path]:
    identity, separator, raw_path = value.partition("=")
    parts = identity.split(":")
    if not separator or len(parts) != 4 or not raw_path:
        raise argparse.ArgumentTypeError(
            "runs must use REPEAT:DEVICE:CPU_CACHE:SGLANG_CACHE=PATH"
        )
    repeat, device, cpu_cache, sglang_cache = parts
    if not repeat:
        raise argparse.ArgumentTypeError("run repeat must not be empty")
    try:
        settings = VisionSettings(
            device=device,
            cpu_cache=_parse_boolean(cpu_cache, field="CPU_CACHE"),
            sglang_cache=_parse_boolean(sglang_cache, field="SGLANG_CACHE"),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return repeat, settings, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-steady-cycle", type=int, default=2)
    parser.add_argument("--expected-steady-cycles", type=int, default=20)
    parser.add_argument("--expected-responses", type=int, default=64)
    parser.add_argument("--optimizer-steps-per-cycle", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    repeat_runs: dict[str, dict[VisionSettings, Path]] = {}
    for repeat, settings, path in args.run:
        if settings in repeat_runs.setdefault(repeat, {}):
            raise ValueError(f"duplicate run for repeat={repeat}, settings={asdict(settings)}")
        repeat_runs[repeat][settings] = path
    analysis = compare_vision_settings(
        repeat_runs,
        first_steady_cycle=args.first_steady_cycle,
        expected_steady_cycles=args.expected_steady_cycles,
        expected_responses=args.expected_responses,
        optimizer_steps_per_cycle=args.optimizer_steps_per_cycle,
    )
    return write_vision_settings(args.output, analysis)


if __name__ == "__main__":
    main()
