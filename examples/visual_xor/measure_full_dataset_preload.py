# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Measure the cost and benefit of preloading every dataset feature."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from examples.visual_xor.compare_vision_settings import (
    CPU,
    CPU_AND_SGLANG_CACHE,
    _analyze_run,
    _summarize_comparison,
    _finite,
    _parse_metrics_log,
)


_PRELOAD_SETTINGS = ("none", "full_dataset")
_EXPECTED_IMAGES_PER_CYCLE = 32
_EXPECTED_DATASET_FEATURES = 64
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OPTIMIZER_COMPLETION = re.compile(
    r"train_one_step rollout=(\d+) step=(\d+): "
    r"finished optimizer\.step \(update_successful=True(?:,[^)]*)?\)"
)
_EVALUATION_EVIDENCE = re.compile(r"relax\.distributed\.ray\.rollout:\d+\s+eval\s+\d+:")
_CHECKPOINT_WRITE_EVIDENCE = re.compile(
    r"\b(?:successfully\s+)?saved checkpoint\b|\bsaving checkpoint\b",
    re.IGNORECASE,
)


def _require_artifact_value(
    repeat: str,
    artifact: Mapping[str, object],
    field: str,
    expected: object,
    requirement: str,
) -> None:
    actual = artifact.get(field)
    if actual != expected:
        raise ValueError(f"{repeat} preload artifact requires {requirement}; got {actual!r}")


def _require_cache(
    repeat: str,
    artifact: Mapping[str, object],
    cache_key: str,
    cache_name: str,
    *,
    require_stores: bool = False,
) -> None:
    cache = artifact.get(cache_key)
    entries = cache.get("entries") if isinstance(cache, dict) else None
    if entries != _EXPECTED_DATASET_FEATURES:
        raise ValueError(f"{repeat} preload artifact requires {cache_name} entries=64; got {entries!r}")
    if require_stores:
        _require_artifact_value(
            repeat,
            cache,
            "stores",
            _EXPECTED_DATASET_FEATURES,
            f"{cache_name} stores=64",
        )
    _require_artifact_value(repeat, cache, "evictions", 0, f"{cache_name} evictions to be zero")


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_feature_records(
    repeat: str,
    artifact: Mapping[str, object],
    feature_ids: list[object],
) -> None:
    records = artifact.get("features")
    if not isinstance(records, list) or len(records) != _EXPECTED_DATASET_FEATURES:
        count = len(records) if isinstance(records, list) else None
        raise ValueError(f"{repeat} preload artifact requires 64 feature records; got {count!r}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{repeat} preload artifact feature {index} must be a metadata object")
        feature_id = record.get("feature_id")
        if not isinstance(feature_id, str) or not feature_id:
            raise ValueError(
                f"{repeat} preload artifact feature {index} requires a non-empty feature_id"
            )
        if feature_id != feature_ids[index]:
            raise ValueError(
                f"{repeat} preload artifact feature record order must match feature_ids at index {index}"
            )
        revision = record.get("vision_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                f"{repeat} preload artifact feature {index} vision_revision must be non-empty"
            )
        schema = record.get("feature_schema_version")
        if not ((isinstance(schema, str) and schema.strip()) or _is_positive_integer(schema)):
            raise ValueError(
                f"{repeat} preload artifact feature {index} feature_schema_version must be non-empty"
            )
        grid = record.get("image_grid_thw")
        valid_grid = (
            isinstance(grid, list)
            and bool(grid)
            and all(
                isinstance(row, list)
                and len(row) == 3
                and all(_is_positive_integer(dimension) for dimension in row)
                for row in grid
            )
        )
        if not valid_grid:
            raise ValueError(
                f"{repeat} preload artifact feature {index} image_grid_thw must contain positive integers"
            )
        for field in ("image_tokens", "feature_bytes"):
            value = record.get(field)
            if not _is_positive_integer(value):
                raise ValueError(
                    f"{repeat} preload artifact feature {index} {field} must be a positive integer; "
                    f"got {value!r}"
                )


def _validate_no_work_or_dataset_change(repeat: str, artifact: Mapping[str, object]) -> None:
    work_counters = artifact.get("work_counters")
    for counter in ("generation_requests", "generated_samples", "training_samples"):
        value = work_counters.get(counter) if isinstance(work_counters, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise ValueError(
                f"{repeat} preload artifact work_counters {counter} must be exactly zero; got {value!r}"
            )

    dataset_state = artifact.get("dataset_state")
    before = dataset_state.get("before") if isinstance(dataset_state, dict) else None
    after = dataset_state.get("after") if isinstance(dataset_state, dict) else None
    required_state_fields = {
        "sample_offset",
        "epoch_id",
        "sample_group_index",
        "sample_index",
        "dataset_size",
        "dataset_fingerprint",
    }
    unchanged = (
        isinstance(dataset_state, dict)
        and dataset_state.get("unchanged") is True
        and isinstance(before, dict)
        and isinstance(after, dict)
        and required_state_fields <= before.keys()
        and required_state_fields <= after.keys()
        and before == after
    )
    if not unchanged:
        raise ValueError(
            f"{repeat} preload artifact dataset_state must prove unchanged=True with identical before and after state"
        )


def _validate_cache_setting(
    repeat: str,
    preload: str,
    vision: Mapping[int, Mapping[str, object]],
    rollout: Mapping[int, Mapping[str, object]],
    *,
    expected_cycles: int,
) -> None:
    for cycle in range(expected_cycles):
        vision_metrics = vision[cycle]
        rollout_metrics = rollout[cycle]
        requests = int(_finite(vision_metrics, "vision_encoder/requests_interval"))
        hits = int(_finite(vision_metrics, "vision_encoder/cache/hits_interval"))
        misses = int(_finite(vision_metrics, "vision_encoder/cache/misses_interval"))
        evictions = int(_finite(vision_metrics, "vision_encoder/cache/evictions_interval"))
        encodes = int(_finite(vision_metrics, "vision_encoder/backend/encode_requests_interval"))
        encoded_images = int(
            _finite(vision_metrics, "vision_encoder/backend/encoded_images_interval")
        )
        id_hits = int(
            _finite(
                rollout_metrics,
                "perf_detail/rollout/sglang_vision_cache_id_only_hits/total",
                default=0.0,
            )
        )
        id_misses = int(
            _finite(
                rollout_metrics,
                "perf_detail/rollout/sglang_vision_cache_id_only_misses/total",
                default=0.0,
            )
        )
        inline_publishes = int(
            _finite(
                rollout_metrics,
                "perf_detail/rollout/sglang_vision_cache_inline_publish_requests/total",
                default=0.0,
            )
        )
        republishes = int(
            _finite(
                rollout_metrics,
                "perf_detail/rollout/sglang_vision_cache_republish/total",
                default=0.0,
            )
        )

        if preload == "full_dataset":
            if requests != _EXPECTED_IMAGES_PER_CYCLE:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} requires exactly 32 CPU cache requests and hits; "
                    f"got requests={requests}, hits={hits}"
                )
            if hits != _EXPECTED_IMAGES_PER_CYCLE:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} requires exactly 32 CPU cache hits and zero "
                    f"misses/encodes; got hits={hits}, misses={misses}, encodes={encodes}"
                )
            if misses != 0 or encodes != 0 or encoded_images != 0:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} requires zero CPU cache misses and backend encodes; "
                    f"got misses={misses}, encode_requests={encodes}, encoded_images={encoded_images}"
                )
            if evictions != 0:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} requires zero CPU cache evictions; got {evictions}"
                )
            if id_hits != _EXPECTED_IMAGES_PER_CYCLE:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} requires exactly 32 SGLang cache ID hits; got {id_hits}"
                )
            if id_misses != 0 or inline_publishes != 0 or republishes != 0:
                raise ValueError(
                    f"{repeat}:{preload} cycle {cycle} has unexpected SGLang-cache activity: "
                    f"id_hits={id_hits}, id_misses={id_misses}, "
                    f"inline_publishes={inline_publishes}, republishes={republishes}"
                )
        elif hits != 0 or id_hits != 0 or id_misses != 0 or inline_publishes != 0 or republishes != 0:
            raise ValueError(
                f"{repeat}:{preload} cycle {cycle} requires effective caches disabled; "
                f"CPU cache hits={hits}, SGLang cache hits/misses/publishes/republishes="
                f"{id_hits}/{id_misses}/{inline_publishes}/{republishes}"
            )
        elif (
            requests != _EXPECTED_IMAGES_PER_CYCLE
            or misses != _EXPECTED_IMAGES_PER_CYCLE
            or encodes != _EXPECTED_IMAGES_PER_CYCLE
        ):
            raise ValueError(
                f"{repeat}:{preload} cycle {cycle} requires exactly 32 CPU cache requests, misses, and encodes; "
                f"got requests={requests}, misses={misses}, encodes={encodes}"
            )
        elif encoded_images != _EXPECTED_IMAGES_PER_CYCLE:
            raise ValueError(
                f"{repeat}:{preload} cycle {cycle} requires exactly 32 backend encoded images; "
                f"got {encoded_images}"
            )


def _validate_lifecycle(
    repeat: str,
    preload: str,
    log_path: str | Path,
    *,
    expected_cycles: int,
    optimizer_steps_per_cycle: int,
) -> dict[str, object]:
    path = Path(log_path)
    plain_text = _ANSI_ESCAPE.sub("", path.read_text(errors="replace"))

    if _EVALUATION_EVIDENCE.search(plain_text):
        raise ValueError(f"{repeat}:{preload} contains evaluation evidence despite ENABLE_EVAL=0")
    if _CHECKPOINT_WRITE_EVIDENCE.search(plain_text):
        raise ValueError(
            f"{repeat}:{preload} contains checkpoint-write evidence despite SAVE_CHECKPOINTS=0"
        )

    completions = Counter(
        (int(match.group(1)), int(match.group(2)))
        for match in _OPTIMIZER_COMPLETION.finditer(plain_text)
    )
    expected_completions = {
        (rollout, step)
        for rollout in range(expected_cycles)
        for step in range(optimizer_steps_per_cycle)
    }
    for rollout, step in sorted(expected_completions):
        count = completions[(rollout, step)]
        if count == 0:
            raise ValueError(
                f"{repeat}:{preload} missing optimizer completion for rollout {rollout} step {step}"
            )
        if count > 1:
            raise ValueError(
                f"{repeat}:{preload} duplicate optimizer completion for rollout {rollout} step {step}; "
                f"observed {count}"
            )
    unexpected = sorted(set(completions) - expected_completions)
    if unexpected:
        rollout, step = unexpected[0]
        raise ValueError(
            f"{repeat}:{preload} contains unexpected optimizer completion for rollout {rollout} step {step}"
        )

    final_step = expected_cycles - 1
    final_sync_marker = f"Actor training completed step {final_step}/{expected_cycles}"
    if final_sync_marker not in plain_text:
        raise ValueError(
            f"{repeat}:{preload} is missing final synchronization completion at actor step {final_step}"
        )
    required_completion_markers = (
        ("All training steps finished", "actor service did not return after final synchronization"),
        ("All rollouts finished", "rollout service completion is missing"),
        ("Service task completed successfully", "is missing controller service completion"),
    )
    for marker, error_message in required_completion_markers:
        if marker not in plain_text:
            raise ValueError(f"{repeat}:{preload} {error_message}")

    return {
        "optimizer_completion_count": len(expected_completions),
        "final_actor_step": final_step,
        "actor_service_completed": True,
        "rollout_service_completed": True,
        "controller_service_completed": True,
        "evaluation_observed": False,
        "checkpoint_write_observed": False,
    }


def _analyze_preload_run(
    repeat: str,
    preload: str,
    log_path: str | Path,
    *,
    expected_cycles: int,
    expected_responses: int,
    optimizer_steps_per_cycle: int,
) -> dict[str, object]:
    vision, rollout, train_data, actor = _parse_metrics_log(log_path)
    expected = set(range(expected_cycles))
    observed = {
        "vision": set(vision),
        "rollout": set(rollout),
        "train_data": set(train_data),
        "actor": set(actor),
    }
    missing = {kind: sorted(expected - cycles) for kind, cycles in observed.items()}
    extra = {kind: sorted(cycles - expected) for kind, cycles in observed.items()}
    if any(missing.values()) or any(extra.values()):
        raise ValueError(
            f"{repeat}:{preload} does not contain exactly cycles 0-{expected_cycles - 1}: "
            f"missing={missing}, extra={extra}"
        )

    _validate_cache_setting(repeat, preload, vision, rollout, expected_cycles=expected_cycles)
    lifecycle = _validate_lifecycle(
        repeat,
        preload,
        log_path,
        expected_cycles=expected_cycles,
        optimizer_steps_per_cycle=optimizer_steps_per_cycle,
    )
    profile = _analyze_run(
        log_path,
        settings=CPU if preload == "none" else CPU_AND_SGLANG_CACHE,
        first_steady_cycle=0,
        expected_steady_cycles=expected_cycles,
        expected_responses=expected_responses,
        optimizer_steps_per_cycle=optimizer_steps_per_cycle,
    )
    measured_seconds = sum(float(cycle["training_cycle_seconds"]) for cycle in profile["steady_cycles"])
    profile["measured_training_seconds"] = measured_seconds
    profile["lifecycle"] = lifecycle
    return profile


def _read_preload(repeat: str, path_value: str | Path) -> tuple[dict[str, object], float]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{repeat} preload artifact does not exist: {path}")
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{repeat} preload artifact is not valid JSON: {path}") from error
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 2:
        actual = artifact.get("schema_version") if isinstance(artifact, dict) else None
        raise ValueError(
            f"{repeat} preload artifact requires schema_version 2; got {actual!r}"
        )
    try:
        wall_seconds = float(artifact["wall_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{repeat} preload artifact requires numeric wall_seconds") from error
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise ValueError(
            f"{repeat} preload artifact wall_seconds must be finite and non-negative; got {wall_seconds}"
        )

    _require_artifact_value(
        repeat, artifact, "dataset_samples", _EXPECTED_DATASET_FEATURES, "dataset_samples=64"
    )
    _require_artifact_value(
        repeat, artifact, "unique_features", _EXPECTED_DATASET_FEATURES, "unique_features=64"
    )
    _require_artifact_value(repeat, artifact, "duplicate_features", 0, "duplicate_features to be zero")
    feature_ids = artifact.get("feature_ids")
    if (
        not isinstance(feature_ids, list)
        or len(feature_ids) != _EXPECTED_DATASET_FEATURES
        or len(set(feature_ids)) != _EXPECTED_DATASET_FEATURES
    ):
        raise ValueError(f"{repeat} preload artifact requires 64 feature IDs with no duplicates")

    _validate_feature_records(repeat, artifact, feature_ids)
    _validate_no_work_or_dataset_change(repeat, artifact)
    _require_cache(repeat, artifact, "cpu_cache", "CPU cache")
    _require_cache(repeat, artifact, "sglang_cache", "SGLang cache", require_stores=True)
    return artifact, wall_seconds


def measure_full_dataset_preload(
    repeat_runs: Mapping[str, Mapping[str, str | Path]],
    preload_artifacts: Mapping[str, str | Path],
    *,
    expected_cycles: int = 22,
    expected_responses: int = 64,
    optimizer_steps_per_cycle: int = 2,
) -> dict[str, object]:
    """Compare training time with and without full-dataset feature preload."""
    if expected_cycles <= 0 or expected_responses <= 0 or optimizer_steps_per_cycle <= 0:
        raise ValueError("cycle, response, and optimizer-step counts must be positive")
    if len(repeat_runs) != 4 or set(repeat_runs) != set(preload_artifacts):
        raise ValueError("full-dataset preload analysis requires exactly four complete repeats")

    repeats: dict[str, object] = {}
    training_only_speedups = []
    speedups_including_preload = []
    for repeat in sorted(repeat_runs):
        runs = repeat_runs[repeat]
        missing = sorted(set(_PRELOAD_SETTINGS) - runs.keys())
        extra = sorted(set(runs) - set(_PRELOAD_SETTINGS))
        if missing or extra:
            raise ValueError(
                f"full-dataset preload repeat {repeat} is incomplete: "
                f"missing={missing}, extra={extra}"
            )

        preload_artifact, preload_seconds = _read_preload(repeat, preload_artifacts[repeat])
        profiles = {
            preload: _analyze_preload_run(
                repeat,
                preload,
                runs[preload],
                expected_cycles=expected_cycles,
                expected_responses=expected_responses,
                optimizer_steps_per_cycle=optimizer_steps_per_cycle,
            )
            for preload in _PRELOAD_SETTINGS
        }
        no_preload_seconds = float(profiles["none"]["measured_training_seconds"])
        full_dataset_seconds = float(profiles["full_dataset"]["measured_training_seconds"])
        profiles["none"]["preload_seconds"] = 0.0
        profiles["full_dataset"]["preload_seconds"] = preload_seconds
        profiles["none"]["training_cycle_seconds_including_preload"] = (
            no_preload_seconds / expected_cycles
        )
        profiles["full_dataset"]["training_cycle_seconds_including_preload"] = (
            preload_seconds + full_dataset_seconds
        ) / expected_cycles

        no_preload_mean = no_preload_seconds / expected_cycles
        full_dataset_mean = full_dataset_seconds / expected_cycles
        training_only_speedup = 1.0 - full_dataset_seconds / no_preload_seconds
        speedup_including_preload = 1.0 - (preload_seconds + full_dataset_seconds) / no_preload_seconds
        cycles_to_recover_preload_time = (
            math.ceil(preload_seconds / (no_preload_mean - full_dataset_mean))
            if full_dataset_mean < no_preload_mean
            else None
        )
        training_only_speedups.append(training_only_speedup)
        speedups_including_preload.append(speedup_including_preload)
        repeats[repeat] = {
            "runs": profiles,
            "preload_artifact": preload_artifact,
            "training_only_speedup": training_only_speedup,
            "speedup_including_preload": speedup_including_preload,
            "cycles_to_recover_preload_time": cycles_to_recover_preload_time,
        }

    recovery_cycles = [
        repeat["cycles_to_recover_preload_time"]
        for repeat in repeats.values()
        if repeat["cycles_to_recover_preload_time"] is not None
    ]
    return {
        "note": "Cycles are repeated observations within a run, not independent repeats.",
        "repeat_count": len(repeats),
        "expected_cycles": expected_cycles,
        "expected_responses_per_cycle": expected_responses,
        "optimizer_steps_per_cycle": optimizer_steps_per_cycle,
        "repeats": repeats,
        "comparisons": {
            "training_only": {
                "baseline": {"preload": "none"},
                "candidate": {"preload": "full_dataset"},
                "speedup": _summarize_comparison(training_only_speedups),
            },
            "including_preload": {
                "baseline": {"preload": "none"},
                "candidate": {"preload": "full_dataset"},
                "speedup": _summarize_comparison(speedups_including_preload),
            },
        },
        "mean_cycles_to_recover_preload_time": (
            statistics.fmean(recovery_cycles) if recovery_cycles else None
        ),
    }


def write_full_dataset_preload_artifact(
    output_path: str | Path, analysis: Mapping[str, object]
) -> dict[str, object]:
    artifact = {"schema_version": 2, **analysis}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _parse_run(value: str) -> tuple[str, str, Path]:
    identity, separator, raw_path = value.partition("=")
    repeat, setting_separator, preload = identity.partition(":")
    if (
        not separator
        or not setting_separator
        or not repeat
        or preload not in _PRELOAD_SETTINGS
        or not raw_path
    ):
        raise argparse.ArgumentTypeError(
            "runs must use REPEAT:PRELOAD=PATH with PRELOAD in none,full_dataset"
        )
    return repeat, preload, Path(raw_path)


def _parse_preload(value: str) -> tuple[str, Path]:
    repeat, separator, raw_path = value.partition("=")
    if not separator or not repeat or not raw_path:
        raise argparse.ArgumentTypeError("preload artifacts must use REPEAT=PATH")
    return repeat, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--preload-artifact", action="append", required=True, type=_parse_preload)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-cycles", type=int, default=22)
    parser.add_argument("--expected-responses", type=int, default=64)
    parser.add_argument("--optimizer-steps-per-cycle", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    repeat_runs: dict[str, dict[str, Path]] = {}
    for repeat, preload, path in args.run:
        if preload in repeat_runs.setdefault(repeat, {}):
            raise ValueError(f"duplicate full-dataset preload run for {repeat}:{preload}")
        repeat_runs[repeat][preload] = path
    preload_artifacts = dict(args.preload_artifact)
    if len(preload_artifacts) != len(args.preload_artifact):
        raise ValueError("preload repeat names must be unique")
    analysis = measure_full_dataset_preload(
        repeat_runs,
        preload_artifacts,
        expected_cycles=args.expected_cycles,
        expected_responses=args.expected_responses,
        optimizer_steps_per_cycle=args.optimizer_steps_per_cycle,
    )
    return write_full_dataset_preload_artifact(args.output, analysis)


if __name__ == "__main__":
    main()
