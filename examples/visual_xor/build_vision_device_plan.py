# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a CPU/GPU vision-device plan from measured times and cycle workloads."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from relax.engine.rollout.vision_device import (
    VISION_DEVICE_PLAN_SCHEMA_VERSION,
    VisionTimeModel,
    VisionWorkload,
    load_vision_device_plan,
)


_PREDICTORS = (
    "intercept_seconds",
    "seconds_per_image",
    "seconds_per_visual_token",
    "seconds_per_feature_byte",
)


def _load_json_lines(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Vision timing measurements do not exist: {source}")
    rows = []
    for line_number, raw_line in enumerate(source.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {source}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Measurement on {source}:{line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"Vision timing measurement file is empty: {source}")
    return rows


def _finite_nonnegative(row: dict[str, Any], name: str, *, row_index: int) -> float:
    if name not in row:
        raise ValueError(f"Measurement row {row_index} is missing {name!r}")
    try:
        value = float(row[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Measurement row {row_index} field {name!r} is not numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Measurement row {row_index} field {name!r} must be finite and non-negative, got {value}"
        )
    return value


def _design_row(row: dict[str, Any], *, row_index: int) -> list[float]:
    return [
        1.0,
        _finite_nonnegative(row, "image_count", row_index=row_index),
        _finite_nonnegative(row, "visual_tokens", row_index=row_index),
        _finite_nonnegative(row, "feature_bytes", row_index=row_index),
    ]


def _fit_nonnegative_least_squares(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Solve this at-most-four-variable NNLS problem by exhaustive active-set search."""
    best_coefficients = np.zeros(design.shape[1], dtype=np.float64)
    best_residual = float(np.square(targets).sum())
    for active_count in range(1, design.shape[1] + 1):
        for active in itertools.combinations(range(design.shape[1]), active_count):
            active_coefficients, *_ = np.linalg.lstsq(design[:, active], targets, rcond=None)
            if bool(np.any(active_coefficients < -1e-12)):
                continue
            coefficients = np.zeros(design.shape[1], dtype=np.float64)
            coefficients[list(active)] = np.maximum(active_coefficients, 0.0)
            residual = float(np.square(targets - design @ coefficients).sum())
            if residual < best_residual:
                best_residual = residual
                best_coefficients = coefficients
    return best_coefficients


def _independent_predictor_columns(design: np.ndarray) -> list[int]:
    """Select a stable independent predictor subset in declared order."""
    active = []
    rank = 0
    for column in range(design.shape[1]):
        candidate = active + [column]
        candidate_rank = int(np.linalg.matrix_rank(design[:, candidate]))
        if candidate_rank > rank:
            active = candidate
            rank = candidate_rank
    return active


def _fit_device(
    rows: Sequence[dict[str, Any]], device: str
) -> tuple[dict[str, float], dict[str, float | int]]:
    device_rows = [row for row in rows if row.get("device") == device and row.get("split", "fit") == "fit"]
    if len(device_rows) < 2:
        raise ValueError(f"Device {device!r} requires at least 2 fit rows, got {len(device_rows)}")
    design = np.asarray(
        [_design_row(row, row_index=index) for index, row in enumerate(device_rows, start=1)],
        dtype=np.float64,
    )
    active_columns = _independent_predictor_columns(design)
    if len(active_columns) < 2:
        raise ValueError(f"Device {device!r} needs at least two distinct workload shapes")
    targets = np.asarray(
        [
            _finite_nonnegative(row, "seconds", row_index=index)
            for index, row in enumerate(device_rows, start=1)
        ],
        dtype=np.float64,
    )
    active_coefficients = _fit_nonnegative_least_squares(design[:, active_columns], targets)
    coefficients = np.zeros(len(_PREDICTORS), dtype=np.float64)
    coefficients[active_columns] = active_coefficients
    predictions = design @ coefficients
    residuals = targets - predictions
    total_variance = float(np.square(targets - targets.mean()).sum())
    residual_variance = float(np.square(residuals).sum())
    nonzero_targets = targets > 0
    mape = (
        float(np.mean(np.abs(residuals[nonzero_targets] / targets[nonzero_targets])))
        if bool(nonzero_targets.any())
        else 0.0
    )
    return (
        dict(zip(_PREDICTORS, (float(value) for value in coefficients), strict=True)),
        {
            "fit_row_count": len(device_rows),
            "design_rank": len(active_columns),
            "active_predictors": [_PREDICTORS[index] for index in active_columns],
            "dropped_collinear_predictors": [
                predictor for index, predictor in enumerate(_PREDICTORS) if index not in active_columns
            ],
            "mae_seconds": float(np.mean(np.abs(residuals))),
            "rmse_seconds": float(np.sqrt(np.mean(np.square(residuals)))),
            "mape": mape,
            "r_squared": 1.0 - residual_variance / total_variance if total_variance else 1.0,
        },
    )


def _load_cycles(path: str | Path) -> dict[str, dict[str, int | float]]:
    cycle_path = Path(path)
    if not cycle_path.is_file():
        raise FileNotFoundError(f"Vision workload file does not exist: {cycle_path}")
    try:
        payload = json.loads(cycle_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision workload file is not valid JSON: {cycle_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Vision workload file must be a JSON object keyed by rollout ID")
    cycles = {}
    for raw_rollout_id, raw_workload in payload.items():
        if not isinstance(raw_workload, dict):
            raise ValueError(f"Cycle {raw_rollout_id!r} must contain an object")
        workload = VisionWorkload.from_mapping(raw_workload, field=f"cycles.{raw_rollout_id}")
        cycles[str(int(raw_rollout_id))] = {
            "image_count": workload.image_count,
            "visual_tokens": workload.visual_tokens,
            "feature_bytes": workload.feature_bytes,
            "overlap_seconds": workload.overlap_seconds,
        }
    return cycles


def build_plan(
    *,
    measurements: str | Path,
    cycles: str | Path,
    minimum_gap: float,
    initial_device: str,
) -> dict[str, Any]:
    rows = _load_json_lines(measurements)
    unknown_devices = sorted({row.get("device") for row in rows} - {"cpu", "gpu"})
    if unknown_devices:
        raise ValueError(f"Measurements contain unsupported devices: {unknown_devices}")
    if not math.isfinite(minimum_gap) or not 0 <= minimum_gap < 1:
        raise ValueError("minimum_gap must be finite and in [0, 1)")
    if initial_device not in {"cpu", "gpu"}:
        raise ValueError("initial_device must be 'cpu' or 'gpu'")

    gpu_time_model, gpu_timing_model_fit = _fit_device(rows, "gpu")
    cpu_time_model, cpu_timing_model_fit = _fit_device(rows, "cpu")
    return {
        "schema_version": VISION_DEVICE_PLAN_SCHEMA_VERSION,
        "initial_device": initial_device,
        "minimum_gap": minimum_gap,
        "time_models": {"gpu": gpu_time_model, "cpu": cpu_time_model},
        "cycles": _load_cycles(cycles),
        "timing_model_fit": {"gpu": gpu_timing_model_fit, "cpu": cpu_timing_model_fit},
        "provenance": {
            "measurements": str(Path(measurements).resolve()),
            "cycles": str(Path(cycles).resolve()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="Measured CPU/GPU times in JSONL format.")
    parser.add_argument("--cycles", required=True, help="JSON workload map keyed by rollout ID.")
    parser.add_argument("--output", required=True, help="Output vision-device plan.")
    parser.add_argument("--minimum-gap", type=float, default=0.05)
    parser.add_argument("--initial-device", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        measurements=args.measurements,
        cycles=args.cycles,
        minimum_gap=args.minimum_gap,
        initial_device=args.initial_device,
    )
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    load_vision_device_plan(output)


if __name__ == "__main__":
    main()
