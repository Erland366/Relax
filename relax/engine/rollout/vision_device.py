# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Choose where to run a frozen Qwen3-VL vision encoder for each rollout cycle."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VISION_DEVICE_PLAN_SCHEMA_VERSION = 2
VISION_DEVICES = ("cpu", "gpu")


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number, got {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number, got {value!r}")
    return number


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.isdecimal():
        number = int(value)
    else:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    if number < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return number


@dataclass(frozen=True)
class VisionWorkload:
    """Vision work expected during one rollout cycle."""

    image_count: int
    visual_tokens: int
    feature_bytes: int
    overlap_seconds: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, field: str) -> "VisionWorkload":
        required = {"image_count", "visual_tokens", "feature_bytes", "overlap_seconds"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"{field} is missing required field(s): {missing}")
        return cls(
            image_count=_nonnegative_int(payload["image_count"], field=f"{field}.image_count"),
            visual_tokens=_nonnegative_int(payload["visual_tokens"], field=f"{field}.visual_tokens"),
            feature_bytes=_nonnegative_int(payload["feature_bytes"], field=f"{field}.feature_bytes"),
            overlap_seconds=_finite_nonnegative(payload["overlap_seconds"], field=f"{field}.overlap_seconds"),
        )


@dataclass(frozen=True)
class VisionTimeModel:
    """Non-negative linear estimate of vision-encoder time."""

    intercept_seconds: float
    seconds_per_image: float
    seconds_per_visual_token: float
    seconds_per_feature_byte: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, field: str) -> "VisionTimeModel":
        supported = {
            "intercept_seconds",
            "seconds_per_image",
            "seconds_per_visual_token",
            "seconds_per_feature_byte",
        }
        unknown = sorted(payload.keys() - supported)
        if unknown:
            raise ValueError(f"{field} contains unsupported coefficient(s): {unknown}")
        return cls(
            **{
                name: _finite_nonnegative(payload.get(name, 0.0), field=f"{field}.{name}")
                for name in supported
            }
        )

    def predict(self, workload: VisionWorkload) -> float:
        return (
            self.intercept_seconds
            + self.seconds_per_image * workload.image_count
            + self.seconds_per_visual_token * workload.visual_tokens
            + self.seconds_per_feature_byte * workload.feature_bytes
        )


@dataclass(frozen=True)
class VisionDeviceChoice:
    """The chosen vision device and the timing estimates behind that choice."""

    rollout_id: int
    device: str
    reason: str
    workload: VisionWorkload
    predicted_gpu_seconds: float
    predicted_cpu_seconds: float
    predicted_cpu_seconds_after_overlap: float
    predicted_gap: float
    minimum_gap: float

    def to_metrics(self) -> dict[str, float | int]:
        return {
            "vision_device/cpu": int(self.device == "cpu"),
            "vision_device/gpu": int(self.device == "gpu"),
            "vision_device/image_count": self.workload.image_count,
            "vision_device/visual_tokens": self.workload.visual_tokens,
            "vision_device/feature_bytes": self.workload.feature_bytes,
            "vision_device/overlap_seconds": self.workload.overlap_seconds,
            "vision_device/predicted_gpu_seconds": self.predicted_gpu_seconds,
            "vision_device/predicted_cpu_seconds": self.predicted_cpu_seconds,
            "vision_device/predicted_cpu_seconds_after_overlap": self.predicted_cpu_seconds_after_overlap,
            "vision_device/predicted_gap": self.predicted_gap,
            "vision_device/minimum_gap": self.minimum_gap,
        }

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "schema_version": VISION_DEVICE_PLAN_SCHEMA_VERSION,
            "rollout_id": self.rollout_id,
            "device": self.device,
            "reason": self.reason,
            "workload": {
                "image_count": self.workload.image_count,
                "visual_tokens": self.workload.visual_tokens,
                "feature_bytes": self.workload.feature_bytes,
                "overlap_seconds": self.workload.overlap_seconds,
            },
            "predicted_gpu_seconds": self.predicted_gpu_seconds,
            "predicted_cpu_seconds": self.predicted_cpu_seconds,
            "predicted_cpu_seconds_after_overlap": self.predicted_cpu_seconds_after_overlap,
            "predicted_gap": self.predicted_gap,
            "minimum_gap": self.minimum_gap,
        }


class VisionDevicePlan:
    """Choose the CPU or GPU from measured times and known cycle workloads."""

    def __init__(
        self,
        *,
        gpu_time_model: VisionTimeModel,
        cpu_time_model: VisionTimeModel,
        cycles: Mapping[int, VisionWorkload],
        minimum_gap: float,
        initial_device: str,
    ) -> None:
        if not cycles:
            raise ValueError("Vision device plan requires at least one rollout cycle")
        if initial_device not in VISION_DEVICES:
            raise ValueError(f"initial_device must be one of {VISION_DEVICES}, got {initial_device!r}")
        self.gpu_time_model = gpu_time_model
        self.cpu_time_model = cpu_time_model
        self.cycles = dict(cycles)
        self.minimum_gap = _finite_nonnegative(minimum_gap, field="minimum_gap")
        if self.minimum_gap >= 1:
            raise ValueError(f"minimum_gap must be less than 1, got {self.minimum_gap}")
        self._choices: dict[int, VisionDeviceChoice] = {}
        previous_device = initial_device
        for rollout_id in sorted(self.cycles):
            choice = self._choose(rollout_id, previous_device=previous_device)
            self._choices[rollout_id] = choice
            previous_device = choice.device

    def choose(self, rollout_id: int) -> VisionDeviceChoice:
        if rollout_id not in self.cycles:
            available = sorted(self.cycles)
            raise KeyError(
                f"Vision device plan has no workload for rollout_id={rollout_id}; "
                f"available rollout IDs are {available}"
            )
        return self._choices[rollout_id]

    def _choose(self, rollout_id: int, *, previous_device: str) -> VisionDeviceChoice:
        workload = self.cycles[rollout_id]
        predicted_gpu = self.gpu_time_model.predict(workload)
        predicted_cpu = self.cpu_time_model.predict(workload)
        predicted_cpu_after_overlap = max(0.0, predicted_cpu - workload.overlap_seconds)
        comparison_scale = max(predicted_gpu, predicted_cpu_after_overlap, 1e-12)
        predicted_gap = (predicted_gpu - predicted_cpu_after_overlap) / comparison_scale

        if predicted_gap > self.minimum_gap:
            device = "cpu"
            reason = "cpu_predicted_faster"
        elif predicted_gap < -self.minimum_gap:
            device = "gpu"
            reason = "gpu_predicted_faster"
        else:
            device = previous_device
            reason = "keep_previous_device"
        return VisionDeviceChoice(
            rollout_id=rollout_id,
            device=device,
            reason=reason,
            workload=workload,
            predicted_gpu_seconds=predicted_gpu,
            predicted_cpu_seconds=predicted_cpu,
            predicted_cpu_seconds_after_overlap=predicted_cpu_after_overlap,
            predicted_gap=predicted_gap,
            minimum_gap=self.minimum_gap,
        )


def load_vision_device_plan(path: str | Path, *, minimum_gap: float | None = None) -> VisionDevicePlan:
    """Load and validate a vision-device plan."""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Vision device plan does not exist: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision device plan is not valid JSON: {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Vision device plan must contain a JSON object: {plan_path}")
    schema_version = payload.get("schema_version")
    if schema_version != VISION_DEVICE_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported vision device plan schema_version: "
            f"expected {VISION_DEVICE_PLAN_SCHEMA_VERSION}, got {schema_version!r}"
        )
    time_models = payload.get("time_models")
    if not isinstance(time_models, dict):
        raise ValueError("Vision device plan requires a 'time_models' object")
    for device in VISION_DEVICES:
        if not isinstance(time_models.get(device), dict):
            raise ValueError(f"Vision device plan requires time_models.{device} coefficients")
    cycle_payloads = payload.get("cycles")
    if not isinstance(cycle_payloads, dict):
        raise ValueError("Vision device plan requires a 'cycles' object")
    cycles: dict[int, VisionWorkload] = {}
    for raw_rollout_id, workload_payload in cycle_payloads.items():
        if not isinstance(workload_payload, dict):
            raise ValueError(f"cycles.{raw_rollout_id} must be an object")
        rollout_id = _nonnegative_int(raw_rollout_id, field=f"cycles key {raw_rollout_id!r}")
        if rollout_id in cycles:
            raise ValueError(f"Duplicate rollout ID after integer normalization: {raw_rollout_id!r}")
        cycles[rollout_id] = VisionWorkload.from_mapping(workload_payload, field=f"cycles.{raw_rollout_id}")
    configured_gap = payload.get("minimum_gap", 0.05)
    selected_gap = configured_gap if minimum_gap is None else minimum_gap
    return VisionDevicePlan(
        gpu_time_model=VisionTimeModel.from_mapping(time_models["gpu"], field="time_models.gpu"),
        cpu_time_model=VisionTimeModel.from_mapping(time_models["cpu"], field="time_models.cpu"),
        cycles=cycles,
        minimum_gap=selected_gap,
        initial_device=payload.get("initial_device", "gpu"),
    )
