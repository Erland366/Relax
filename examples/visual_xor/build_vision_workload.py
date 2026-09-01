# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build an ordered low/high-resolution visual-XOR workload."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], bytes):
        return value[0]
    raise TypeError(f"Expected one encoded image byte string, got {type(value).__name__}")


def _resize_png(value: Any, resolution: int) -> list[bytes]:
    with Image.open(io.BytesIO(_image_bytes(value))) as image:
        resized = image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        resized.save(output, format="PNG")
    return [output.getvalue()]


def _validate_balanced_group(group: pd.DataFrame, *, name: str) -> None:
    if "label" not in group:
        raise ValueError("Vision workload input is missing the 'label' column")
    counts = group["label"].value_counts().to_dict()
    if set(counts) != {"A", "B"} or counts["A"] != counts["B"]:
        raise ValueError(f"Vision workload {name} group is not A/B balanced: {counts}")


def _processed_image_shape(checkpoint: str | Path, prompt: str, image_bytes: bytes) -> tuple[int, int, int]:
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        processed = processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
    grid = processed["image_grid_thw"]
    merge_size = int(config.vision_config.spatial_merge_size)
    visual_tokens = int(grid.prod(dim=1).sum()) // (merge_size**2)
    stream_count = 1 + len(config.vision_config.deepstack_visual_indexes)
    feature_bytes = visual_tokens * int(config.text_config.hidden_size) * stream_count * 2
    return visual_tokens, feature_bytes, int(grid.numel() * grid.element_size())


def build_workload(
    *,
    input_data: str | Path,
    output_data: str | Path,
    cycle_output: str | Path,
    checkpoint: str | Path,
    low_resolution: int,
    high_resolution: int,
    rollout_batch_size: int,
    num_rollout: int,
    low_overlap_seconds: float,
    high_overlap_seconds: float,
) -> dict[str, Any]:
    if low_resolution <= 0 or high_resolution <= low_resolution:
        raise ValueError("Workload resolutions must satisfy 0 < low_resolution < high_resolution")
    if rollout_batch_size <= 0 or num_rollout <= 0:
        raise ValueError("rollout_batch_size and num_rollout must be positive")
    if low_overlap_seconds < 0 or high_overlap_seconds < 0:
        raise ValueError("Workload overlap seconds must be non-negative")
    input_path = Path(input_data)
    if not input_path.is_file():
        raise FileNotFoundError(f"Vision workload input does not exist: {input_path}")
    frame = pd.read_parquet(input_path)
    required_rows = 2 * rollout_batch_size
    if len(frame) < required_rows:
        raise ValueError(f"Vision workload requires at least {required_rows} rows, got {len(frame)}")
    frame = frame.iloc[:required_rows].copy()
    low_group = frame.iloc[:rollout_batch_size]
    high_group = frame.iloc[rollout_batch_size:required_rows]
    _validate_balanced_group(low_group, name="low-resolution")
    _validate_balanced_group(high_group, name="high-resolution")

    frame.loc[low_group.index, "image"] = low_group["image"].map(
        lambda value: _resize_png(value, low_resolution)
    )
    frame.loc[high_group.index, "image"] = high_group["image"].map(
        lambda value: _resize_png(value, high_resolution)
    )
    for index in low_group.index:
        frame.at[index, "metadata"] = {
            **dict(frame.at[index, "metadata"]),
            "vision_workload": "low",
            "vision_resolution": low_resolution,
        }
    for index in high_group.index:
        frame.at[index, "metadata"] = {
            **dict(frame.at[index, "metadata"]),
            "vision_workload": "high",
            "vision_resolution": high_resolution,
        }

    low_tokens, low_feature_bytes, low_grid_bytes = _processed_image_shape(
        checkpoint,
        str(frame.iloc[0]["prompt"]),
        _image_bytes(frame.iloc[0]["image"]),
    )
    high_tokens, high_feature_bytes, high_grid_bytes = _processed_image_shape(
        checkpoint,
        str(frame.iloc[rollout_batch_size]["prompt"]),
        _image_bytes(frame.iloc[rollout_batch_size]["image"]),
    )
    workloads = {
        "low": {
            "image_count": rollout_batch_size,
            "visual_tokens": rollout_batch_size * low_tokens,
            "feature_bytes": rollout_batch_size * (low_feature_bytes + low_grid_bytes),
            "overlap_seconds": low_overlap_seconds,
        },
        "high": {
            "image_count": rollout_batch_size,
            "visual_tokens": rollout_batch_size * high_tokens,
            "feature_bytes": rollout_batch_size * (high_feature_bytes + high_grid_bytes),
            "overlap_seconds": high_overlap_seconds,
        },
    }
    cycles = {str(cycle): workloads["low" if cycle % 2 == 0 else "high"] for cycle in range(num_rollout)}
    output_path = Path(output_data)
    cycle_path = Path(cycle_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    cycle_path.write_text(json.dumps(cycles, indent=2) + "\n")
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "cycles": str(cycle_path.resolve()),
        "rows": len(frame),
        "low_resolution": low_resolution,
        "high_resolution": high_resolution,
        "low_visual_tokens_per_image": low_tokens,
        "high_visual_tokens_per_image": high_tokens,
        "workloads": workloads,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles-output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--low-resolution", type=int, default=112)
    parser.add_argument("--high-resolution", type=int, default=448)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--num-rollout", type=int, default=24)
    parser.add_argument("--low-overlap-seconds", type=float, required=True)
    parser.add_argument("--high-overlap-seconds", type=float, required=True)
    parser.add_argument("--manifest-output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_workload(
        input_data=args.input,
        output_data=args.output,
        cycle_output=args.cycles_output,
        checkpoint=args.checkpoint,
        low_resolution=args.low_resolution,
        high_resolution=args.high_resolution,
        rollout_batch_size=args.rollout_batch_size,
        num_rollout=args.num_rollout,
        low_overlap_seconds=args.low_overlap_seconds,
        high_overlap_seconds=args.high_overlap_seconds,
    )
    if args.manifest_output:
        manifest = Path(args.manifest_output)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
