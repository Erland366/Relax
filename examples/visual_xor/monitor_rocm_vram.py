# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run one command while recording per-device ROCm VRAM peaks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


_TOTAL_KEY = "VRAM Total Memory (B)"
_USED_KEY = "VRAM Total Used Memory (B)"


def parse_rocm_smi_vram_csv(output: str) -> dict[str, dict[str, int]]:
    """Parse ``rocm-smi --showmeminfo vram --csv`` output."""
    rows = csv.DictReader(line for line in output.splitlines() if line.strip())
    parsed = {}
    for row in rows:
        device = row.get("device")
        if not device:
            continue
        try:
            parsed[device] = {
                "total_bytes": int(row[_TOTAL_KEY]),
                "used_bytes": int(row[_USED_KEY]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid ROCm VRAM row for {device}: {row}") from error
    if not parsed:
        raise ValueError("ROCm VRAM output contained no device rows")
    return parsed


def summarize_vram_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize baseline, final, per-device peak, and simultaneous total peak."""
    if not samples:
        raise ValueError("At least one VRAM sample is required")
    devices = sorted(samples[0]["devices"])
    if any(sorted(sample["devices"]) != devices for sample in samples):
        raise ValueError("ROCm device set changed while monitoring VRAM")

    device_summary = {}
    for device in devices:
        baseline = samples[0]["devices"][device]["used_bytes"]
        final = samples[-1]["devices"][device]["used_bytes"]
        peak = max(sample["devices"][device]["used_bytes"] for sample in samples)
        device_summary[device] = {
            "total_bytes": samples[0]["devices"][device]["total_bytes"],
            "baseline_used_bytes": baseline,
            "peak_used_bytes": peak,
            "peak_delta_bytes": peak - baseline,
            "final_used_bytes": final,
        }

    total_used_by_sample = [
        sum(sample["devices"][device]["used_bytes"] for device in devices) for sample in samples
    ]
    return {
        "devices": device_summary,
        "baseline_total_used_bytes": total_used_by_sample[0],
        "peak_simultaneous_total_used_bytes": max(total_used_by_sample),
        "peak_simultaneous_total_delta_bytes": max(total_used_by_sample) - total_used_by_sample[0],
        "final_total_used_bytes": total_used_by_sample[-1],
    }


def read_rocm_vram() -> dict[str, dict[str, int]]:
    completed = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_rocm_smi_vram_csv(completed.stdout)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def monitor_command(*, command: Sequence[str], label: str, interval_seconds: float) -> dict[str, Any]:
    if not command:
        raise ValueError("A command is required after --")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    samples = [{"elapsed_seconds": 0.0, "devices": read_rocm_vram()}]
    process = subprocess.Popen(list(command))
    try:
        while process.poll() is None:
            time.sleep(interval_seconds)
            samples.append(
                {
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                    "devices": read_rocm_vram(),
                }
            )
    except BaseException:
        process.terminate()
        process.wait()
        raise

    samples.append(
        {
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "devices": read_rocm_vram(),
        }
    )
    return {
        "schema_version": 1,
        "label": label,
        "command": list(command),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "sample_interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "exit_code": process.returncode,
        **summarize_vram_samples(samples),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = monitor_command(
        command=args.command,
        label=args.label,
        interval_seconds=args.interval_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
