#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize Qwen3-0.6B fully_async profiling artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


TRACE_SUFFIXES = (".trace.json", ".trace.json.gz", ".pt.trace.json", ".pt.trace.json.gz")
MARKER_PATTERNS = [
    re.compile(r"wandb.*offline", re.IGNORECASE),
    re.compile(r"sgl_kernel", re.IGNORECASE),
    re.compile(r"TimelineTrace adapter enabled", re.IGNORECASE),
    re.compile(r"Starting SGLang profiling", re.IGNORECASE),
    re.compile(r"PyTorch profiler .* enabled", re.IGNORECASE),
    re.compile(r"Actor training completed step", re.IGNORECASE),
    re.compile(r"Job '.+' succeeded", re.IGNORECASE),
    re.compile(r"Traceback|Exception|Error|failed", re.IGNORECASE),
]


@dataclass(frozen=True)
class TraceFile:
    path: Path
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", required=True, type=Path, help="Profiling artifact root.")
    return parser.parse_args()


def read_launch_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def load_json_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        trace_events = payload.get("traceEvents", [])
        if isinstance(trace_events, list):
            return [event for event in trace_events if isinstance(event, dict)]
    raise ValueError(f"Unsupported trace JSON format: {path}")


def collect_timeline_events(timeline_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    timeline_files = sorted(timeline_dir.glob("timeline_step_*.json"))
    events: list[dict[str, Any]] = []
    for path in timeline_files:
        events.extend(load_json_events(path))
    return timeline_files, events


def group_event_durations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    durations_by_name: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", "<unnamed>"))
        duration_us = event.get("dur")
        if isinstance(duration_us, (int, float)):
            durations_by_name[name].append(float(duration_us) / 1000.0)

    rows = []
    for name, durations_ms in durations_by_name.items():
        rows.append(
            {
                "name": name,
                "count": len(durations_ms),
                "total_ms": sum(durations_ms),
                "avg_ms": mean(durations_ms),
                "max_ms": max(durations_ms),
            }
        )
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)


def event_step(event: dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict) and "step" in args:
        return str(args["step"])
    return "unknown"


def collect_async_gaps(events: list[dict[str, Any]], min_gap_ms: float = 1.0) -> list[dict[str, Any]]:
    events_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        ts = event.get("ts")
        dur = event.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        events_by_step[event_step(event)].append(event)

    gaps = []
    for step, step_events in events_by_step.items():
        ordered = sorted(step_events, key=lambda event: float(event["ts"]))
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = float(previous["ts"]) + float(previous["dur"])
            current_start = float(current["ts"])
            gap_ms = (current_start - previous_end) / 1000.0
            if gap_ms < min_gap_ms:
                continue
            gaps.append(
                {
                    "step": step,
                    "gap_ms": gap_ms,
                    "after": str(previous.get("name", "<unnamed>")),
                    "before": str(current.get("name", "<unnamed>")),
                }
            )
    return sorted(gaps, key=lambda row: row["gap_ms"], reverse=True)


def is_trace_file(path: Path) -> bool:
    text = path.name
    return any(text.endswith(suffix) for suffix in TRACE_SUFFIXES)


def collect_trace_files(root: Path) -> list[TraceFile]:
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_trace_file(path):
            files.append(TraceFile(path=path, size_bytes=path.stat().st_size))
    return files


def resolve_train_trace_dir(profile_root: Path, launch_env: dict[str, str]) -> Path:
    tb_name = launch_env.get("RELAX_TB_EXPERIMENT_NAME")
    if tb_name:
        tb_path = Path(tb_name)
        if tb_path.is_absolute():
            return tb_path / "train_trace"
        return profile_root / "traces" / tb_name / "train_trace"
    trace_dirs = sorted((profile_root / "traces").glob("*/train_trace"))
    if trace_dirs:
        return trace_dirs[0]
    return profile_root / "traces" / "unknown" / "train_trace"


def resolve_sglang_trace_dir(profile_root: Path, launch_env: dict[str, str]) -> Path:
    output_dir = launch_env.get("PASS2_RELAX_SGLANG_PROFILE_OUTPUT_DIR") or launch_env.get(
        "RELAX_SGLANG_PROFILE_OUTPUT_DIR"
    )
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else profile_root / path
    return profile_root / "sglang_trace"


def extract_run_markers(run_log: Path, max_markers: int = 40) -> list[str]:
    markers = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(pattern.search(line) for pattern in MARKER_PATTERNS):
            markers.append(line.strip())
    return markers[-max_markers:]


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_None found._"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(profile_root: Path) -> tuple[str, list[str]]:
    if not profile_root.exists():
        raise FileNotFoundError(f"Profile root does not exist: {profile_root}")

    run_log = profile_root / "run.log"
    launch_env_path = profile_root / "launch.env"
    if not run_log.is_file():
        raise FileNotFoundError(f"Expected run log is missing: {run_log}")
    if not launch_env_path.is_file():
        raise FileNotFoundError(f"Expected launch env file is missing: {launch_env_path}")

    launch_env = read_launch_env(launch_env_path)
    profile_pass = launch_env.get("RELAX_PROFILE_PASS", "unknown")
    timeline_dir = Path(launch_env.get("RELAX_TIMELINE_DUMP_DIR", str(profile_root / "timeline")))
    if not timeline_dir.is_absolute():
        timeline_dir = profile_root / timeline_dir
    train_trace_dir = resolve_train_trace_dir(profile_root, launch_env)
    sglang_trace_dir = resolve_sglang_trace_dir(profile_root, launch_env)

    warnings = []
    if not timeline_dir.is_dir():
        raise FileNotFoundError(f"Expected timeline directory is missing: {timeline_dir}")

    timeline_files, timeline_events = collect_timeline_events(timeline_dir)
    if not timeline_files and profile_pass in {"timeline", "both"}:
        raise FileNotFoundError(f"No timeline_step_*.json files found in expected timeline directory: {timeline_dir}")
    if not timeline_events:
        warnings.append("No timeline events were loaded from timeline files.")

    duration_rows = group_event_durations(timeline_events)
    gap_rows = collect_async_gaps(timeline_events)
    train_traces = collect_trace_files(train_trace_dir)
    sglang_traces = collect_trace_files(sglang_trace_dir)
    run_markers = extract_run_markers(run_log)

    if profile_pass in {"focused", "both"} and not train_traces:
        warnings.append(f"No training profiler trace files found under {train_trace_dir}.")
    if profile_pass in {"focused", "both"} and not sglang_traces:
        warnings.append(f"No SGLang profiler trace files found under {sglang_trace_dir}.")
    if not run_markers:
        warnings.append("No run-log markers matched the summarizer patterns.")

    lines = [
        "# Fully Async Profiling Report",
        "",
        "## Launch Configuration",
        "",
        f"- Profile root: `{profile_root}`",
        f"- Profile pass: `{profile_pass}`",
        f"- Run log: `{rel(run_log, profile_root)}`",
        f"- Launch env: `{rel(launch_env_path, profile_root)}`",
        f"- Timeline dir: `{rel(timeline_dir, profile_root)}`",
        f"- Train trace dir: `{rel(train_trace_dir, profile_root)}`",
        f"- SGLang trace dir: `{rel(sglang_trace_dir, profile_root)}`",
        f"- W&B mode: `{launch_env.get('WANDB_MODE', '<unset>')}`",
        f"- SGL kernel path: `{launch_env.get('RELAX_SGL_KERNEL_BUILD_DIR', '<unset>')}`",
        "",
        "## Artifact Index",
        "",
        f"- Timeline files: `{len(timeline_files)}`",
        f"- Timeline events: `{len(timeline_events)}`",
        f"- Training trace files: `{len(train_traces)}`",
        f"- SGLang trace files: `{len(sglang_traces)}`",
        "",
        "## Timeline Event Durations",
        "",
    ]
    lines.extend(
        table(
            ["Event", "Count", "Total ms", "Avg ms", "Max ms"],
            [
                [
                    str(row["name"]),
                    str(row["count"]),
                    f"{row['total_ms']:.2f}",
                    f"{row['avg_ms']:.2f}",
                    f"{row['max_ms']:.2f}",
                ]
                for row in duration_rows[:30]
            ],
        )
    )
    lines.extend(["", "## Step-Level Async Gaps", ""])
    lines.extend(
        table(
            ["Step", "Gap ms", "After", "Before"],
            [
                [row["step"], f"{row['gap_ms']:.2f}", row["after"], row["before"]]
                for row in gap_rows[:30]
            ],
        )
    )
    lines.extend(["", "## Training Trace Files", ""])
    lines.extend(
        table(
            ["Path", "Size"],
            [[rel(trace.path, profile_root), format_size(trace.size_bytes)] for trace in train_traces[:50]],
        )
    )
    lines.extend(["", "## SGLang Trace Files", ""])
    lines.extend(
        table(
            ["Path", "Size"],
            [[rel(trace.path, profile_root), format_size(trace.size_bytes)] for trace in sglang_traces[:50]],
        )
    )
    lines.extend(["", "## Run Log Markers", ""])
    if run_markers:
        lines.extend([f"- `{marker}`" for marker in run_markers])
    else:
        lines.append("_None found._")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Actionable Next Checks",
            "",
            "Observed facts:",
        ]
    )
    if gap_rows:
        largest = gap_rows[0]
        lines.append(
            f"- Largest visible timeline gap is {largest['gap_ms']:.2f} ms on step {largest['step']} "
            f"between `{largest['after']}` and `{largest['before']}`."
        )
    else:
        lines.append("- No timeline gaps above 1.00 ms were detected between adjacent complete events.")
    lines.append(f"- Timeline coverage contains {len(timeline_events)} events across {len(timeline_files)} files.")
    lines.append(
        "- Train/SGLang trace coverage contains "
        f"{len(train_traces)} and {len(sglang_traces)} files respectively."
    )
    lines.extend(["", "Hypotheses to verify next:"])
    if warnings:
        lines.append("- Fix missing artifact coverage before drawing kernel-level conclusions.")
    if gap_rows:
        lines.append("- Inspect the largest gap in Perfetto or `chrome://tracing` before adding new timers.")
    if train_traces:
        lines.append(
            "- Open the training traces in TensorBoard profiler to identify actor/log-prob operator hotspots."
        )
    if sglang_traces:
        lines.append(
            "- Open the SGLang traces by rollout step to separate prefill/decode kernel time from orchestration wait."
        )
    if not train_traces and not sglang_traces:
        lines.append(
            "- Run the focused pass to collect operator-level traces after the timeline-only pass identifies gaps."
        )

    return "\n".join(lines) + "\n", warnings


def main() -> None:
    args = parse_args()
    profile_root = args.profile_root.resolve()
    report, warnings = build_report(profile_root)
    report_path = profile_root / "profile_report.md"
    report_path.write_text(report, encoding="utf-8")
    if warnings:
        print(f"Wrote {report_path} with {len(warnings)} warning(s).")
    else:
        print(f"Wrote {report_path}.")


if __name__ == "__main__":
    main()
