#!/usr/bin/env python3
"""Summarize Qwen3-0.6B fully_async profiling artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


_TIMESTAMP_RE = re.compile(r"\[(?P<ts>[^\]]+)\]")
_PHASE_RE = re.compile(r"launch_phase: (?P<name>.+)")
_RAY_JOB_RE = re.compile(r"raysubmit_[A-Za-z0-9]+")
_TRAINING_LOOP_RE = re.compile(r"training_loop end elapsed=(?P<seconds>[0-9.]+)s")


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip("'\"")
    return values


def _load_timeline_events(timeline_dir: Path) -> list[dict]:
    events = []
    for path in sorted(timeline_dir.glob("timeline_step_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("traceEvents", [])
        if not isinstance(payload, list):
            raise ValueError(f"Timeline file must contain a list or traceEvents object: {path}")
        for event in payload:
            if isinstance(event, dict):
                event = dict(event)
                event["_source_file"] = str(path)
                events.append(event)
    return events


def _event_identity(event: dict) -> tuple:
    args = event.get("args") or {}
    return (
        event.get("name"),
        event.get("ph"),
        event.get("ts"),
        event.get("dur"),
        event.get("pid"),
        event.get("tid"),
        event.get("cat"),
        args.get("step"),
    )


def _dedupe_timeline_events(events: list[dict]) -> list[dict]:
    """Timeline files are cumulative snapshots; report duplicated spans once."""
    seen = set()
    deduped = []
    for event in events:
        key = _event_identity(event)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _event_area(name: str) -> str:
    if name.startswith("advantages"):
        return "advantages"
    if name == "rollout":
        return "rollout"
    if name == "log_probs":
        return "actor_fwd"
    if "weight" in name or name == "recv_weight_fully_async":
        return "weight_sync"
    if name.startswith("train") or name == "actor_train" or name.startswith("get_data_batch_"):
        return "actor_train"
    if name == "init_actor":
        return "startup"
    return "other"


def _timeline_summary(events: list[dict]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name")
        dur = event.get("dur")
        if not name or dur is None:
            continue
        grouped[str(name)].append(float(dur) / 1_000_000.0)

    rows = []
    for name, durations in grouped.items():
        total = sum(durations)
        rows.append(
            {
                "name": name,
                "count": len(durations),
                "total_s": total,
                "avg_s": total / len(durations),
                "max_s": max(durations),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_s"]), reverse=True)


def _area_summary(events: list[dict]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name")
        dur = event.get("dur")
        if not name or dur is None:
            continue
        grouped[_event_area(str(name))].append(float(dur) / 1_000_000.0)

    rows = []
    for area, durations in grouped.items():
        total = sum(durations)
        rows.append(
            {
                "area": area,
                "count": len(durations),
                "total_s": total,
                "avg_s": total / len(durations),
                "max_s": max(durations),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_s"]), reverse=True)


def _step_area_summary(events: list[dict], *, min_step: int | None = None) -> list[dict[str, float | int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        step = args.get("step")
        if not isinstance(step, int):
            continue
        if min_step is not None and step < min_step:
            continue
        name = event.get("name")
        dur = event.get("dur")
        if not name or dur is None:
            continue
        grouped[(str(step), _event_area(str(name)))].append(float(dur) / 1_000_000.0)

    rows = []
    for (step, area), durations in grouped.items():
        total = sum(durations)
        rows.append({"step": step, "area": area, "count": len(durations), "total_s": total})
    return sorted(rows, key=lambda row: (int(str(row["step"])), -float(row["total_s"])))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _critical_path_summary(events: list[dict]) -> list[dict[str, float | int | str]]:
    by_step: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        args = event.get("args") or {}
        step = args.get("step", "unknown")
        if event.get("ph") == "X" and "ts" in event and "dur" in event:
            by_step[str(step)].append(event)

    rows = []
    for step, step_events in by_step.items():
        intervals = [
            (float(event["ts"]) / 1_000_000.0, (float(event["ts"]) + float(event.get("dur", 0))) / 1_000_000.0)
            for event in step_events
        ]
        merged = _merge_intervals(intervals)
        if not merged:
            continue
        wall_s = max(end for _, end in intervals) - min(start for start, _ in intervals)
        covered_s = sum(end - start for start, end in merged)
        idle_s = max(0.0, wall_s - covered_s)
        top_event = max(step_events, key=lambda event: float(event.get("dur", 0)))
        rows.append(
            {
                "step": step,
                "event_count": len(step_events),
                "wall_s": wall_s,
                "covered_s": covered_s,
                "idle_s": idle_s,
                "top_event": str(top_event.get("name", "unknown")),
                "top_event_s": float(top_event.get("dur", 0)) / 1_000_000.0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (str(row["step"]) == "unknown", int(row["step"]) if str(row["step"]).isdigit() else 10**9),
    )


def _step_gap_summary(events: list[dict]) -> list[dict[str, str | float]]:
    by_step: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        args = event.get("args") or {}
        step = args.get("step", "unknown")
        if "ts" in event and "dur" in event:
            by_step[str(step)].append(event)

    rows = []
    for step, step_events in by_step.items():
        ordered = sorted(step_events, key=lambda event: float(event["ts"]))
        for left, right in zip(ordered, ordered[1:], strict=False):
            left_end = float(left["ts"]) + float(left.get("dur", 0))
            gap_s = (float(right["ts"]) - left_end) / 1_000_000.0
            if gap_s <= 0:
                continue
            rows.append(
                {
                    "step": step,
                    "from": str(left.get("name", "unknown")),
                    "to": str(right.get("name", "unknown")),
                    "gap_s": gap_s,
                }
            )
    return sorted(rows, key=lambda row: float(row["gap_s"]), reverse=True)[:20]


def _parse_iso_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_wall_clock_summary(log_path: Path) -> dict[str, str]:
    summary = {
        "ray_job_submit_s": "",
        "training_loop_s": "",
        "first_actor_train": "",
        "main_success": "no",
    }
    if not log_path.exists():
        return summary

    phase_starts: dict[str, datetime] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts_match = _TIMESTAMP_RE.search(line)
        phase_match = _PHASE_RE.search(line)
        if ts_match and phase_match:
            phase_text = phase_match.group("name")
            ts = _parse_iso_timestamp(ts_match.group("ts"))
            if ts is not None and phase_text.endswith(" begin"):
                phase_starts[phase_text.removesuffix(" begin")] = ts
            elif ts is not None and phase_text.endswith(" end"):
                phase_name = phase_text.removesuffix(" end")
                started = phase_starts.get(phase_name)
                if started is not None and phase_name == "ray_job_submit":
                    summary["ray_job_submit_s"] = f"{(ts - started).total_seconds():.2f}"

        training_match = _TRAINING_LOOP_RE.search(line)
        if training_match:
            summary["training_loop_s"] = training_match.group("seconds")
        if "Actor training step" in line and not summary["first_actor_train"]:
            summary["first_actor_train"] = line.strip()
        if "Main func successfully" in line:
            summary["main_success"] = "yes"

    return summary


def _recommend_next_check(
    area_rows: list[dict[str, float | int | str]],
    steady_rows: list[dict[str, float | int]],
) -> str:
    candidates = steady_rows or area_rows
    if not candidates:
        return "insufficient evidence: no timeline events were available"

    top = max(candidates, key=lambda row: float(row["total_s"]))
    area = str(top.get("area", "unknown"))
    if area == "weight_sync":
        return "weight_sync: weight receive/update spans dominate warmup-excluded timing"
    if area == "advantages":
        return "orchestration: advantages time is dominated by data movement rather than compute"
    if area == "rollout":
        return "rollout: split request routing, prefill, decode, reward, and queue transfer next"
    if area == "actor_train":
        return "train: inspect train traces and data-batch spans before kernel tuning"
    return f"{area}: inspect this area first"


def _index_files(path: Path) -> list[dict[str, str | int]]:
    if not path.exists():
        return []
    return [
        {"path": str(file), "size_bytes": file.stat().st_size}
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]


def _extract_log_markers(log_path: Path) -> dict[str, list[str]]:
    markers = {
        "launch_phase": [],
        "ray_jobs": [],
        "progress": [],
        "profile": [],
        "errors": [],
    }
    if not log_path.exists():
        return markers

    seen_jobs = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "launch_phase:" in line:
            phase_match = _PHASE_RE.search(line)
            ts_match = _TIMESTAMP_RE.search(line)
            if phase_match:
                ts = ts_match.group("ts") if ts_match else "unknown-time"
                markers["launch_phase"].append(f"{ts} {phase_match.group('name')}")
        for job_id in _RAY_JOB_RE.findall(line):
            if job_id not in seen_jobs:
                seen_jobs.add(job_id)
                markers["ray_jobs"].append(job_id)
        if "Start rollout" in line or "Actor training step" in line or "Main func successfully" in line:
            markers["progress"].append(line.strip())
        if "profiler" in line.lower() or "profiling" in line.lower() or "timeline" in line.lower():
            markers["profile"].append(line.strip())
        if "Traceback" in line or "Error" in line or "Exception" in line or "failed" in line.lower():
            markers["errors"].append(line.strip())

    for key in markers:
        markers[key] = markers[key][-30:]
    return markers


def _expected_missing(expected_pass: str, profile_root: Path, train_files: list, sglang_files: list) -> list[str]:
    missing = []
    run_log = profile_root / "run.log"
    launch_env = profile_root / "launch.env"
    timeline_files = list((profile_root / "timeline").glob("timeline_step_*.json"))

    if not run_log.exists():
        missing.append("run.log")
    if not launch_env.exists():
        missing.append("launch.env")
    if expected_pass in {"timeline", "torch", "sglang", "all"} and not timeline_files:
        missing.append("timeline/timeline_step_*.json")
    if expected_pass in {"torch", "all"} and not train_files:
        missing.append("traces/<experiment>/train_trace/*")
    if expected_pass in {"sglang", "all"} and not sglang_files:
        missing.append("sglang_trace/*")
    return missing


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None found._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def build_report(profile_root: Path, expected_pass: str, run_status: str) -> tuple[str, list[str]]:
    env = _read_env(profile_root / "launch.env")
    raw_timeline_events = (
        _load_timeline_events(profile_root / "timeline") if (profile_root / "timeline").exists() else []
    )
    timeline_events = _dedupe_timeline_events(raw_timeline_events)
    timeline_rows = _timeline_summary(timeline_events)
    area_rows = _area_summary(timeline_events)
    step_area_rows = _step_area_summary(timeline_events)
    steady_area_rows = _step_area_summary(timeline_events, min_step=1)
    critical_path_rows = _critical_path_summary(timeline_events)
    gap_rows = _step_gap_summary(timeline_events)
    experiment = env.get("RELAX_TB_EXPERIMENT_NAME", "<unknown>")
    train_trace_dir = profile_root / "traces" / experiment / "train_trace"
    sglang_trace_dir = profile_root / "sglang_trace"
    train_files = _index_files(train_trace_dir)
    sglang_files = _index_files(sglang_trace_dir)
    markers = _extract_log_markers(profile_root / "run.log")
    wall_clock = _extract_wall_clock_summary(profile_root / "run.log")
    missing = _expected_missing(expected_pass, profile_root, train_files, sglang_files)
    recommendation = _recommend_next_check(area_rows, steady_area_rows)

    config_keys = [
        "PROFILE_PASS",
        "RELAX_TB_EXPERIMENT_NAME",
        "HIP_VISIBLE_DEVICES",
        "ACTOR_RESOURCE_GPUS",
        "ROLLOUT_RESOURCE_GPUS",
        "ACTOR_FWD_RESOURCE_GPUS",
        "MAX_STALENESS",
        "NUM_ROLLOUT",
        "NUM_STEPS_PER_ROLLOUT",
        "ROLLOUT_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT",
        "GLOBAL_BATCH_SIZE",
        "SEQ_LENGTH",
        "ROLLOUT_MAX_RESPONSE_LEN",
        "RELAX_TIMELINE_DUMP_DIR",
        "RELAX_USE_PYTORCH_PROFILER",
        "RELAX_PROFILE_TARGETS",
        "RELAX_SGLANG_PROFILE",
        "RELAX_SGLANG_PROFILE_OUTPUT_DIR",
    ]

    config_rows = [[key, env.get(key, "")] for key in config_keys if key in env]
    timeline_table = _format_table(
        ["Event", "Count", "Total s", "Avg s", "Max s"],
        [
            [
                str(row["name"]),
                str(row["count"]),
                f"{float(row['total_s']):.3f}",
                f"{float(row['avg_s']):.3f}",
                f"{float(row['max_s']):.3f}",
            ]
            for row in timeline_rows[:30]
        ],
    )
    area_table = _format_table(
        ["Area", "Count", "Total s", "Avg s", "Max s"],
        [
            [
                str(row["area"]),
                str(row["count"]),
                f"{float(row['total_s']):.3f}",
                f"{float(row['avg_s']):.3f}",
                f"{float(row['max_s']):.3f}",
            ]
            for row in area_rows
        ],
    )
    step_area_table = _format_table(
        ["Step", "Area", "Count", "Total s"],
        [
            [str(row["step"]), str(row["area"]), str(row["count"]), f"{float(row['total_s']):.3f}"]
            for row in step_area_rows[:80]
        ],
    )
    steady_area_table = _format_table(
        ["Step", "Area", "Count", "Total s"],
        [
            [str(row["step"]), str(row["area"]), str(row["count"]), f"{float(row['total_s']):.3f}"]
            for row in steady_area_rows[:80]
        ],
    )
    critical_path_table = _format_table(
        ["Step", "Events", "Wall s", "Covered s", "Idle Gap s", "Top Event", "Top Event s"],
        [
            [
                str(row["step"]),
                str(row["event_count"]),
                f"{float(row['wall_s']):.3f}",
                f"{float(row['covered_s']):.3f}",
                f"{float(row['idle_s']):.3f}",
                str(row["top_event"]),
                f"{float(row['top_event_s']):.3f}",
            ]
            for row in critical_path_rows
        ],
    )
    gap_table = _format_table(
        ["Step", "From", "To", "Gap s"],
        [[str(row["step"]), str(row["from"]), str(row["to"]), f"{float(row['gap_s']):.3f}"] for row in gap_rows],
    )
    train_table = _format_table(
        ["Path", "Size bytes"],
        [[item["path"], str(item["size_bytes"])] for item in train_files[:80]],
    )
    sglang_table = _format_table(
        ["Path", "Size bytes"],
        [[item["path"], str(item["size_bytes"])] for item in sglang_files[:80]],
    )

    marker_lines = []
    for key in ["ray_jobs", "launch_phase", "progress", "profile", "errors"]:
        marker_lines.append(f"### {key.replace('_', ' ').title()}")
        values = markers[key]
        if values:
            marker_lines.extend(f"- `{value}`" for value in values)
        else:
            marker_lines.append("- _None found._")
        marker_lines.append("")

    missing_lines = "\n".join(f"- `{item}`" for item in missing) if missing else "- _None._"
    hypothesis_lines = [
        "- Use the largest positive timeline gaps as the next instrumentation target.",
        "- If train traces exist but timeline gaps are outside train events, "
        "prioritize orchestration before kernel tuning.",
        "- If SGLang traces dominate the steady-state window, split the next pass by prefill and decode.",
    ]

    report = f"""# Fully Async Profile Report

## Launch

- Profile root: `{profile_root}`
- Expected pass: `{expected_pass}`
- Run status: `{run_status}`
- Experiment: `{experiment}`

{_format_table(["Key", "Value"], config_rows)}
## Artifact Checks

Missing expected artifacts:

{missing_lines}

## Research Summary

| Key | Value |
| --- | --- |
| Raw timeline event count | `{len(raw_timeline_events)}` |
| Deduplicated timeline event count | `{len(timeline_events)}` |
| Ray job submit wall time s | `{wall_clock['ray_job_submit_s'] or 'unknown'}` |
| Training loop wall time s | `{wall_clock['training_loop_s'] or 'unknown'}` |
| Main success marker | `{wall_clock['main_success']}` |
| First actor train marker | `{wall_clock['first_actor_train'] or 'unknown'}` |
| Recommended next check | `{recommendation}` |

Notes:

- Timeline event durations below are deduplicated because metrics service dumps can be cumulative snapshots.
- Area totals may still exceed wall-clock time because fully_async services and ranks overlap.
- Use warmup-excluded rows for steady-state research decisions; short `all`
  profiles are startup-biased and profiler-perturbed.

## Timeline Area Durations

{area_table}
## Step Area Durations

{step_area_table}
## Warmup-Excluded Step Area Durations

{steady_area_table}
## Critical Path Approximation

{critical_path_table}
## Timeline Event Durations

{timeline_table}
## Step-Level Timeline Gaps

{gap_table}
## Train Trace Index

{train_table}
## SGLang Trace Index

{sglang_table}
## Log Markers

{chr(10).join(marker_lines)}
## Actionable Next Checks

Observed facts:

- Raw timeline event count: `{len(raw_timeline_events)}`
- Deduplicated timeline event count: `{len(timeline_events)}`
- Train trace file count: `{len(train_files)}`
- SGLang trace file count: `{len(sglang_files)}`
- Missing expected artifact count: `{len(missing)}`

Hypotheses to test next:

{chr(10).join(hypothesis_lines)}
"""
    return report, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument(
        "--expected-pass",
        choices=["smoke", "timeline", "torch", "sglang", "all"],
        default="timeline",
    )
    parser.add_argument("--run-status", default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_root = args.profile_root.resolve()
    if not profile_root.exists():
        raise SystemExit(f"profile root does not exist: {profile_root}")

    report, missing = build_report(profile_root, args.expected_pass, args.run_status)
    report_path = profile_root / "profile_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    if args.run_status == "success" and missing:
        raise SystemExit(f"missing expected artifacts: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
