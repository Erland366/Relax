# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Measure overlap between Relax rollout and actor-training intervals."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TIMESTAMP_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\b")
_ROLLOUT_PATTERN = re.compile(r"\b(Start|Finish) rollout (\d+)/\d+\b")
_ACTOR_PATTERN = re.compile(
    r"\btrain_one_step rollout=(\d+) step=(\d+): "
    r"(starting forward_backward|finished optimizer\.step)\b"
)


def _record_event(
    events: dict[Any, datetime],
    key: Any,
    timestamp: datetime,
    *,
    event_name: str,
) -> None:
    if key in events:
        raise ValueError(f"Duplicate {event_name} event for {key!r}")
    events[key] = timestamp


def _make_interval(
    start: datetime,
    finish: datetime,
    *,
    context: str,
) -> dict[str, Any]:
    duration = (finish - start).total_seconds()
    if duration < 0:
        raise ValueError(f"{context} finished before it started")
    return {
        "started_at": start.isoformat(sep=" "),
        "finished_at": finish.isoformat(sep=" "),
        "duration_seconds": duration,
    }


def _merge_intervals(intervals: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, finish in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, finish))
            continue
        previous_start, previous_finish = merged[-1]
        merged[-1] = (previous_start, max(previous_finish, finish))
    return merged


def _intersection_seconds(
    left: Sequence[tuple[datetime, datetime]],
    right: Sequence[tuple[datetime, datetime]],
) -> float:
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        left_start, left_finish = left[left_index]
        right_start, right_finish = right[right_index]
        intersection_start = max(left_start, right_start)
        intersection_finish = min(left_finish, right_finish)
        if intersection_finish > intersection_start:
            total += (intersection_finish - intersection_start).total_seconds()
        if left_finish <= right_finish:
            left_index += 1
        else:
            right_index += 1
    return total


def analyze_overlap_timeline(lines: Iterable[str]) -> dict[str, Any]:
    """Parse Relax log lines and summarize rollout/actor interval overlap."""
    rollout_starts: dict[int, datetime] = {}
    rollout_finishes: dict[int, datetime] = {}
    actor_starts: dict[tuple[int, int], datetime] = {}
    actor_finishes: dict[tuple[int, int], datetime] = {}

    for line_number, line in enumerate(lines, start=1):
        plain_line = _ANSI_ESCAPE_PATTERN.sub("", line)
        rollout_match = _ROLLOUT_PATTERN.search(plain_line)
        actor_match = _ACTOR_PATTERN.search(plain_line)
        if rollout_match is None and actor_match is None:
            continue

        timestamp_match = _TIMESTAMP_PATTERN.search(plain_line)
        if timestamp_match is None:
            raise ValueError(f"Timeline event on line {line_number} has no Relax timestamp")
        timestamp = datetime.fromisoformat(timestamp_match.group(1))

        if rollout_match is not None:
            event, rollout_text = rollout_match.groups()
            rollout = int(rollout_text)
            target = rollout_starts if event == "Start" else rollout_finishes
            _record_event(target, rollout, timestamp, event_name=f"rollout {event.lower()}")
            continue

        rollout_text, step_text, event = actor_match.groups()
        key = (int(rollout_text), int(step_text))
        target = actor_starts if event == "starting forward_backward" else actor_finishes
        _record_event(target, key, timestamp, event_name=f"actor {event}")

    complete_rollouts = sorted(rollout_starts.keys() & rollout_finishes.keys())
    complete_actors = sorted(actor_starts.keys() & actor_finishes.keys())

    rollout_intervals = []
    raw_rollout_intervals = []
    for rollout in complete_rollouts:
        start = rollout_starts[rollout]
        finish = rollout_finishes[rollout]
        raw_rollout_intervals.append((start, finish))
        rollout_intervals.append(
            {
                "rollout": rollout,
                **_make_interval(start, finish, context=f"Rollout {rollout}"),
            }
        )

    actor_intervals = []
    raw_actor_intervals = []
    for rollout, step in complete_actors:
        start = actor_starts[(rollout, step)]
        finish = actor_finishes[(rollout, step)]
        raw_actor_intervals.append((start, finish))
        actor_intervals.append(
            {
                "rollout": rollout,
                "step": step,
                **_make_interval(start, finish, context=f"Actor rollout={rollout} step={step}"),
            }
        )

    merged_rollouts = _merge_intervals(raw_rollout_intervals)
    merged_actors = _merge_intervals(raw_actor_intervals)
    total_rollout_seconds = sum((finish - start).total_seconds() for start, finish in merged_rollouts)
    total_actor_seconds = sum((finish - start).total_seconds() for start, finish in merged_actors)
    overlap_seconds = _intersection_seconds(merged_rollouts, merged_actors)
    overlap_percent = 0.0 if total_rollout_seconds == 0 else 100.0 * overlap_seconds / total_rollout_seconds

    unmatched_actor_starts = [
        {"rollout": rollout, "step": step} for rollout, step in sorted(actor_starts.keys() - actor_finishes.keys())
    ]
    unmatched_actor_finishes = [
        {"rollout": rollout, "step": step} for rollout, step in sorted(actor_finishes.keys() - actor_starts.keys())
    ]
    return {
        "rollout_intervals": rollout_intervals,
        "actor_intervals": actor_intervals,
        "summary": {
            "rollout_start_count": len(rollout_starts),
            "rollout_finish_count": len(rollout_finishes),
            "actor_start_count": len(actor_starts),
            "actor_finish_count": len(actor_finishes),
            "complete_rollout_interval_count": len(rollout_intervals),
            "complete_actor_interval_count": len(actor_intervals),
            "unmatched_rollout_starts": sorted(rollout_starts.keys() - rollout_finishes.keys()),
            "unmatched_rollout_finishes": sorted(rollout_finishes.keys() - rollout_starts.keys()),
            "unmatched_actor_starts": unmatched_actor_starts,
            "unmatched_actor_finishes": unmatched_actor_finishes,
            "total_rollout_seconds": total_rollout_seconds,
            "total_actor_optimizer_wall_seconds": total_actor_seconds,
            "overlap_seconds": overlap_seconds,
            "rollout_overlap_percent": overlap_percent,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Relax log to analyze")
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with args.log.open() as log_file:
        artifact = analyze_overlap_timeline(log_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
