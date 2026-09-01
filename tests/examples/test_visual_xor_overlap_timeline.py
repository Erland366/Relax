# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.visual_xor.analyze_overlap_timeline import analyze_overlap_timeline


def _line(timestamp: str, message: str, *, actor: str = "Rollout") -> str:
    return (
        f"\x1b[36m({actor} pid=123)\x1b[0m "
        f"\x1b[32m{timestamp}\x1b[0m | \x1b[1;37mINFO\x1b[0m | "
        f"\x1b[1;37m{message}\x1b[0m"
    )


def test_analyze_overlap_timeline_pairs_ids_and_measures_intersection():
    lines = [
        _line("2026-08-06 10:00:00", "Start rollout 0/2"),
        _line(
            "2026-08-06 10:00:05",
            "train_one_step rollout=0 step=0: starting forward_backward (num_microbatches=16)",
            actor="MegatronTrainRayActor",
        ),
        _line("2026-08-06 10:00:10", "Finish rollout 0/2"),
        _line("2026-08-06 10:00:10", "Start rollout 1/2"),
        _line(
            "2026-08-06 10:00:12",
            "train_one_step rollout=0 step=0: finished optimizer.step (update_successful=True)",
            actor="MegatronTrainRayActor",
        ),
        _line(
            "2026-08-06 10:00:14",
            "train_one_step rollout=0 step=1: starting forward_backward (num_microbatches=16)",
            actor="MegatronTrainRayActor",
        ),
        _line(
            "2026-08-06 10:00:18",
            "train_one_step rollout=0 step=1: finished optimizer.step (update_successful=True)",
            actor="MegatronTrainRayActor",
        ),
        _line("2026-08-06 10:00:20", "Finish rollout 1/2"),
    ]

    artifact = analyze_overlap_timeline(lines)

    assert [(interval["rollout"], interval["duration_seconds"]) for interval in artifact["rollout_intervals"]] == [
        (0, 10.0),
        (1, 10.0),
    ]
    assert [
        (interval["rollout"], interval["step"], interval["duration_seconds"])
        for interval in artifact["actor_intervals"]
    ] == [(0, 0, 7.0), (0, 1, 4.0)]
    assert artifact["summary"] == {
        "rollout_start_count": 2,
        "rollout_finish_count": 2,
        "actor_start_count": 2,
        "actor_finish_count": 2,
        "complete_rollout_interval_count": 2,
        "complete_actor_interval_count": 2,
        "unmatched_rollout_starts": [],
        "unmatched_rollout_finishes": [],
        "unmatched_actor_starts": [],
        "unmatched_actor_finishes": [],
        "total_rollout_seconds": 20.0,
        "total_actor_optimizer_wall_seconds": 11.0,
        "overlap_seconds": 11.0,
        "rollout_overlap_percent": 55.0,
    }


def test_analyze_overlap_timeline_reports_unmatched_events_from_partial_logs():
    lines = [
        _line("2026-08-06 10:00:00", "Start rollout 2/5"),
        _line("2026-08-06 10:00:01", "Finish rollout 3/5"),
        _line(
            "2026-08-06 10:00:02",
            "train_one_step rollout=2 step=0: starting forward_backward (num_microbatches=16)",
            actor="MegatronTrainRayActor",
        ),
        _line(
            "2026-08-06 10:00:03",
            "train_one_step rollout=3 step=0: finished optimizer.step (update_successful=True)",
            actor="MegatronTrainRayActor",
        ),
    ]

    summary = analyze_overlap_timeline(lines)["summary"]

    assert summary["unmatched_rollout_starts"] == [2]
    assert summary["unmatched_rollout_finishes"] == [3]
    assert summary["unmatched_actor_starts"] == [{"rollout": 2, "step": 0}]
    assert summary["unmatched_actor_finishes"] == [{"rollout": 3, "step": 0}]
    assert summary["complete_rollout_interval_count"] == 0
    assert summary["complete_actor_interval_count"] == 0


@pytest.mark.parametrize(
    "messages",
    [
        ["Start rollout 0/2", "Start rollout 0/2"],
        ["Finish rollout 0/2", "Finish rollout 0/2"],
        [
            "train_one_step rollout=0 step=0: starting forward_backward (num_microbatches=16)",
            "train_one_step rollout=0 step=0: starting forward_backward (num_microbatches=16)",
        ],
        [
            "train_one_step rollout=0 step=0: finished optimizer.step (update_successful=True)",
            "train_one_step rollout=0 step=0: finished optimizer.step (update_successful=True)",
        ],
    ],
)
def test_analyze_overlap_timeline_rejects_ambiguous_duplicate_events(messages):
    lines = [_line(f"2026-08-06 10:00:0{index}", message) for index, message in enumerate(messages)]

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        analyze_overlap_timeline(lines)
