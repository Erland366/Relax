import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "summarize_fully_async_profile.py"
SPEC = importlib.util.spec_from_file_location("summarize_fully_async_profile", MODULE_PATH)
summarize = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize)


def test_profile_report_indexes_timeline_and_traces(tmp_path):
    profile_root = tmp_path / "profile"
    timeline_dir = profile_root / "timeline"
    train_dir = profile_root / "traces" / "exp" / "train_trace"
    sglang_dir = profile_root / "sglang_trace" / "rollout_1"
    timeline_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    sglang_dir.mkdir(parents=True)

    (profile_root / "launch.env").write_text(
        "\n".join(
            [
                "PROFILE_PASS=all",
                "RELAX_TB_EXPERIMENT_NAME=exp",
                "HIP_VISIBLE_DEVICES=0,1,2,3",
                "RELAX_USE_PYTORCH_PROFILER=1",
                "RELAX_SGLANG_PROFILE=1",
            ]
        ),
        encoding="utf-8",
    )
    (profile_root / "run.log").write_text(
        "\n".join(
            [
                "[2026-06-12T00:00:00+00:00] launch_phase: ray_job_submit begin",
                "raysubmit_ABC123",
                "Start rollout 1/4",
                "Actor training step 1/4",
                "PyTorch profiler for overall training is enabled",
            ]
        ),
        encoding="utf-8",
    )
    (timeline_dir / "timeline_step_1.json").write_text(
        json.dumps(
            [
                {"name": "rollout", "ph": "X", "ts": 1_000_000, "dur": 2_000_000, "args": {"step": 1}},
                {"name": "actor_train", "ph": "X", "ts": 4_000_000, "dur": 1_000_000, "args": {"step": 1}},
            ]
        ),
        encoding="utf-8",
    )
    (train_dir / "worker.pt.trace.json.gz").write_bytes(b"train")
    (sglang_dir / "engine0.trace.json.gz").write_bytes(b"sglang")

    report, missing = summarize.build_report(profile_root, "all", "success")

    assert missing == []
    assert "rollout" in report
    assert "actor_train" in report
    assert "worker.pt.trace.json.gz" in report
    assert "engine0.trace.json.gz" in report
    assert "Step-Level Timeline Gaps" in report
    assert "raysubmit_ABC123" in report


def test_profile_report_flags_missing_expected_artifacts(tmp_path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / "launch.env").write_text("PROFILE_PASS=torch\nRELAX_TB_EXPERIMENT_NAME=exp\n", encoding="utf-8")
    (profile_root / "run.log").write_text("Main func successfully\n", encoding="utf-8")

    report, missing = summarize.build_report(profile_root, "torch", "success")

    assert "timeline/timeline_step_*.json" in missing
    assert "traces/<experiment>/train_trace/*" in missing
    assert "Missing expected artifacts" in report
    assert "timeline/timeline_step_*.json" in report


def test_profile_report_deduplicates_cumulative_timeline_snapshots(tmp_path):
    profile_root = tmp_path / "profile"
    timeline_dir = profile_root / "timeline"
    timeline_dir.mkdir(parents=True)

    (profile_root / "launch.env").write_text("PROFILE_PASS=timeline\nRELAX_TB_EXPERIMENT_NAME=exp\n", encoding="utf-8")
    (profile_root / "run.log").write_text(
        "\n".join(
            [
                "[2026-06-12T00:00:00+00:00] launch_phase: ray_job_submit begin",
                "[2026-06-12T00:10:00+00:00] launch_phase: ray_job_submit end",
                "2026-06-12 00:09:00 | INFO | __main__:114 launch_timing: training_loop end elapsed=120.50s",
                "Actor training step 1/4",
                "Main func successfully",
            ]
        ),
        encoding="utf-8",
    )
    rollout = {
        "name": "rollout",
        "ph": "X",
        "ts": 1_000_000,
        "dur": 2_000_000,
        "pid": 1,
        "tid": 1,
        "args": {"step": 0},
    }
    advantages = {
        "name": "advantages_compute",
        "ph": "X",
        "ts": 3_000_000,
        "dur": 1_000_000,
        "pid": 2,
        "tid": 1,
        "args": {"step": 0},
    }
    log_probs = {
        "name": "log_probs",
        "ph": "X",
        "ts": 5_000_000,
        "dur": 3_000_000,
        "pid": 3,
        "tid": 1,
        "args": {"step": 1},
    }
    weight_sync = {
        "name": "recv_weight_fully_async",
        "ph": "X",
        "ts": 9_000_000,
        "dur": 4_000_000,
        "pid": 4,
        "tid": 1,
        "args": {"step": 1},
    }

    (timeline_dir / "timeline_step_0.json").write_text(json.dumps([rollout, advantages]), encoding="utf-8")
    (timeline_dir / "timeline_step_1.json").write_text(
        json.dumps([rollout, advantages, log_probs, weight_sync]),
        encoding="utf-8",
    )

    report, missing = summarize.build_report(profile_root, "timeline", "success")

    assert missing == []
    assert "Research Summary" in report
    assert "Raw timeline event count | `6`" in report
    assert "Deduplicated timeline event count | `4`" in report
    assert "Ray job submit wall time s | `600.00`" in report
    assert "Training loop wall time s | `120.50`" in report
    assert "| actor_fwd |" in report
    assert "| weight_sync |" in report
    assert "Warmup-Excluded Step Area Durations" in report
    assert "| 1 | actor_fwd | 1 | 3.000 |" in report
    assert "| 1 | weight_sync | 1 | 4.000 |" in report
