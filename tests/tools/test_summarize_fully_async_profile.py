import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "summarize_fully_async_profile.py"


def write_profile_root(tmp_path: Path, *, include_traces: bool) -> Path:
    profile_root = tmp_path / "profile"
    timeline_dir = profile_root / "timeline"
    train_trace_dir = profile_root / "traces" / "qwen3-profile" / "train_trace"
    sglang_trace_dir = profile_root / "sglang_trace" / "rollout_1"
    timeline_dir.mkdir(parents=True)
    train_trace_dir.mkdir(parents=True)
    sglang_trace_dir.mkdir(parents=True)

    (profile_root / "run.log").write_text(
        "\n".join(
            [
                "wandb: W&B syncing is set to `offline`",
                "TimelineTrace adapter enabled, dumping to: /tmp/timeline",
                "PyTorch profiler for overall training is enabled",
                "Starting SGLang profiling on 1 engines for rollout step 1",
                "Job 'raysubmit_test' succeeded",
            ]
        ),
        encoding="utf-8",
    )
    (profile_root / "launch.env").write_text(
        "\n".join(
            [
                "RELAX_PROFILE_PASS=both",
                "WANDB_MODE=offline",
                "RELAX_SGL_KERNEL_BUILD_DIR=/tmp/sgl-kernel/build",
                f"RELAX_TB_EXPERIMENT_NAME={profile_root / 'traces' / 'qwen3-profile'}",
                f"RELAX_TIMELINE_DUMP_DIR={timeline_dir}",
                f"PASS2_RELAX_SGLANG_PROFILE_OUTPUT_DIR={profile_root / 'sglang_trace'}",
            ]
        ),
        encoding="utf-8",
    )
    (timeline_dir / "timeline_step_1.json").write_text(
        json.dumps(
            [
                {"name": "rollout", "ph": "X", "ts": 0, "dur": 1000, "pid": 1, "tid": 1, "args": {"step": 1}},
                {"name": "train", "ph": "X", "ts": 5000, "dur": 2000, "pid": 2, "tid": 1, "args": {"step": 1}},
                {"name": "rollout", "ph": "X", "ts": 9000, "dur": 3000, "pid": 1, "tid": 1, "args": {"step": 2}},
            ]
        ),
        encoding="utf-8",
    )

    if include_traces:
        (train_trace_dir / "train_overall_rank0_dp0_tp0_pp0.1.pt.trace.json.gz").write_bytes(b"train")
        (sglang_trace_dir / "engine0-1-TP-0.trace.json.gz").write_bytes(b"sglang")

    return profile_root


def run_summarizer(profile_root: Path) -> str:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--profile-root", str(profile_root)],
        check=True,
        text=True,
        capture_output=True,
    )
    return (profile_root / "profile_report.md").read_text(encoding="utf-8")


def test_summarize_fully_async_profile_indexes_artifacts_and_timeline(tmp_path):
    profile_root = write_profile_root(tmp_path, include_traces=True)

    report = run_summarizer(profile_root)

    assert "## Artifact Index" in report
    assert "| rollout | 2 | 4.00 | 2.00 | 3.00 |" in report
    assert "| train | 1 | 2.00 | 2.00 | 2.00 |" in report
    assert "| 1 | 4.00 | rollout | train |" in report
    assert "train_overall_rank0_dp0_tp0_pp0.1.pt.trace.json.gz" in report
    assert "engine0-1-TP-0.trace.json.gz" in report
    assert "- None." in report


def test_summarize_fully_async_profile_reports_missing_trace_warnings(tmp_path):
    profile_root = write_profile_root(tmp_path, include_traces=False)

    report = run_summarizer(profile_root)

    assert "No training profiler trace files found" in report
    assert "No SGLang profiler trace files found" in report
    assert "Fix missing artifact coverage before drawing kernel-level conclusions" in report
