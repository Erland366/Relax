# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRELOAD_RUNNER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_preload_all_vision_features.sh"


def test_runner_is_direct_counterbalanced_and_locks_the_workload():
    assert PRELOAD_RUNNER.is_file(), f"full-dataset preload runner is missing: {PRELOAD_RUNNER}"
    runner = PRELOAD_RUNNER.read_text()

    assert "#SBATCH" not in runner
    assert not re.search(r"\bsbatch\b", runner)
    assert "SLURM_JOB_ID" not in runner
    assert 'orders=("none full_dataset" "full_dataset none" "none full_dataset" "full_dataset none")' in runner
    assert "train_seeds=(1234 1235 1236 1237)" in runner
    assert "rollout_seeds=(42 43 44 45)" in runner

    for fixed_setting in (
        "NUM_ROLLOUT=22",
        "NUM_STEPS_PER_ROLLOUT=2",
        "ROLLOUT_BATCH_SIZE=32",
        "N_SAMPLES_PER_PROMPT=2",
        "GLOBAL_BATCH_SIZE=32",
        "RAY_NUM_CPUS=16",
        "CPU_VISION_CAPACITY_MODE=1",
        "MAX_STALENESS=4",
        "VISION_ENCODER_NUM_REPLICAS=1",
        "VISION_ENCODER_NUM_CPUS=1",
        "SKIP_GPU_VISION_ENCODER=1",
        "VISION_ENCODER_MAX_IMAGES_PER_REQUEST=8",
        "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0",
        "SAVE_CHECKPOINTS=0",
        "ENABLE_EVAL=0",
    ):
        assert fixed_setting in runner

    assert "examples.visual_xor.cpu_vision_capacity_preflight" in runner
    assert "--min-affinity-cpus=16" in runner
    assert "--min-physical-cores=16" in runner
    assert "examples.visual_xor.measure_full_dataset_preload" in runner
    no_preload = re.search(r"none\)(?P<body>.*?)\s+;;", runner, flags=re.DOTALL)
    full_dataset = re.search(r"full_dataset\)(?P<body>.*?)\s+;;", runner, flags=re.DOTALL)
    assert no_preload is not None
    assert "cpu_cache_bytes=1" in no_preload.group("body")
    assert "sglang_bytes=0" in no_preload.group("body")
    assert "preload_enabled=0" in no_preload.group("body")
    assert full_dataset is not None
    assert "cpu_cache_bytes=1073741824" in full_dataset.group("body")
    assert "sglang_bytes=1073741824" in full_dataset.group("body")
    assert "preload_enabled=1" in full_dataset.group("body")


def test_runner_records_preload_artifacts_and_versioned_analysis_inputs():
    runner = PRELOAD_RUNNER.read_text()

    expected_header = (
        "repeat\\torder_position\\tpreload\\ttrain_seed\\trollout_seed\\tpreload_enabled\\t"
        "preload_artifact\\tstarted_at\\tended_at\\texit_status\\tconsole_log\\tinternal_run_log"
    )
    assert expected_header in runner
    assert "schema_version" in runner
    assert "wall_seconds" in runner
    assert re.search(r'analysis_args\+=\(--run "\$\{repeat\}:\$\{preload\}=\$\{console_log\}"\)', runner)
    assert re.search(
        r'preload_args\+=\(--preload-artifact "\$\{repeat\}=\$\{preload_artifact\}"\)', runner
    )


def test_runner_checks_hardware_inputs_and_records_provenance():
    runner = PRELOAD_RUNNER.read_text()

    assert "torch.cuda.device_count()" in runner
    assert "exactly four visible GPUs" in runner
    assert "MI210" in runner
    assert "Required full-dataset preload input does not exist" in runner
    assert "input-sha256.txt" in runner
    assert "sha256sum" in runner
    assert "cpu-topology.txt" in runner
    assert "lscpu" in runner
    assert "rocm-smi" in runner
    assert "--showmeminfo vram" in runner
    assert "vram-before.csv" in runner
    assert "vram-after.csv" in runner


def test_runner_strips_ansi_before_extracting_preload_summary():
    runner = PRELOAD_RUNNER.read_text()
    extraction = runner.split("extract_preload_summary()", 1)[1].split("validate_completed_run()", 1)[0]

    assert r"\x1b\[" in extraction
    assert "re.sub" in extraction


def test_runner_always_shuts_down_owned_ray_and_serve_state():
    runner = PRELOAD_RUNNER.read_text()

    assert re.search(r"trap\s+[\"']?cleanup_[a-z_]+[\"']?\s+EXIT", runner)
    assert "ray serve shutdown -y" in runner
    assert re.search(r"\bray stop\b", runner)


def test_capacity_preflight_can_require_sixteen_distinct_physical_cores(monkeypatch, capsys):
    from examples.visual_xor import cpu_vision_capacity_preflight as preflight

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cpu_vision_capacity_preflight",
            "--vision-encoder-num-replicas=1",
            "--vision-encoder-num-cpus=1",
            "--min-affinity-cpus=16",
            "--min-physical-cores=16",
        ],
    )
    monkeypatch.setattr(preflight.os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(
        preflight,
        "_physical_core_keys",
        lambda _cpu_ids: [(0, cpu_id % 8) for cpu_id in range(16)],
    )

    with pytest.raises(SystemExit):
        preflight.main()

    assert "at least 16 distinct physical cores; found 8" in capsys.readouterr().err
