import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAUNCHER = REPO_ROOT / "scripts" / "training" / "multimodal" / "amd_qwen3_4b_2gpu_e2e.sh"
CPU_VISION_LAUNCHER = (
    REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"
)
GPU_VISION_LAUNCHER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"
CPU_WORKER_LAYOUTS_LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "qwen3_vl_compare_cpu_worker_layouts.sbatch"
VISION_CACHE_REUSE_LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "qwen3_vl_measure_vision_cache_reuse.sbatch"
MATCHED_SLURM_LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "qwen3_vl_compare_vision_settings.sbatch"
VISION_DEVICE_COMPARE_RUNNER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_compare_vision_devices.sh"
GEO3K_LAUNCHER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_4b_geo3k_fully_async_8gpus.sh"


def _run_debug_launcher(
    launcher: Path,
    tmp_path: Path,
    *,
    enable_eval: str | None,
    eval_config_exists: bool,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"")
    prompt_set = tmp_path / "train.parquet"
    prompt_set.write_bytes(b"")
    eval_config = tmp_path / "eval.json"
    if eval_config_exists:
        eval_config.write_text("{}")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_path = tmp_path / "base-launch.env"
    launch_trace = tmp_path / "launch.trace"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text('#!/bin/sh\nprintf "base:%s\\n" "$*" >> "$LAUNCH_TRACE"\nenv > "$CAPTURE_ENV"\n')
    fake_bash.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/bin/sh\nprintf "preflight:%s\\n" "$*" >> "$LAUNCH_TRACE"\nexit "${PREFLIGHT_EXIT_CODE:-0}"\n'
    )
    fake_python.chmod(0o755)
    (fake_bin / "python3").symlink_to(fake_python)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_ENV": str(capture_path),
            "EVAL_CONFIG": str(eval_config),
            "EVAL_INTERVAL": "7",
            "EVAL_MAX_CONTEXT_LEN": "999",
            "EVAL_MAX_PROMPT_LEN": "999",
            "EVAL_MAX_RESPONSE_LEN": "999",
            "HF_CHECKPOINT": str(checkpoint),
            "LAUNCH_TRACE": str(launch_trace),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PROMPT_SET": str(prompt_set),
            "PYTHON": str(fake_python),
        }
    )
    if extra_env is not None:
        env.update(extra_env)
    if enable_eval is None:
        env.pop("ENABLE_EVAL", None)
    else:
        env["ENABLE_EVAL"] = enable_eval

    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    captured_env = {}
    if capture_path.exists():
        captured_env = dict(line.split("=", 1) for line in capture_path.read_text().splitlines() if "=" in line)
    return result, captured_env


def _capacity_preflight_module():
    return importlib.import_module("examples.visual_xor.cpu_vision_capacity_preflight")


def _physical_core_keys(logical_cpu_count: int, distinct_core_count: int) -> list[tuple[int, int]]:
    return [(0, cpu_id % distinct_core_count) for cpu_id in range(logical_cpu_count)]


def test_cpu_vision_capacity_preflight_is_import_light():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'ray', 'relax', 'torch', 'transformers'}:
            raise AssertionError(f'heavy import during capacity preflight import: {fullname}')
        return None

sys.meta_path.insert(0, BlockHeavyImports())
import examples.visual_xor.cpu_vision_capacity_preflight
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cpu_vision_capacity_preflight_rejects_too_few_scheduler_affinity_cpus():
    with pytest.raises(ValueError, match=r"at least 16 scheduler-affinity CPUs.*found 15"):
        _capacity_preflight_module().validate_cpu_capacity(
            affinity_cpu_ids=range(15),
            physical_core_keys=_physical_core_keys(15, 8),
            reserved_vit_cpus=8,
        )


def test_cpu_vision_capacity_preflight_rejects_too_few_distinct_physical_cores():
    with pytest.raises(ValueError, match=r"at least 8 distinct physical cores.*found 7"):
        _capacity_preflight_module().validate_cpu_capacity(
            affinity_cpu_ids=range(16),
            physical_core_keys=_physical_core_keys(16, 7),
            reserved_vit_cpus=8,
        )


def test_cpu_vision_capacity_preflight_rejects_more_than_eight_reserved_vit_cpus():
    with pytest.raises(ValueError, match=r"at most 8 reserved ViT CPUs.*got 9"):
        _capacity_preflight_module().validate_cpu_capacity(
            affinity_cpu_ids=range(16),
            physical_core_keys=_physical_core_keys(16, 8),
            reserved_vit_cpus=9,
        )


def test_cpu_vision_capacity_preflight_accepts_required_topology():
    _capacity_preflight_module().validate_cpu_capacity(
        affinity_cpu_ids=range(16),
        physical_core_keys=_physical_core_keys(16, 8),
        reserved_vit_cpus=8,
    )


def test_training_launcher_forwards_validated_cpu_vision_replica_count():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"' in launcher
    assert "require_positive_integer VISION_ENCODER_NUM_REPLICAS" in launcher
    assert '--vision-encoder-num-replicas "${VISION_ENCODER_NUM_REPLICAS}"' in launcher
    assert "replicas=${VISION_ENCODER_NUM_REPLICAS}" in launcher


def test_cpu_vision_launchers_forward_batch_wait_timeout_with_off_by_default():
    training_launcher = TRAINING_LAUNCHER.read_text()
    cpu_vision_launcher = CPU_VISION_LAUNCHER.read_text()

    assert (
        'VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS="${VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS:-0}"'
        in training_launcher
    )
    assert (
        '--vision-encoder-batch-wait-timeout-ms "${VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS}"'
        in training_launcher
    )
    assert (
        'export VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS="${VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS:-0}"'
        in cpu_vision_launcher
    )


def test_cpu_vision_launchers_forward_sglang_feature_cache_with_off_by_default():
    training_launcher = TRAINING_LAUNCHER.read_text()
    cpu_vision_launcher = CPU_VISION_LAUNCHER.read_text()

    assert (
        'SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${SGLANG_VISION_FEATURE_CACHE_MAX_BYTES:-0}"'
        in training_launcher
    )
    assert "require_nonnegative_integer SGLANG_VISION_FEATURE_CACHE_MAX_BYTES" in training_launcher
    assert (
        '--sglang-vision-feature-cache-max-bytes "${SGLANG_VISION_FEATURE_CACHE_MAX_BYTES}"'
        in training_launcher
    )
    assert (
        'export SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${SGLANG_VISION_FEATURE_CACHE_MAX_BYTES:-0}"'
        in cpu_vision_launcher
    )


def test_cpu_vision_launchers_forward_all_feature_preload_only_when_enabled():
    training_launcher = TRAINING_LAUNCHER.read_text()
    cpu_vision_launcher = CPU_VISION_LAUNCHER.read_text()

    assert 'PRELOAD_VISION_FEATURES="${PRELOAD_VISION_FEATURES:-0}"' in training_launcher
    assert "require_boolean_flag PRELOAD_VISION_FEATURES" in training_launcher
    conditional_append = re.search(
        r'if \[ "\$\{PRELOAD_VISION_FEATURES\}" = "1" \]; then\s+'
        r"(?P<array>[A-Z_]+)\+=\(--preload-vision-features\)\s+fi",
        training_launcher,
    )
    assert conditional_append is not None
    preload_args = conditional_append.group("array")
    assert f'"${{{preload_args}[@]}}"' in training_launcher
    assert training_launcher.count("--preload-vision-features") == 1
    assert (
        'export PRELOAD_VISION_FEATURES="${PRELOAD_VISION_FEATURES:-0}"'
        in cpu_vision_launcher
    )


def test_cpu_vision_launchers_forward_automatic_vision_device_only_with_explicit_plan():
    training_launcher = TRAINING_LAUNCHER.read_text()
    cpu_vision_launcher = CPU_VISION_LAUNCHER.read_text()

    assert 'VISION_DEVICE_MODE="${VISION_DEVICE_MODE:-fixed}"' in training_launcher
    assert "--vision-device-mode" in training_launcher
    assert "--vision-device-plan" in training_launcher
    assert "VISION_DEVICE_MODE=automatic requires VISION_ENCODER_DEVICE=cpu" in training_launcher
    assert "requires the GPU vision encoder" in training_launcher
    assert 'export VISION_DEVICE_MODE="${VISION_DEVICE_MODE:-fixed}"' in cpu_vision_launcher


def test_vision_device_comparison_runner_uses_matched_counterbalanced_profiles_and_strict_analysis():
    runner = VISION_DEVICE_COMPARE_RUNNER.read_text()

    assert runner.count('"gpu cpu automatic"') == 1
    assert "train_seeds=(1234 1235 1236 1237 1238)" in runner
    assert "NUM_ROLLOUT=24" in runner
    assert "ROLLOUT_BATCH_SIZE=32" in runner
    assert "N_SAMPLES_PER_PROMPT=2" in runner
    assert "VISION_ENCODER_CACHE_MAX_BYTES=1" in runner
    assert "examples.visual_xor.compare_vision_device_choices" in runner
    assert '"require_both_devices": True' in runner
    assert '"expected_generation_requests_per_cycle": 32' in runner
    assert "PROFILE_TIMEOUT_SECONDS" in runner
    assert 'exec 9>"/tmp/relax-ray-${UID}.lock"' in runner
    assert "RAY_ADDRESS must be unset" in runner
    assert "refusing to claim runtime ownership" in runner


def test_geo3k_paper_launcher_exposes_all_vision_modes_without_checkpoints_by_default():
    launcher = GEO3K_LAUNCHER.read_text()

    for mode in ("gpu", "cpu", "cpu_skip_gpu_encoder", "automatic"):
        assert f"    {mode})" in launcher
    assert "Automatic Geo3K device choice requires runtime workload measurements" in launcher
    assert "requires exactly 8 visible GPUs" in launcher
    assert "must be inherited from the eight-GPU allocation" in launcher
    assert "eight unique device identifiers" in launcher
    assert "unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES" not in launcher
    assert "INPUT_KEY=prompt" in launcher
    assert "LABEL_KEY=reward_model" in launcher
    assert "RM_TYPE=geo3k" in launcher
    assert 'SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"' in launcher


def test_cpu_worker_layout_matrix_keeps_batching_off_after_scaling_result():
    launcher = CPU_WORKER_LAYOUTS_LAUNCHER.read_text()

    assert "#SBATCH --cpus-per-task=20" in launcher
    assert "profiles=(replicas1_threads4 replicas2_threads2 replicas4_threads1)" in launcher
    assert "replicas=(1 2 4)" in launcher
    assert "threads=(4 2 1)" in launcher
    assert "RAY_NUM_CPUS=20" in launcher
    assert "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0" in launcher
    assert "SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=0" in launcher


def test_vision_cache_reuse_slurm_matrix_isolates_cpu_cache_and_transport_reuse():
    launcher = VISION_CACHE_REUSE_LAUNCHER.read_text()

    assert "profiles=(cpu_cache_only cpu_and_sglang_cache)" in launcher
    assert "cpu_cache_bytes=(1073741824 1073741824)" in launcher
    assert "sglang_cache_bytes=(0 1073741824)" in launcher
    assert "VISION_ENCODER_NUM_REPLICAS=1" in launcher
    assert "VISION_ENCODER_NUM_CPUS=1" in launcher
    assert "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0" in launcher


def test_vision_cache_reuse_slurm_requires_complete_profiles_and_runs_analysis():
    launcher = VISION_CACHE_REUSE_LAUNCHER.read_text()

    assert "validate_completed_profile" in launcher
    assert "expected_cycles = set(range(12))" in launcher
    assert "incomplete CPU vision metrics" in launcher
    assert "examples.visual_xor.measure_vision_cache_reuse" in launcher
    assert '--profile "cpu_cache_only=${ARTIFACT_DIR}/cpu_cache_only.console.log"' in launcher
    assert '--profile "cpu_and_sglang_cache=${ARTIFACT_DIR}/cpu_and_sglang_cache.console.log"' in launcher


def test_training_launcher_skips_gpu_encoder_only_for_boolean_one():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'SKIP_GPU_VISION_ENCODER="${SKIP_GPU_VISION_ENCODER:-0}"' in launcher
    assert "require_boolean_flag SKIP_GPU_VISION_ENCODER" in launcher

    conditional_append = re.search(
        r'if \[ "\$\{SKIP_GPU_VISION_ENCODER\}" = "1" \]; then\s+'
        r"(?P<array>[A-Z_]+)\+=\(--skip-gpu-vision-encoder\)\s+fi",
        launcher,
    )
    assert conditional_append is not None

    skip_gpu_encoder_args = conditional_append.group("array")
    assert f'"${{{skip_gpu_encoder_args}[@]}}"' in launcher
    assert launcher.count("--skip-gpu-vision-encoder") == 1
    assert "skip_gpu_encoder=${SKIP_GPU_VISION_ENCODER}" in launcher


def test_fully_async_cpu_vision_launcher_keeps_gpu_encoder_skip_opt_in():
    launcher = CPU_VISION_LAUNCHER.read_text()

    assert 'export VISION_ENCODER_NUM_CPUS="${VISION_ENCODER_NUM_CPUS:-1}"' in launcher
    assert 'export VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"' in launcher
    assert 'export SKIP_GPU_VISION_ENCODER="${SKIP_GPU_VISION_ENCODER:-0}"' in launcher
    assert "CPU_VISION_MODE=cpu-skip-gpu-encoder" in launcher
    assert "CPU_VISION_MODE=cpu" in launcher
    assert "visual-xor-refinement-${CPU_VISION_MODE}-${RUN_TAG}.log" in launcher


def test_cpu_vision_capacity_mode_runs_preflight_before_base_launcher(tmp_path):
    result, launched_env = _run_debug_launcher(
        CPU_VISION_LAUNCHER,
        tmp_path,
        enable_eval="0",
        eval_config_exists=False,
        extra_env={
            "CPU_VISION_CAPACITY_MODE": "1",
            "VISION_ENCODER_NUM_CPUS": "4",
            "VISION_ENCODER_NUM_REPLICAS": "2",
        },
    )

    assert result.returncode == 0, result.stderr
    trace = (tmp_path / "launch.trace").read_text().splitlines()
    assert len(trace) == 2
    assert trace[0].startswith("preflight:")
    assert "examples.visual_xor.cpu_vision_capacity_preflight" in trace[0]
    assert "--vision-encoder-num-replicas 2" in trace[0]
    assert "--vision-encoder-num-cpus 4" in trace[0]
    assert trace[1].startswith("base:")
    assert launched_env["CPU_VISION_CAPACITY_MODE"] == "1"


def test_cpu_vision_capacity_preflight_failure_stops_before_base_launcher(tmp_path):
    result, launched_env = _run_debug_launcher(
        CPU_VISION_LAUNCHER,
        tmp_path,
        enable_eval="0",
        eval_config_exists=False,
        extra_env={
            "CPU_VISION_CAPACITY_MODE": "1",
            "PREFLIGHT_EXIT_CODE": "23",
        },
    )

    assert result.returncode == 23
    trace = (tmp_path / "launch.trace").read_text().splitlines()
    assert len(trace) == 1
    assert trace[0].startswith("preflight:")
    assert launched_env == {}


@pytest.mark.parametrize(
    ("rollout_batch_size", "samples_per_prompt"),
    [("8", "8"), ("16", "4"), ("32", "2"), ("64", "1")],
)
def test_cpu_vision_launcher_accepts_demand_profile_overrides(
    tmp_path,
    rollout_batch_size,
    samples_per_prompt,
):
    result, launched_env = _run_debug_launcher(
        CPU_VISION_LAUNCHER,
        tmp_path,
        enable_eval="0",
        eval_config_exists=False,
        extra_env={
            "N_SAMPLES_PER_PROMPT": samples_per_prompt,
            "NUM_ROLLOUT": "12",
            "ROLLOUT_BATCH_SIZE": rollout_batch_size,
        },
    )

    assert result.returncode == 0, result.stderr
    assert launched_env["ROLLOUT_BATCH_SIZE"] == rollout_batch_size
    assert launched_env["N_SAMPLES_PER_PROMPT"] == samples_per_prompt
    assert launched_env["NUM_ROLLOUT"] == "12"


def test_gpu_vision_launcher_accepts_the_same_two_cycle_matrix_overrides():
    launcher = GPU_VISION_LAUNCHER.read_text()

    assert 'export NUM_ROLLOUT="${NUM_ROLLOUT:-250}"' in launcher
    assert 'export EVAL_INTERVAL="${EVAL_INTERVAL:-4}"' in launcher
    assert 'export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"' in launcher
    assert "export FREEZE_VISION_MODEL=1" in launcher
    assert "visual-xor-refinement-gpu-${RUN_TAG}.log" in launcher


def test_gpu_vision_launcher_accepts_prompts32_samples2_overrides(tmp_path):
    result, launched_env = _run_debug_launcher(
        GPU_VISION_LAUNCHER,
        tmp_path,
        enable_eval="0",
        eval_config_exists=False,
        extra_env={
            "GLOBAL_BATCH_SIZE": "32",
            "N_SAMPLES_PER_PROMPT": "2",
            "NUM_ROLLOUT": "22",
            "ROLLOUT_BATCH_SIZE": "32",
        },
    )

    assert result.returncode == 0, result.stderr
    assert launched_env["ROLLOUT_BATCH_SIZE"] == "32"
    assert launched_env["N_SAMPLES_PER_PROMPT"] == "2"
    assert launched_env["GLOBAL_BATCH_SIZE"] == "32"


def test_training_launcher_forwards_explicit_train_and_rollout_seeds():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'TRAIN_SEED="${TRAIN_SEED:-1234}"' in launcher
    assert 'ROLLOUT_SEED="${ROLLOUT_SEED:-42}"' in launcher
    assert "require_nonnegative_integer TRAIN_SEED" in launcher
    assert "require_nonnegative_integer ROLLOUT_SEED" in launcher
    assert '--seed "${TRAIN_SEED}"' in launcher
    assert '--rollout-seed "${ROLLOUT_SEED}"' in launcher
    assert "train_seed=${TRAIN_SEED}, rollout_seed=${ROLLOUT_SEED}" in launcher


def test_matched_slurm_launcher_balances_modes_and_locks_workload():
    launcher = MATCHED_SLURM_LAUNCHER.read_text()

    assert "#SBATCH --cpus-per-task=16" in launcher
    assert "#SBATCH --hint=nomultithread" in launcher
    assert "#SBATCH --gres=gpu:mi210:4" in launcher
    assert "#SBATCH --exclusive" in launcher
    assert "#SBATCH --time=03:00:00" in launcher
    assert '"gpu:0:0 cpu:0:0 cpu:1:0 cpu:1:1"' in launcher
    assert '"cpu:0:0 cpu:1:0 cpu:1:1 gpu:0:0"' in launcher
    assert "train_seeds=(1234 1235 1236 1237)" in launcher
    assert "rollout_seeds=(42 43 44 45)" in launcher
    assert "NUM_ROLLOUT=22" in launcher
    assert "ROLLOUT_BATCH_SIZE=32" in launcher
    assert "N_SAMPLES_PER_PROMPT=2" in launcher
    assert "NUM_STEPS_PER_ROLLOUT=2" in launcher
    assert "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0" in launcher
    assert "SAVE_CHECKPOINTS=0" in launcher
    assert "ENABLE_EVAL=0" in launcher
    assert "validate_cpu_vision_parity" in launcher
    assert "validate_sglang_cpu_vision_parity" in launcher
    assert "compare_vision_settings" in launcher


def test_training_launcher_can_freeze_gpu_vision_for_comparable_benchmarks():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'FREEZE_VISION_MODEL="${FREEZE_VISION_MODEL:-0}"' in launcher
    assert "require_boolean_flag FREEZE_VISION_MODEL" in launcher
    assert 'if [ "${VISION_ENCODER_DEVICE:-gpu}" = "cpu" ]; then' in launcher
    assert "FREEZE_VISION_MODEL=1" in launcher
    assert 'if [ "${FREEZE_VISION_MODEL}" = "1" ]; then' in launcher


@pytest.mark.parametrize("launcher", [GPU_VISION_LAUNCHER, CPU_VISION_LAUNCHER])
def test_visual_xor_debug_launcher_keeps_evaluation_enabled_by_default(launcher, tmp_path):
    result, launched_env = _run_debug_launcher(
        launcher,
        tmp_path,
        enable_eval=None,
        eval_config_exists=True,
    )

    assert result.returncode == 0, result.stderr
    assert launched_env["EVAL_CONFIG"] == str(tmp_path / "eval.json")
    assert launched_env["EVAL_INTERVAL"] == "7"
    assert launched_env["EVAL_MAX_CONTEXT_LEN"] == "512"
    assert launched_env["EVAL_MAX_PROMPT_LEN"] == "511"
    assert launched_env["EVAL_MAX_RESPONSE_LEN"] == "3"


@pytest.mark.parametrize("launcher", [GPU_VISION_LAUNCHER, CPU_VISION_LAUNCHER])
def test_visual_xor_debug_launcher_can_explicitly_disable_evaluation(launcher, tmp_path):
    result, launched_env = _run_debug_launcher(
        launcher,
        tmp_path,
        enable_eval="0",
        eval_config_exists=False,
    )

    assert result.returncode == 0, result.stderr
    assert "EVAL_CONFIG" not in launched_env
    assert "EVAL_INTERVAL" not in launched_env
    assert "EVAL_MAX_CONTEXT_LEN" not in launched_env
    assert "EVAL_MAX_PROMPT_LEN" not in launched_env
    assert "EVAL_MAX_RESPONSE_LEN" not in launched_env


@pytest.mark.parametrize("launcher", [GPU_VISION_LAUNCHER, CPU_VISION_LAUNCHER])
def test_visual_xor_debug_launcher_rejects_invalid_enable_eval(launcher, tmp_path):
    result, _ = _run_debug_launcher(
        launcher,
        tmp_path,
        enable_eval="sometimes",
        eval_config_exists=True,
    )

    assert result.returncode != 0
    assert "ENABLE_EVAL" in result.stderr
    assert "0 or 1" in result.stderr
