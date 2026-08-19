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
NATIVE_GPU_LAUNCHER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"
E3_SLURM_LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "qwen3_vl_cpu_vit_e3.sbatch"
E4_CACHE_SLURM_LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "qwen3_vl_cpu_vit_e4_cache.sbatch"


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


def test_cpu_vision_launchers_forward_batch_wait_timeout_with_disabled_default():
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


def test_cpu_vision_launchers_forward_sglang_feature_cache_with_disabled_default():
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


def test_cpu_vision_live_slurm_matrix_keeps_batching_disabled_after_e2_negative_gate():
    launcher = E3_SLURM_LAUNCHER.read_text()

    assert "#SBATCH --cpus-per-task=20" in launcher
    assert "profiles=(D2_1x4 D2_2x2 D2_4x1)" in launcher
    assert "replicas=(1 2 4)" in launcher
    assert "threads=(4 2 1)" in launcher
    assert "RAY_NUM_CPUS=20" in launcher
    assert "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0" in launcher
    assert "SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=0" in launcher


def test_cpu_vision_cache_slurm_matrix_isolates_producer_and_transport_reuse():
    launcher = E4_CACHE_SLURM_LAUNCHER.read_text()

    assert "profiles=(tier1_only tier1_tier2)" in launcher
    assert "producer_cache_bytes=(1073741824 1073741824)" in launcher
    assert "sglang_cache_bytes=(0 1073741824)" in launcher
    assert "VISION_ENCODER_NUM_REPLICAS=1" in launcher
    assert "VISION_ENCODER_NUM_CPUS=1" in launcher
    assert "VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0" in launcher


def test_cpu_vision_cache_slurm_requires_complete_profiles_and_runs_analysis():
    launcher = E4_CACHE_SLURM_LAUNCHER.read_text()

    assert "validate_completed_profile" in launcher
    assert "expected_cycles = set(range(12))" in launcher
    assert "incomplete CPU vision metrics" in launcher
    assert "examples.visual_xor.analyze_cpu_vision_cache" in launcher
    assert '--profile "tier1_only=${ARTIFACT_DIR}/tier1_only.console.log"' in launcher
    assert '--profile "tier1_tier2=${ARTIFACT_DIR}/tier1_tier2.console.log"' in launcher


def test_training_launcher_adds_gpu_weight_omission_only_for_boolean_one():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'VISION_ENCODER_OMIT_GPU_WEIGHTS="${VISION_ENCODER_OMIT_GPU_WEIGHTS:-0}"' in launcher
    assert "require_boolean_flag VISION_ENCODER_OMIT_GPU_WEIGHTS" in launcher

    conditional_append = re.search(
        r'if \[ "\$\{VISION_ENCODER_OMIT_GPU_WEIGHTS\}" = "1" \]; then\s+'
        r"(?P<array>[A-Z_]+)\+=\(--vision-encoder-omit-gpu-weights\)\s+fi",
        launcher,
    )
    assert conditional_append is not None

    omission_args = conditional_append.group("array")
    assert f'"${{{omission_args}[@]}}"' in launcher
    assert launcher.count("--vision-encoder-omit-gpu-weights") == 1
    assert "omit_gpu_weights=${VISION_ENCODER_OMIT_GPU_WEIGHTS}" in launcher


def test_fully_async_cpu_vision_launcher_keeps_gpu_weight_omission_opt_in():
    launcher = CPU_VISION_LAUNCHER.read_text()

    assert 'export VISION_ENCODER_NUM_CPUS="${VISION_ENCODER_NUM_CPUS:-1}"' in launcher
    assert 'export VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"' in launcher
    assert 'export VISION_ENCODER_OMIT_GPU_WEIGHTS="${VISION_ENCODER_OMIT_GPU_WEIGHTS:-0}"' in launcher
    assert "CPU_VISION_MODE=cpu-omitted" in launcher
    assert "CPU_VISION_MODE=cpu-resident" in launcher
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


def test_native_gpu_launcher_accepts_the_same_two_cycle_matrix_overrides():
    launcher = NATIVE_GPU_LAUNCHER.read_text()

    assert 'export NUM_ROLLOUT="${NUM_ROLLOUT:-250}"' in launcher
    assert 'export EVAL_INTERVAL="${EVAL_INTERVAL:-4}"' in launcher
    assert 'export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"' in launcher
    assert "export FREEZE_VISION_MODEL=1" in launcher
    assert "visual-xor-refinement-native-gpu-${RUN_TAG}.log" in launcher


def test_training_launcher_can_freeze_native_gpu_vision_for_comparable_benchmarks():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'FREEZE_VISION_MODEL="${FREEZE_VISION_MODEL:-0}"' in launcher
    assert "require_boolean_flag FREEZE_VISION_MODEL" in launcher
    assert 'if [ "${VISION_ENCODER_BACKEND:-disabled}" != "disabled" ]; then' in launcher
    assert "FREEZE_VISION_MODEL=1" in launcher
    assert 'if [ "${FREEZE_VISION_MODEL}" = "1" ]; then' in launcher


@pytest.mark.parametrize("launcher", [NATIVE_GPU_LAUNCHER, CPU_VISION_LAUNCHER])
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


@pytest.mark.parametrize("launcher", [NATIVE_GPU_LAUNCHER, CPU_VISION_LAUNCHER])
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


@pytest.mark.parametrize("launcher", [NATIVE_GPU_LAUNCHER, CPU_VISION_LAUNCHER])
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
