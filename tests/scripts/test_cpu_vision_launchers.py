import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAUNCHER = REPO_ROOT / "scripts" / "training" / "multimodal" / "amd_qwen3_4b_2gpu_e2e.sh"
CPU_VISION_LAUNCHER = (
    REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"
)
NATIVE_GPU_LAUNCHER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"


def _run_debug_launcher(
    launcher: Path,
    tmp_path: Path,
    *,
    enable_eval: str | None,
    eval_config_exists: bool,
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
    fake_bash = fake_bin / "bash"
    fake_bash.write_text('#!/bin/sh\nenv > "$CAPTURE_ENV"\n')
    fake_bash.chmod(0o755)

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
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PROMPT_SET": str(prompt_set),
        }
    )
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


def test_training_launcher_forwards_validated_cpu_vision_replica_count():
    launcher = TRAINING_LAUNCHER.read_text()

    assert 'VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"' in launcher
    assert "require_positive_integer VISION_ENCODER_NUM_REPLICAS" in launcher
    assert '--vision-encoder-num-replicas "${VISION_ENCODER_NUM_REPLICAS}"' in launcher
    assert "replicas=${VISION_ENCODER_NUM_REPLICAS}" in launcher


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
