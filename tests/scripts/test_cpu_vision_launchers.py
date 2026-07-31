import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAUNCHER = REPO_ROOT / "scripts" / "training" / "multimodal" / "amd_qwen3_4b_2gpu_e2e.sh"
CPU_VISION_LAUNCHER = (
    REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"
)
NATIVE_GPU_LAUNCHER = REPO_ROOT / "scripts" / "debug" / "qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"


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
    assert 'visual-xor-refinement-${CPU_VISION_MODE}-${RUN_TAG}.log' in launcher


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
