#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

#SBATCH --job-name=relax-cpu-vision-matched
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=qirong_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:mi210:4
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --chdir=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
#SBATCH --output=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610/log/slurm-compare-vision-settings-%j.log

set -euo pipefail

ROOT_DIR="/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610"
REFINEMENT_DATA="/vast/users/qirong.ho/erland/Python_project/SFT_training/visual_xor_refinement_data"
HF_CHECKPOINT="/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft"
CONDA_ENV_NAME="relaxrl_rocm_after_fix"
CONDA_ROOT="/vast/users/qirong.ho/miniforge3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
SGLANG_PYTHON_DIR="/vast/users/qirong.ho/erland/Python_project/sglang/python"
SGL_KERNEL_BUILD_DIR="/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312"
MEGATRON_BRIDGE_DIR="/vast/users/qirong.ho/erland/Python_project/Megatron-Bridge"
MEGATRON_DIR="/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM"
ROCM_HSA_RUNTIME_PRELOAD="/opt/rocm-7.0.0/lib/libhsa-runtime64.so.1"

orders=(
    "gpu:0:0 cpu:0:0 cpu:1:0 cpu:1:1"
    "cpu:0:0 cpu:1:0 cpu:1:1 gpu:0:0"
    "cpu:1:0 cpu:1:1 gpu:0:0 cpu:0:0"
    "cpu:1:1 gpu:0:0 cpu:0:0 cpu:1:0"
)
train_seeds=(1234 1235 1236 1237)
rollout_seeds=(42 43 44 45)

if [ "${VISION_SETTINGS_DRY_RUN:-0}" = "1" ]; then
    for repeat_index in "${!orders[@]}"; do
        echo "repeat_$((repeat_index + 1)): train_seed=${train_seeds[repeat_index]} rollout_seed=${rollout_seeds[repeat_index]} order=${orders[repeat_index]}"
    done
    exit 0
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "This experiment must run inside a Slurm allocation; use VISION_SETTINGS_DRY_RUN=1 for local validation." >&2
    exit 2
fi

ARTIFACT_DIR="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_${SLURM_JOB_ID}_vision_settings"
MANIFEST_PATH="${ARTIFACT_DIR}/manifest.tsv"
ANALYSIS_PATH="${ARTIFACT_DIR}/vision_settings.json"

unset VIRTUAL_ENV RAY_ADDRESS MASTER_ADDR CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_PATH}/bin:${CONDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHON="${CONDA_ENV_PATH}/bin/python"

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"

assert_no_conflicting_user_jobs() {
    local current_user active_job_ids job_id
    current_user="$(id -un)"
    active_job_ids="$(squeue -h -u "${current_user}" -w "$(hostname -s)" -t R,CG -o '%A')" || {
        echo "Unable to query active Slurm jobs; refusing Ray cleanup" >&2
        return 1
    }
    while IFS= read -r job_id; do
        if [ -n "${job_id}" ] && [ "${job_id}" != "${SLURM_JOB_ID}" ]; then
            echo "Refusing Ray cleanup: job ${job_id} also belongs to ${current_user} on this node" >&2
            return 1
        fi
    done <<< "${active_job_ids}"
}

cleanup_serve() {
    set +e
    if assert_no_conflicting_user_jobs; then
        timeout 60 ray serve shutdown -y
    fi
    set -e
}
trap cleanup_serve EXIT

validate_completed_run() {
    local console_log="$1"
    local device="$2"
    "${PYTHON}" - "${console_log}" "${device}" <<'PY'
import sys

from examples.visual_xor.compare_vision_settings import _parse_metrics_log


console_log, device = sys.argv[1:]
vision, rollout, train_data, actor = _parse_metrics_log(console_log)
expected = set(range(22))
observed = {"rollout": set(rollout), "train_data": set(train_data), "actor": set(actor)}
if device == "cpu":
    observed["vision"] = set(vision)
missing = {name: sorted(expected - cycles) for name, cycles in observed.items()}
extra = {name: sorted(cycles - expected) for name, cycles in observed.items()}
if any(missing.values()) or any(extra.values()):
    raise SystemExit(f"vision-settings run {device} has incomplete cycles: missing={missing}, extra={extra}")
PY
}

for required_path in \
    "${PYTHON}" \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${SGLANG_PYTHON_DIR}" \
    "${SGL_KERNEL_BUILD_DIR}" \
    "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" \
    "${MEGATRON_DIR}" \
    "${ROCM_HSA_RUNTIME_PRELOAD}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required vision-settings input does not exist: ${required_path}" >&2
        exit 2
    fi
done

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
if [ "$(command -v python)" != "${PYTHON}" ]; then
    echo "Conda activation resolved unexpected Python: $(command -v python); expected ${PYTHON}" >&2
    exit 2
fi

export CONDA_ENV_NAME RELAX_ROOT_DIR="${ROOT_DIR}" HF_CHECKPOINT REFINEMENT_DATA
export RELAX_SGL_KERNEL_BUILD_DIR="${SGL_KERNEL_BUILD_DIR}"
export SGLANG_PYTHON_DIR MEGATRON_BRIDGE_DIR MEGATRON_DIR ROCM_HSA_RUNTIME_PRELOAD
export PYTHONPATH="${SGL_KERNEL_BUILD_DIR}:${SGLANG_PYTHON_DIR}:${MEGATRON_BRIDGE_DIR}/src:${MEGATRON_DIR}:${ROOT_DIR}"
export LD_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
export HIP_VISIBLE_DEVICES=0,1,2,3
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE=0,1,2,3

python -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas 1 \
    --vision-encoder-num-cpus 1

python - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"matched study requires exactly four visible GPUs, found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if not all("MI210" in name for name in names):
    raise SystemExit(f"matched study requires four MI210 GPUs, found {names}")
print(f"GPU preflight passed: {names}")
PY

echo "=== compare Hugging Face CPU and GPU outputs ==="
python -m examples.visual_xor.validate_cpu_vision_parity \
    --checkpoint "${HF_CHECKPOINT}" \
    --dataset "${REFINEMENT_DATA}/refinement_rl_eval.parquet" \
    --output "${ARTIFACT_DIR}/hf_parity.json" \
    --device cuda:0 \
    --num-images 8 2>&1 | tee "${ARTIFACT_DIR}/hf_parity.log"

echo "=== compare live scalar and grouped CPU outputs ==="
set +e
HIP_VISIBLE_DEVICES=0 python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
    --checkpoint "${HF_CHECKPOINT}" \
    --dataset "${REFINEMENT_DATA}/refinement_rl_eval.parquet" \
    --output "${ARTIFACT_DIR}/sglang_parity.json" \
    --num-images 8 \
    --parallel-samples 8 \
    --host 127.0.0.1 \
    --port 31000 \
    --base-gpu-id 0 2>&1 | tee "${ARTIFACT_DIR}/sglang_parity.log"
parity_status="${PIPESTATUS[0]}"
set -e

python - "${ARTIFACT_DIR}/sglang_parity.json" "${parity_status}" <<'PY'
import json
import sys
from pathlib import Path


artifact = json.loads(Path(sys.argv[1]).read_text())
status = int(sys.argv[2])
summary = artifact["summary"]
gates = artifact["gates"]
scalar_passed = (
    summary["generated_token_match_rate"] >= gates["generated_token_match_rate_min"]
    and summary["text_match_rate"] >= gates["text_match_rate_min"]
    and summary["max_action_logprob_absolute_delta"] <= gates["action_logprob_absolute_delta_max"]
    and summary["max_action_margin_absolute_delta"] <= gates["action_margin_absolute_delta_max"]
    and summary["max_action_probability_absolute_delta"] <= gates["action_probability_absolute_delta_max"]
)
grouped_cpu_passed = artifact["parallel_sampling"]["passed"]
grouped_gpu_passed = artifact["gpu_parallel_sampling"]["passed"]
print(
    "CPU/GPU output comparison: "
    f"scalar_cpu={scalar_passed}, grouped_cpu={grouped_cpu_passed}, "
    f"grouped_gpu={grouped_gpu_passed}, cli_status={status}"
)
if not scalar_passed or not grouped_cpu_passed:
    raise SystemExit("CPU/GPU output comparison failed")
PY

git -C "${ROOT_DIR}" rev-parse HEAD > "${ARTIFACT_DIR}/git-head.txt"
git -C "${ROOT_DIR}" status --porcelain=v1 -uall > "${ARTIFACT_DIR}/git-status.txt"
sha256sum \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    > "${ARTIFACT_DIR}/input-sha256.txt"
scontrol show job "${SLURM_JOB_ID}" > "${ARTIFACT_DIR}/slurm-job.txt"
lscpu --extended=CPU,CORE,SOCKET,NODE,ONLINE > "${ARTIFACT_DIR}/cpu-topology.txt"
rocm-smi --showid --showproductname --showmeminfo vram --csv > "${ARTIFACT_DIR}/vram-before.csv"

printf 'repeat\torder_position\tdevice\tcpu_cache\tsglang_cache\ttrain_seed\trollout_seed\tcpu_cache_bytes\tsglang_cache_bytes\tstarted_at\tended_at\texit_status\tconsole_log\tinternal_run_log\n' \
    > "${MANIFEST_PATH}"

analysis_args=()
for repeat_index in "${!orders[@]}"; do
    repeat="repeat_$((repeat_index + 1))"
    train_seed="${train_seeds[repeat_index]}"
    rollout_seed="${rollout_seeds[repeat_index]}"
    read -ra settings <<< "${orders[repeat_index]}"
    for order_position in "${!settings[@]}"; do
        IFS=: read -r device cpu_cache sglang_cache <<< "${settings[order_position]}"
        launcher="${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"
        cpu_cache_bytes=0
        sglang_bytes=0
        skip_gpu_encoder=0
        if [ "${device}" = "cpu" ]; then
            launcher="${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"
            cpu_cache_bytes=1
            skip_gpu_encoder=1
        fi
        if [ "${cpu_cache}" = "1" ]; then
            cpu_cache_bytes=1073741824
        fi
        if [ "${sglang_cache}" = "1" ]; then
            sglang_bytes=1073741824
        fi

        assert_no_conflicting_user_jobs
        cleanup_serve
        run_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        setting_slug="device_${device}-cpu_cache_${cpu_cache}-sglang_cache_${sglang_cache}"
        console_log="${ARTIFACT_DIR}/${repeat}-${setting_slug}.console.log"
        internal_run_log="${ARTIFACT_DIR}/${repeat}-${setting_slug}.internal.log"
        echo "=== vision settings ${repeat} position=${order_position} ${setting_slug}: train_seed=${train_seed} rollout_seed=${rollout_seed} ==="
        set +e
        env \
            CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
            CPU_VISION_CAPACITY_MODE=1 \
            ENABLE_EVAL=0 \
            SAVE_CHECKPOINTS=0 \
            HF_CHECKPOINT="${HF_CHECKPOINT}" \
            REFINEMENT_DATA="${REFINEMENT_DATA}" \
            PROMPT_SET="${REFINEMENT_DATA}/refinement_rl_train.parquet" \
            NUM_ROLLOUT=22 \
            NUM_STEPS_PER_ROLLOUT=2 \
            ROLLOUT_BATCH_SIZE=32 \
            N_SAMPLES_PER_PROMPT=2 \
            GLOBAL_BATCH_SIZE=32 \
            TRAIN_SEED="${train_seed}" \
            ROLLOUT_SEED="${rollout_seed}" \
            MAX_STALENESS=4 \
            RAY_NUM_CPUS=16 \
            VISION_ENCODER_NUM_REPLICAS=1 \
            VISION_ENCODER_NUM_CPUS=1 \
            SKIP_GPU_VISION_ENCODER="${skip_gpu_encoder}" \
            VISION_ENCODER_CACHE_MAX_BYTES="${cpu_cache_bytes}" \
            VISION_ENCODER_MAX_IMAGES_PER_REQUEST=8 \
            VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0 \
            SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${sglang_bytes}" \
            RELAX_SGL_KERNEL_BUILD_DIR="${SGL_KERNEL_BUILD_DIR}" \
            RUN_LOG="${internal_run_log}" \
            WANDB_MODE=online \
            bash "${launcher}" 2>&1 | tee "${console_log}"
        exit_status="${PIPESTATUS[0]}"
        set -e
        run_ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${repeat}" "${order_position}" "${device}" "${cpu_cache}" "${sglang_cache}" \
            "${train_seed}" "${rollout_seed}" "${cpu_cache_bytes}" "${sglang_bytes}" \
            "${run_started_at}" "${run_ended_at}" \
            "${exit_status}" "${console_log}" "${internal_run_log}" >> "${MANIFEST_PATH}"
        if [ "${exit_status}" -ne 0 ]; then
            echo "Vision-settings run ${repeat}:${setting_slug} failed with exit status ${exit_status}" >&2
            exit "${exit_status}"
        fi
        validate_completed_run "${console_log}" "${device}"
        analysis_args+=(--run "${repeat}:${device}:${cpu_cache}:${sglang_cache}=${console_log}")
    done
done

python -m examples.visual_xor.compare_vision_settings \
    "${analysis_args[@]}" \
    --first-steady-cycle 2 \
    --expected-steady-cycles 20 \
    --expected-responses 64 \
    --optimizer-steps-per-cycle 2 \
    --output "${ANALYSIS_PATH}"

rocm-smi --showid --showproductname --showmeminfo vram --csv > "${ARTIFACT_DIR}/vram-after.csv"
cleanup_serve
echo "Matched CPU-vision study completed; manifest=${MANIFEST_PATH}; analysis=${ANALYSIS_PATH}"
