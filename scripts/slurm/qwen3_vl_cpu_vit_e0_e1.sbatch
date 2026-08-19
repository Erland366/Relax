#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

#SBATCH --job-name=relax-cpu-vit-e0-e1
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
#SBATCH --time=12:00:00
#SBATCH --chdir=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
#SBATCH --output=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610/log/slurm-cpu-vit-e0-e1-%j.log

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
ARTIFACT_DIR="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_${SLURM_JOB_ID}_e0_e1"
MANIFEST_PATH="${ARTIFACT_DIR}/manifest.tsv"

unset VIRTUAL_ENV RAY_ADDRESS MASTER_ADDR CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_PATH}/bin:${CONDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHON="${CONDA_ENV_PATH}/bin/python"

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"

assert_no_conflicting_user_jobs() {
    local current_user active_job_ids job_id
    current_user="$(id -un)"
    if ! active_job_ids="$(squeue -h -u "${current_user}" -w "$(hostname -s)" -t R,CG -o '%A')"; then
        echo "Unable to query active Slurm jobs on $(hostname -s); refusing local Ray cleanup" >&2
        return 1
    fi

    while IFS= read -r job_id; do
        if [ -n "${job_id}" ] && [ "${job_id}" != "${SLURM_JOB_ID}" ]; then
            echo "Refusing to start Relax on shared node $(hostname -s): active job ${job_id} belongs to ${current_user}" >&2
            return 1
        fi
    done <<< "${active_job_ids}"
}

cleanup_ray() {
    set +e
    if assert_no_conflicting_user_jobs; then
        timeout 60 ray serve shutdown -y
        timeout 60 ray stop --force
    else
        echo "Skipping broad Ray cleanup because another user-owned Slurm job is active on this node" >&2
    fi
}
trap cleanup_ray EXIT

for required_path in \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${REFINEMENT_DATA}/refinement_rl_eval.parquet" \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${SGLANG_PYTHON_DIR}" \
    "${SGL_KERNEL_BUILD_DIR}" \
    "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" \
    "${MEGATRON_DIR}" \
    "${ROCM_HSA_RUNTIME_PRELOAD}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required CPU-ViT experiment input does not exist: ${required_path}" >&2
        exit 2
    fi
done

assert_no_conflicting_user_jobs

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
if [ "$(command -v python)" != "${PYTHON}" ]; then
    echo "Conda activation resolved unexpected Python: $(command -v python); expected ${PYTHON}" >&2
    exit 2
fi

export CONDA_ENV_NAME
export RELAX_ROOT_DIR="${ROOT_DIR}"
export HF_CHECKPOINT
export REFINEMENT_DATA
export RELAX_SGL_KERNEL_BUILD_DIR="${SGL_KERNEL_BUILD_DIR}"
export SGLANG_PYTHON_DIR
export MEGATRON_BRIDGE_DIR
export MEGATRON_DIR
export ROCM_HSA_RUNTIME_PRELOAD
export PYTHONPATH="${SGL_KERNEL_BUILD_DIR}:${SGLANG_PYTHON_DIR}:${MEGATRON_BRIDGE_DIR}/src:${MEGATRON_DIR}:${ROOT_DIR}"
export LD_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
export HIP_VISIBLE_DEVICES=0,1,2,3
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE=0,1,2,3

printf 'stage\tprofile\trollout_batch_size\tn_samples_per_prompt\tstarted_at\tended_at\tconsole_log\tinternal_run_log\n' > "${MANIFEST_PATH}"

echo "=== CPU-ViT Slurm allocation ==="
echo "job_id=${SLURM_JOB_ID} host=$(hostname -s) cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
taskset -pc "$$"
python -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas 4 \
    --vision-encoder-num-cpus 2

echo "=== E0 Hugging Face CPU/native parity ==="
python -m examples.visual_xor.validate_cpu_vision_parity \
    --checkpoint "${HF_CHECKPOINT}" \
    --dataset "${REFINEMENT_DATA}/refinement_rl_eval.parquet" \
    --output "${ARTIFACT_DIR}/e0_hf_parity.json" \
    --device cuda:0 \
    --num-images 8 2>&1 | tee "${ARTIFACT_DIR}/e0_hf_parity.log"

echo "=== E0 live scalar and grouped CPU parity ==="
set +e
HIP_VISIBLE_DEVICES=0 python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
    --checkpoint "${HF_CHECKPOINT}" \
    --dataset "${REFINEMENT_DATA}/refinement_rl_eval.parquet" \
    --output "${ARTIFACT_DIR}/e0_sglang_parity.json" \
    --num-images 8 \
    --parallel-samples 8 \
    --host 127.0.0.1 \
    --port 31000 \
    --base-gpu-id 0 2>&1 | tee "${ARTIFACT_DIR}/e0_sglang_parity.log"
live_parity_status="${PIPESTATUS[0]}"
set -e

python - "${ARTIFACT_DIR}/e0_sglang_parity.json" "${live_parity_status}" <<'PY'
import json
import sys
from pathlib import Path


artifact_path = Path(sys.argv[1])
cli_status = int(sys.argv[2])
artifact = json.loads(artifact_path.read_text())
summary = artifact["summary"]
gates = artifact["gates"]
scalar_passed = (
    summary["generated_token_match_rate"] >= gates["generated_token_match_rate_min"]
    and summary["text_match_rate"] >= gates["text_match_rate_min"]
    and summary["max_action_logprob_absolute_delta"]
    <= gates["action_logprob_absolute_delta_max"]
    and summary["max_action_margin_absolute_delta"]
    <= gates["action_margin_absolute_delta_max"]
    and summary["max_action_probability_absolute_delta"]
    <= gates["action_probability_absolute_delta_max"]
)
grouped_cpu_passed = artifact["parallel_sampling"]["passed"]
grouped_native_passed = artifact["native_parallel_sampling"]["passed"]
print(
    "E0 live parity: "
    f"scalar_cpu={scalar_passed}, grouped_cpu={grouped_cpu_passed}, "
    f"grouped_native={grouped_native_passed}, cli_status={cli_status}"
)
if not scalar_passed or not grouped_cpu_passed:
    raise SystemExit("E0 CPU parity gate failed")
PY

profiles=(D0 D1 D2 D3)
rollout_batch_sizes=(8 16 32 64)
n_samples_per_prompt=(8 4 2 1)

for index in "${!profiles[@]}"; do
    profile="${profiles[index]}"
    rollout_batch_size="${rollout_batch_sizes[index]}"
    samples_per_prompt="${n_samples_per_prompt[index]}"
    profile_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    marker_path="${ARTIFACT_DIR}/${profile}.started"
    console_log="${ARTIFACT_DIR}/e1_${profile,,}.console.log"
    touch "${marker_path}"

    assert_no_conflicting_user_jobs
    echo "=== E1 ${profile}: rollout_batch=${rollout_batch_size}, samples_per_prompt=${samples_per_prompt} ==="
    env \
        CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
        CPU_VISION_CAPACITY_MODE=1 \
        ENABLE_EVAL=0 \
        HF_CHECKPOINT="${HF_CHECKPOINT}" \
        REFINEMENT_DATA="${REFINEMENT_DATA}" \
        PROMPT_SET="${REFINEMENT_DATA}/refinement_rl_train.parquet" \
        NUM_ROLLOUT=12 \
        NUM_STEPS_PER_ROLLOUT=2 \
        ROLLOUT_BATCH_SIZE="${rollout_batch_size}" \
        N_SAMPLES_PER_PROMPT="${samples_per_prompt}" \
        GLOBAL_BATCH_SIZE=32 \
        MAX_STALENESS=4 \
        RAY_NUM_CPUS=16 \
        VISION_ENCODER_NUM_REPLICAS=1 \
        VISION_ENCODER_NUM_CPUS=1 \
        VISION_ENCODER_OMIT_GPU_WEIGHTS=1 \
        VISION_ENCODER_CACHE_MAX_BYTES=1 \
        VISION_ENCODER_MAX_BATCH_SIZE=8 \
        VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0 \
        RELAX_SGL_KERNEL_BUILD_DIR="${SGL_KERNEL_BUILD_DIR}" \
        WANDB_MODE=online \
        bash "${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh" \
        2>&1 | tee "${console_log}"

    cleanup_ray

    profile_ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    internal_run_log="$({
        find "${ROOT_DIR}/log" -maxdepth 1 -type f \
            -name 'visual-xor-refinement-cpu-omitted-*.log' \
            -newer "${marker_path}" -printf '%T@\t%p\n'
    } | sort -n | tail -n 1 | cut -f 2-)"
    if [ -z "${internal_run_log}" ]; then
        echo "E1 ${profile} completed without a discoverable run log" >&2
        exit 2
    fi
    printf 'E1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${profile}" \
        "${rollout_batch_size}" \
        "${samples_per_prompt}" \
        "${profile_started_at}" \
        "${profile_ended_at}" \
        "${console_log}" \
        "${internal_run_log}" >> "${MANIFEST_PATH}"
done

echo "E0-E1 CPU-ViT job completed; manifest=${MANIFEST_PATH}"
