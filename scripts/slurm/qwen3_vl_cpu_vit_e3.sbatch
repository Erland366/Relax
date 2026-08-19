#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

#SBATCH --job-name=relax-cpu-vit-e3
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=qirong_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:mi210:4
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
#SBATCH --output=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610/log/slurm-cpu-vit-e3-%j.log

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
E2_ARTIFACT="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_142801_e2/e2_scaling.json"
ARTIFACT_DIR="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_${SLURM_JOB_ID}_e3"
MANIFEST_PATH="${ARTIFACT_DIR}/manifest.tsv"

unset VIRTUAL_ENV RAY_ADDRESS MASTER_ADDR CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_PATH}/bin:${CONDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHON="${CONDA_ENV_PATH}/bin/python"

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"

assert_no_conflicting_user_jobs() {
    local current_user active_job_ids job_id
    current_user="$(id -un)"
    active_job_ids="$(squeue -h -u "${current_user}" -w "$(hostname -s)" -t R,CG -o '%A')" || {
        echo "Unable to query active Slurm jobs; refusing local Ray cleanup" >&2
        return 1
    }
    while IFS= read -r job_id; do
        if [ -n "${job_id}" ] && [ "${job_id}" != "${SLURM_JOB_ID}" ]; then
            echo "Refusing local Ray cleanup: job ${job_id} also belongs to ${current_user}" >&2
            return 1
        fi
    done <<< "${active_job_ids}"
}

cleanup_ray() {
    set +e
    if assert_no_conflicting_user_jobs; then
        timeout 60 ray serve shutdown -y
        timeout 60 ray stop --force
    fi
}
trap cleanup_ray EXIT

for required_path in \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${SGLANG_PYTHON_DIR}" \
    "${SGL_KERNEL_BUILD_DIR}" \
    "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" \
    "${MEGATRON_DIR}" \
    "${ROCM_HSA_RUNTIME_PRELOAD}" \
    "${E2_ARTIFACT}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required CPU-ViT E3 input does not exist: ${required_path}" >&2
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

python - "${E2_ARTIFACT}" <<'PY'
import json
import sys
from pathlib import Path


artifact = json.loads(Path(sys.argv[1]).read_text())
recommended = artifact["recommended_configuration"]
if (recommended["num_replicas"], recommended["threads_per_replica"]) != (1, 1):
    raise SystemExit(f"E3 wrapper expects the measured E2 winner to be 1x1, got {recommended}")
PY

printf 'stage\tprofile\treplicas\tthreads_per_replica\tstarted_at\tended_at\tconsole_log\tinternal_run_log\n' \
    > "${MANIFEST_PATH}"

python -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas 4 \
    --vision-encoder-num-cpus 2

profiles=(D2_1x4 D2_2x2 D2_4x1)
replicas=(1 2 4)
threads=(4 2 1)

for index in "${!profiles[@]}"; do
    profile="${profiles[index]}"
    replica_count="${replicas[index]}"
    thread_count="${threads[index]}"
    profile_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    marker_path="${ARTIFACT_DIR}/${profile}.started"
    console_log="${ARTIFACT_DIR}/${profile,,}.console.log"
    touch "${marker_path}"

    assert_no_conflicting_user_jobs
    echo "=== E3 ${profile}: replicas=${replica_count}, threads=${thread_count} ==="
    # The live four-CPU ViT layouts need four additional Ray CPUs beyond
    # the 16-CPU baseline so Serve can schedule Rollout outside the GPU
    # placement groups. ViT reservations remain exactly four CPUs.
    env \
        CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
        CPU_VISION_CAPACITY_MODE=1 \
        ENABLE_EVAL=0 \
        HF_CHECKPOINT="${HF_CHECKPOINT}" \
        REFINEMENT_DATA="${REFINEMENT_DATA}" \
        PROMPT_SET="${REFINEMENT_DATA}/refinement_rl_train.parquet" \
        NUM_ROLLOUT=12 \
        NUM_STEPS_PER_ROLLOUT=2 \
        ROLLOUT_BATCH_SIZE=32 \
        N_SAMPLES_PER_PROMPT=2 \
        GLOBAL_BATCH_SIZE=32 \
        MAX_STALENESS=4 \
        RAY_NUM_CPUS=20 \
        VISION_ENCODER_NUM_REPLICAS="${replica_count}" \
        VISION_ENCODER_NUM_CPUS="${thread_count}" \
        VISION_ENCODER_OMIT_GPU_WEIGHTS=1 \
        VISION_ENCODER_CACHE_MAX_BYTES=1 \
        VISION_ENCODER_MAX_BATCH_SIZE=8 \
        VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0 \
        SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=0 \
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
        echo "E3 ${profile} completed without a discoverable run log" >&2
        exit 2
    fi
    printf 'E3\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${profile}" "${replica_count}" "${thread_count}" \
        "${profile_started_at}" "${profile_ended_at}" "${console_log}" "${internal_run_log}" \
        >> "${MANIFEST_PATH}"
done

echo "E3 CPU-ViT live topology job completed; manifest=${MANIFEST_PATH}"
