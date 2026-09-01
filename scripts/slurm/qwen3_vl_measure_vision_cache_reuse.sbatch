#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

#SBATCH --job-name=relax-vision-cache-reuse
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
#SBATCH --time=01:00:00
#SBATCH --chdir=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
#SBATCH --output=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610/log/slurm-vision-cache-reuse-%j.log

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
CPU_DEMAND_ARTIFACT="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand.json"
CPU_SCALING_ARTIFACT="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_142801_cpu_vision_scaling/cpu_vision_scaling.json"
ARTIFACT_DIR="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_${SLURM_JOB_ID}_vision_cache_reuse"
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

validate_completed_profile() {
    local console_log="$1"
    local profile="$2"
    "${PYTHON}" - "${console_log}" "${profile}" <<'PY'
import sys

from examples.visual_xor.measure_vision_cache_reuse import _parse_metrics_log


console_log, profile = sys.argv[1:]
vision_cycles, rollout_cycles, training_cycles = _parse_metrics_log(console_log)
expected_cycles = set(range(12))
observed = {
    "vision": set(vision_cycles),
    "rollout": set(rollout_cycles),
    "actor": set(training_cycles),
}
missing = {name: sorted(expected_cycles - cycles) for name, cycles in observed.items()}
extra = {name: sorted(cycles - expected_cycles) for name, cycles in observed.items()}
if any(missing.values()) or any(extra.values()):
    raise SystemExit(
        f"vision cache reuse cache profile {profile} has incomplete CPU vision metrics: "
        f"missing={missing}, extra={extra}"
    )
PY
}

for required_path in \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${SGLANG_PYTHON_DIR}" \
    "${SGL_KERNEL_BUILD_DIR}" \
    "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" \
    "${MEGATRON_DIR}" \
    "${ROCM_HSA_RUNTIME_PRELOAD}" \
    "${CPU_DEMAND_ARTIFACT}" \
    "${CPU_SCALING_ARTIFACT}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required CPU-ViT vision cache reuse cache input does not exist: ${required_path}" >&2
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

python - "${CPU_DEMAND_ARTIFACT}" "${CPU_SCALING_ARTIFACT}" <<'PY'
import json
import sys
from pathlib import Path


cpu_demand = json.loads(Path(sys.argv[1]).read_text())
cpu_scaling = json.loads(Path(sys.argv[2]).read_text())
recommended = cpu_scaling["recommended_configuration"]
if (recommended["num_replicas"], recommended["threads_per_replica"]) != (1, 1):
    raise SystemExit(f"vision cache reuse cache wrapper expects the measured CPU scaling winner to be 1x1, got {recommended}")
if "prompts32_samples2" not in cpu_demand["profiles"]:
    raise SystemExit("vision cache reuse cache wrapper requires the completed CPU vision demand prompts32_samples2 no-cache control")
PY

printf 'stage\tprofile\tcpu_cache_bytes\tsglang_cache_bytes\tstarted_at\tended_at\tconsole_log\tinternal_run_log\n' \
    > "${MANIFEST_PATH}"

python -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas 1 \
    --vision-encoder-num-cpus 1

# CPU vision demand prompts32_samples2 is the cache-disabled control. These two profiles isolate CPU encode
# reuse (CPU cache) from repeated feature serialization/transport reuse (SGLang cache).
profiles=(cpu_cache_only cpu_and_sglang_cache)
cpu_cache_bytes=(1073741824 1073741824)
sglang_cache_bytes=(0 1073741824)

for index in "${!profiles[@]}"; do
    profile="${profiles[index]}"
    cpu_cache_bytes="${cpu_cache_bytes[index]}"
    sglang_bytes="${sglang_cache_bytes[index]}"
    profile_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    marker_path="${ARTIFACT_DIR}/${profile}.started"
    console_log="${ARTIFACT_DIR}/${profile}.console.log"
    touch "${marker_path}"

    assert_no_conflicting_user_jobs
    echo "=== vision cache reuse cache ${profile}: CPU cache=${cpu_cache_bytes}, sglang=${sglang_bytes} ==="
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
        RAY_NUM_CPUS=16 \
        VISION_ENCODER_NUM_REPLICAS=1 \
        VISION_ENCODER_NUM_CPUS=1 \
        SKIP_GPU_VISION_ENCODER=1 \
        VISION_ENCODER_CACHE_MAX_BYTES="${cpu_cache_bytes}" \
        VISION_ENCODER_MAX_IMAGES_PER_REQUEST=8 \
        VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0 \
        SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${sglang_bytes}" \
        RELAX_SGL_KERNEL_BUILD_DIR="${SGL_KERNEL_BUILD_DIR}" \
        WANDB_MODE=online \
        bash "${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh" \
        2>&1 | tee "${console_log}"

    validate_completed_profile "${console_log}" "${profile}"
    cleanup_ray
    profile_ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    internal_run_log="$({
        find "${ROOT_DIR}/log" -maxdepth 1 -type f \
            -name 'visual-xor-refinement-cpu-skip-gpu-encoder-*.log' \
            -newer "${marker_path}" -printf '%T@\t%p\n'
    } | sort -n | tail -n 1 | cut -f 2-)"
    if [ -z "${internal_run_log}" ]; then
        echo "vision cache reuse cache ${profile} completed without a discoverable run log" >&2
        exit 2
    fi
    printf 'vision_cache_reuse\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${profile}" "${cpu_cache_bytes}" "${sglang_bytes}" \
        "${profile_started_at}" "${profile_ended_at}" "${console_log}" "${internal_run_log}" \
        >> "${MANIFEST_PATH}"
done

python -m examples.visual_xor.measure_vision_cache_reuse \
    --profile "cpu_cache_only=${ARTIFACT_DIR}/cpu_cache_only.console.log" \
    --profile "cpu_and_sglang_cache=${ARTIFACT_DIR}/cpu_and_sglang_cache.console.log" \
    --output "${ARTIFACT_DIR}/vision_cache_reuse.json"

echo "vision cache reuse CPU-ViT cache confirmation completed; manifest=${MANIFEST_PATH}; analysis=${ARTIFACT_DIR}/vision_cache_reuse.json"
