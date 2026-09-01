#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

#SBATCH --job-name=relax-cpu-vision-scaling
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
#SBATCH --time=04:00:00
#SBATCH --chdir=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
#SBATCH --output=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610/log/slurm-cpu-vision-scaling-%j.log

set -euo pipefail

ROOT_DIR="/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610"
REFINEMENT_DATA="/vast/users/qirong.ho/erland/Python_project/SFT_training/visual_xor_refinement_data"
HF_CHECKPOINT="/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft"
CONDA_ENV_NAME="relaxrl_rocm_after_fix"
CONDA_ROOT="/vast/users/qirong.ho/miniforge3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
ROCM_HSA_RUNTIME_PRELOAD="/opt/rocm-7.0.0/lib/libhsa-runtime64.so.1"
CPU_DEMAND_ARTIFACT="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand.json"
CPU_DEMAND_PEAK="5.858835639941522"
ARTIFACT_DIR="${ROOT_DIR}/benchmark_results/cpu_vision/slurm_${SLURM_JOB_ID}_e2"
OUTPUT_PATH="${ARTIFACT_DIR}/cpu_vision_scaling.json"

unset VIRTUAL_ENV RAY_ADDRESS MASTER_ADDR CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_PATH}/bin:${CONDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHON="${CONDA_ENV_PATH}/bin/python"
export PYTHONPATH="${ROOT_DIR}"
export LD_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"

for required_path in \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    "${ROCM_HSA_RUNTIME_PRELOAD}" \
    "${CPU_DEMAND_ARTIFACT}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required CPU-ViT CPU scaling input does not exist: ${required_path}" >&2
        exit 2
    fi
done

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
if [ "$(command -v python)" != "${PYTHON}" ]; then
    echo "Conda activation resolved unexpected Python: $(command -v python); expected ${PYTHON}" >&2
    exit 2
fi

python - "${CPU_DEMAND_ARTIFACT}" "${CPU_DEMAND_PEAK}" <<'PY'
import json
import math
import sys
from pathlib import Path


artifact = json.loads(Path(sys.argv[1]).read_text())
configured_peak = float(sys.argv[2])
measured_peak = float(artifact["peak_unique_images_per_second"])
if not math.isclose(configured_peak, measured_peak, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        f"Configured CPU vision demand peak demand {configured_peak} does not match artifact {measured_peak}"
    )
PY

echo "=== CPU-ViT CPU scaling Slurm allocation ==="
echo "job_id=${SLURM_JOB_ID} host=$(hostname -s) cpus_per_task=${SLURM_CPUS_PER_TASK}"
srun --exact --ntasks=1 --cpus-per-task=16 --hint=nomultithread --cpu-bind=cores \
    bash -c 'taskset -pc "$$"'
srun --exact --ntasks=1 --cpus-per-task=16 --hint=nomultithread --cpu-bind=cores \
    python -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas 4 \
    --vision-encoder-num-cpus 2

echo "=== CPU scaling offline topology and explicit-batch sweep ==="
srun --exact --ntasks=1 --cpus-per-task=16 --hint=nomultithread --cpu-bind=cores \
    python -m examples.visual_xor.measure_cpu_vision_scaling \
    --checkpoint "${HF_CHECKPOINT}" \
    --dataset "${REFINEMENT_DATA}/refinement_rl_train.parquet" \
    --output "${OUTPUT_PATH}" \
    --peak-unique-images-per-second "${CPU_DEMAND_PEAK}" \
    --replica-counts 1,2,4 \
    --thread-counts 1,2,4,8 \
    --batch-sizes 1,2,4,8 \
    --max-total-cpus 8 \
    --num-images 64 \
    --repeats 5

python - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path


artifact_path = Path(sys.argv[1])
artifact = json.loads(artifact_path.read_text())
if len(artifact["results"]) != 36:
    raise SystemExit(f"CPU scaling expected 36 matrix records, got {len(artifact['results'])}")
if any(record["num_images"] != 320 for record in artifact["results"]):
    raise SystemExit("CPU scaling contains a matrix record without exactly 64 images x 5 repeats")
if artifact["recommended_configuration"] is None:
    raise SystemExit("CPU scaling found no CPU layout meeting the 1.25x capacity gate")
print(json.dumps(artifact["recommended_configuration"], indent=2))
print(json.dumps(artifact["batching_recommendations"], indent=2))
print(f"CPU scaling artifact: {artifact_path}")
PY
