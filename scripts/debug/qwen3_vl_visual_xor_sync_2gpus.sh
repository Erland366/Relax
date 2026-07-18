#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export PROMPT_SET="/vast/users/qirong.ho/erland/Python_project/SFT_training/visual_xor_data/xor_train.parquet"
export HF_CHECKPOINT="/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-bootstrap-sft"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the accepted qwen3-vl-0.37b visual SFT directory}"
: "${PROMPT_SET:?Set PROMPT_SET to the generated xor_train.parquet file}"
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    echo "HF_CHECKPOINT does not contain config.json: ${HF_CHECKPOINT}" >&2
    exit 2
fi
if [ ! -f "${PROMPT_SET}" ]; then
    echo "PROMPT_SET does not exist: ${PROMPT_SET}" >&2
    exit 2
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra VISUAL_XOR_GPU_IDS <<< "${HIP_VISIBLE_DEVICES}"
if [ "${#VISUAL_XOR_GPU_IDS[@]}" -ne 2 ]; then
    echo "HIP_VISIBLE_DEVICES must contain exactly two GPU IDs, got ${HIP_VISIBLE_DEVICES}" >&2
    exit 2
fi
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE="${HIP_VISIBLE_DEVICES}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES FULLY_ASYNC HYBRID

# This task intentionally hard-pins the simple two-service graph so ambient
# debug variables cannot silently turn it into an async or colocated run.
export RELAX_EXECUTION_MODE=sync
export USE_COLLOCATE=0
export MAX_STALENESS=0
export RAY_NUM_GPUS=2
export NUM_GPUS_PER_NODE=2
export ACTOR_RESOURCE_GPUS=1
export ACTOR_FWD_RESOURCE_GPUS=0
export ROLLOUT_RESOURCE_GPUS=1
export RESOURCE_JSON='{"actor":[1,1],"rollout":[1,1]}'
export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1
export ENABLE_SEQUENCE_PARALLEL=0
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export SGLANG_PIPELINE_PARALLEL_SIZE=1
export SGLANG_DATA_PARALLEL_SIZE=1
export SGLANG_EXPERT_PARALLEL_SIZE=1
export SGLANG_ENABLE_DP_ATTENTION=0

export LR="${LR:-3e-6}"

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_ASSET_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)/relax_assets"
export ASSET_DIR="${ASSET_DIR:-${DEFAULT_ASSET_DIR}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"
export MODEL_CONFIG_NAME=qwen3-vl-0.37B
export MODEL_ASSET_NAME=Qwen3-VL-0.37B-Visual-Bootstrap-SFT
export MODEL_LOG_NAME=qwen3-vl-0.37b-visual-xor
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
export MULTIMODAL_KEYS='{"image":"image"}'

export RM_TYPE=visual_xor
export REWARD_KEY=score
export USE_KL_LOSS=0
export USE_BALANCE_DATA=1
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export NUM_STEPS_PER_ROLLOUT=1
# Stable visual-XOR candidate: four images with eight samples each. The
# 64-sample variant wedged in ROCm forward/backward during the full-data run.
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=32
export MICRO_BATCH_SIZE=1
export SEQ_LENGTH=512
export ROLLOUT_MAX_CONTEXT_LEN=512
export ROLLOUT_MAX_PROMPT_LEN=511
export ROLLOUT_MAX_RESPONSE_LEN=3
export UPDATE_WEIGHTS_INTERVAL=1
export ENABLE_RECOMPUTE=1

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1
export SGLANG_SERVER_CONCURRENCY=16
export SGLANG_MAX_RUNNING_REQUESTS=16
export SGLANG_MAX_TOTAL_TOKENS=8192
export SGLANG_MEM_FRACTION_STATIC=0.4

export SAVE_CHECKPOINTS=0
unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG LOAD_DIR

export RUN_LOG="${ROOT_DIR}/log/visual-xor-${RUN_TAG}.log"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
# Keep this task out of an inherited text-task W&B project or group.
export WANDB_PROJECT=relax-amd-visual-xor
export WANDB_GROUP="visual-xor-sync-${RUN_TAG}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/log/wandb}"

bash scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
