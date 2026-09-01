#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export REFINEMENT_DATA="${REFINEMENT_DATA:-/vast/users/qirong.ho/erland/Python_project/SFT_training/visual_xor_refinement_data}"
export PROMPT_SET="${PROMPT_SET:-${REFINEMENT_DATA}/refinement_rl_train.parquet}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft}"

ENABLE_EVAL="${ENABLE_EVAL:-1}"
if [ "${ENABLE_EVAL}" = "1" ]; then
    export EVAL_CONFIG="${EVAL_CONFIG:-${REFINEMENT_DATA}/refinement_eval_config.json}"
elif [ "${ENABLE_EVAL}" = "0" ]; then
    unset EVAL_CONFIG EVAL_INTERVAL EVAL_MAX_CONTEXT_LEN EVAL_MAX_PROMPT_LEN EVAL_MAX_RESPONSE_LEN
else
    echo "ENABLE_EVAL must be 0 or 1, got ${ENABLE_EVAL}" >&2
    exit 2
fi

if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    echo "HF_CHECKPOINT does not contain config.json: ${HF_CHECKPOINT}" >&2
    echo "Train and accept the visual-XOR refinement SFT before launching Relax." >&2
    exit 2
fi
if [ ! -f "${PROMPT_SET}" ]; then
    echo "PROMPT_SET does not exist: ${PROMPT_SET}" >&2
    exit 2
fi
if [ "${ENABLE_EVAL}" = "1" ] && [ ! -f "${EVAL_CONFIG}" ]; then
    echo "EVAL_CONFIG does not exist: ${EVAL_CONFIG}" >&2
    exit 2
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra VISUAL_XOR_GPU_IDS <<< "${HIP_VISIBLE_DEVICES}"
if [ "${#VISUAL_XOR_GPU_IDS[@]}" -ne 4 ]; then
    echo "HIP_VISIBLE_DEVICES must contain exactly 4 GPU IDs, got ${HIP_VISIBLE_DEVICES}" >&2
    exit 2
fi
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE="${HIP_VISIBLE_DEVICES}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES FULLY_ASYNC HYBRID

# Keep the refinement benchmark deterministic: one actor GPU, one rollout GPU,
# no stale policy, and no ambient async/colocation overrides.
export RELAX_EXECUTION_MODE=fully_async
export USE_COLLOCATE=0
export MAX_STALENESS="${MAX_STALENESS:-4}"
export RAY_NUM_GPUS=4
export NUM_GPUS_PER_NODE=4
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
export ACTOR_RESOURCE_GPUS=2
export ACTOR_FWD_RESOURCE_GPUS=1
export ROLLOUT_RESOURCE_GPUS=1
export TRUE_ON_POLICY_MODE=FALSE
export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,1],"actor_fwd":[1,1],"advantages":[1,0]}'
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
# Keep the GPU baseline equivalent to the CPU-vision modes: compute vision
# on the GPU, but do not train the frozen tower or projector.
export FREEZE_VISION_MODEL=1

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_ASSET_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)/relax_assets"
export ASSET_DIR="${ASSET_DIR:-${DEFAULT_ASSET_DIR}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"
export MODEL_CONFIG_NAME=qwen3-vl-0.37B
export MODEL_ASSET_NAME=Qwen3-VL-0.37B-Visual-XOR-Refinement-SFT
export MODEL_LOG_NAME=qwen3-vl-0.37b-visual-xor-refinement-gpu
export SYSTEM_PROMPT="Answer with exactly one uppercase character: A or B. Do not explain."
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
export MULTIMODAL_KEYS='{"image":"image"}'

export RM_TYPE=visual_xor
export REWARD_KEY=score
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export LR="${LR:-3e-6}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-250}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE=1
export SEQ_LENGTH=512
export ROLLOUT_MAX_CONTEXT_LEN=512
export ROLLOUT_MAX_PROMPT_LEN=511
export ROLLOUT_MAX_RESPONSE_LEN=3
export UPDATE_WEIGHTS_INTERVAL=1
export ENABLE_RECOMPUTE=1

# The 64 training rows repeat every 16 updates in exact 00,01,10,11 groups.
# Streaming and shuffling would destroy that diagnostic invariant.
export USE_STREAMING_DATASET=0
export ROLLOUT_SHUFFLE=0

# Evaluate the held-out images and two anti-shortcut controls before training,
# halfway through each 16-step data cycle, and at every cycle boundary.
if [ "${ENABLE_EVAL}" = "1" ]; then
    export EVAL_INTERVAL="${EVAL_INTERVAL:-4}"
    export EVAL_MAX_CONTEXT_LEN=512
    export EVAL_MAX_PROMPT_LEN=511
    export EVAL_MAX_RESPONSE_LEN=3
fi

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1
export SGLANG_SERVER_CONCURRENCY=16
export SGLANG_MAX_RUNNING_REQUESTS=16
export SGLANG_MAX_TOTAL_TOKENS=8192
export SGLANG_MEM_FRACTION_STATIC=0.4

export SAVE_CHECKPOINTS=0
unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG LOAD_DIR

export RUN_LOG="${RUN_LOG:-${ROOT_DIR}/log/visual-xor-refinement-gpu-${RUN_TAG}.log}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT=relax-amd-visual-xor-refinement
export WANDB_GROUP="visual-xor-refinement-gpu-async-${RUN_TAG}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/log/wandb}"

bash scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
