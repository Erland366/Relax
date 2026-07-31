#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

export REFINEMENT_DATA="${REFINEMENT_DATA:-/vast/users/qirong.ho/erland/Python_project/SFT_training/visual_xor_refinement_data}"
export PROMPT_SET="${PROMPT_SET:-${REFINEMENT_DATA}/refinement_rl_train.parquet}"
export EVAL_CONFIG="${EVAL_CONFIG:-${REFINEMENT_DATA}/refinement_eval_config.json}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft}"

for required_file in "${HF_CHECKPOINT}/config.json" "${HF_CHECKPOINT}/model.safetensors" "${PROMPT_SET}" "${EVAL_CONFIG}"; do
    if [ ! -f "${required_file}" ]; then
        echo "Required CPU-vision input does not exist: ${required_file}" >&2
        exit 2
    fi
done

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra VISUAL_XOR_GPU_IDS <<< "${HIP_VISIBLE_DEVICES}"
if [ "${#VISUAL_XOR_GPU_IDS[@]}" -ne 4 ]; then
    echo "HIP_VISIBLE_DEVICES must contain exactly 4 GPU IDs, got ${HIP_VISIBLE_DEVICES}" >&2
    exit 2
fi
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE="${HIP_VISIBLE_DEVICES}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES FULLY_ASYNC HYBRID

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
export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,1],"actor_fwd":[1,1],"advantages":[1,0],"vision_encoder":[1,0]}'
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

export VISION_ENCODER_BACKEND=pytorch
# Keep the correctness smoke schedulable alongside Ray/Serve control actors.
# CPU scaling runs should override this on an allocation with measured CPU capacity.
export VISION_ENCODER_NUM_CPUS="${VISION_ENCODER_NUM_CPUS:-1}"
export VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"
export VISION_ENCODER_OMIT_GPU_WEIGHTS="${VISION_ENCODER_OMIT_GPU_WEIGHTS:-0}"
export VISION_ENCODER_CACHE_MAX_BYTES="${VISION_ENCODER_CACHE_MAX_BYTES:-1073741824}"
export VISION_ENCODER_MAX_BATCH_SIZE="${VISION_ENCODER_MAX_BATCH_SIZE:-8}"
if [ "${VISION_ENCODER_OMIT_GPU_WEIGHTS}" = "1" ]; then
    CPU_VISION_MODE=cpu-omitted
else
    CPU_VISION_MODE=cpu-resident
fi

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_ASSET_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)/relax_assets"
export ASSET_DIR="${ASSET_DIR:-${DEFAULT_ASSET_DIR}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"
export MODEL_CONFIG_NAME=qwen3-vl-0.37B
export MODEL_ASSET_NAME=Qwen3-VL-0.37B-Visual-XOR-Refinement-SFT
export MODEL_LOG_NAME="qwen3-vl-0.37b-visual-xor-refinement-${CPU_VISION_MODE}"
export SYSTEM_PROMPT="Answer with exactly one uppercase character: A or B. Do not explain."
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
export MULTIMODAL_KEYS='{"image":"image"}'

export RM_TYPE=visual_xor
export REWARD_KEY=score
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export LR="${LR:-3e-6}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-250}"
export NUM_STEPS_PER_ROLLOUT=2
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=32
export MICRO_BATCH_SIZE=1
export SEQ_LENGTH=512
export ROLLOUT_MAX_CONTEXT_LEN=512
export ROLLOUT_MAX_PROMPT_LEN=511
export ROLLOUT_MAX_RESPONSE_LEN=3
export UPDATE_WEIGHTS_INTERVAL=1
export ENABLE_RECOMPUTE=1
export USE_STREAMING_DATASET=0
export ROLLOUT_SHUFFLE=0

export EVAL_INTERVAL="${EVAL_INTERVAL:-4}"
export EVAL_MAX_CONTEXT_LEN=512
export EVAL_MAX_PROMPT_LEN=511
export EVAL_MAX_RESPONSE_LEN=3

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1
export SGLANG_SERVER_CONCURRENCY=16
export SGLANG_MAX_RUNNING_REQUESTS=16
export SGLANG_MAX_TOTAL_TOKENS=8192
export SGLANG_MEM_FRACTION_STATIC=0.4

export SAVE_CHECKPOINTS=0
unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG LOAD_DIR

export RUN_LOG="${ROOT_DIR}/log/visual-xor-refinement-${CPU_VISION_MODE}-${RUN_TAG}.log"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT=relax-amd-visual-xor-refinement
export WANDB_GROUP="visual-xor-refinement-${CPU_VISION_MODE}-async-${RUN_TAG}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/log/wandb}"

bash scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
