#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
GEO3K_DATA="${GEO3K_DATA:-}"
PROMPT_SET="${PROMPT_SET:-${GEO3K_DATA:+${GEO3K_DATA}/train.parquet}}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
EVAL_CONFIG="${EVAL_CONFIG:-${GEO3K_DATA:+${GEO3K_DATA}/relax_eval_config.json}}"
VISION_RUN_MODE="${VISION_RUN_MODE:-gpu}"

if [ -z "${HF_CHECKPOINT}" ] || [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    echo "HF_CHECKPOINT must contain Qwen3-VL-4B config.json: ${HF_CHECKPOINT:-unset}" >&2
    exit 2
fi
if ! compgen -G "${HF_CHECKPOINT}/*.safetensors" >/dev/null; then
    echo "HF_CHECKPOINT must contain Qwen3-VL-4B safetensors: ${HF_CHECKPOINT}" >&2
    exit 2
fi
if [ -z "${GEO3K_DATA}" ] || [ ! -f "${PROMPT_SET}" ]; then
    echo "Geo3K training parquet does not exist: ${PROMPT_SET:-unset}" >&2
    exit 2
fi
if [ "${ENABLE_EVAL}" = "1" ]; then
    if [ ! -f "${EVAL_CONFIG}" ]; then
        echo "Geo3K evaluation config does not exist: ${EVAL_CONFIG:-unset}" >&2
        echo "Create it with: ${PYTHON} -m examples.geo3k.build_eval_config --test-data ${GEO3K_DATA}/test.parquet --output ${GEO3K_DATA}/relax_eval_config.json" >&2
        exit 2
    fi
elif [ "${ENABLE_EVAL}" = "0" ]; then
    unset EVAL_CONFIG EVAL_INTERVAL EVAL_MAX_CONTEXT_LEN EVAL_MAX_PROMPT_LEN EVAL_MAX_RESPONSE_LEN
else
    echo "ENABLE_EVAL must be 0 or 1, got ${ENABLE_EVAL}" >&2
    exit 2
fi

if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    echo "HIP_VISIBLE_DEVICES must be inherited from the eight-GPU allocation or set explicitly" >&2
    exit 2
fi
IFS=',' read -ra GEO3K_GPU_IDS <<< "${HIP_VISIBLE_DEVICES}"
if [ "${#GEO3K_GPU_IDS[@]}" -ne 8 ]; then
    echo "The Geo3K paper launcher requires exactly 8 visible GPUs, got ${HIP_VISIBLE_DEVICES}" >&2
    exit 2
fi
if [ "$(printf '%s\n' "${GEO3K_GPU_IDS[@]}" | sort -u | wc -l)" -ne 8 ]; then
    echo "HIP_VISIBLE_DEVICES must contain eight unique device identifiers: ${HIP_VISIBLE_DEVICES}" >&2
    exit 2
fi
export RELAX_HIP_VISIBLE_DEVICES_OVERRIDE="${HIP_VISIBLE_DEVICES}"
unset FULLY_ASYNC HYBRID
"${PYTHON}" - <<'PY'
import torch


count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"Geo3K paper launcher requires exactly eight PyTorch-visible GPUs; found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if not all("MI210" in name for name in names):
    raise SystemExit(f"Geo3K paper launcher requires eight MI210 GPUs; found {names}")
print(f"GPU preflight passed: {names}")
PY

case "${VISION_RUN_MODE}" in
    gpu)
        export VISION_ENCODER_DEVICE=gpu
        export SKIP_GPU_VISION_ENCODER=0
        export VISION_DEVICE_MODE=fixed
        export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,4],"actor_fwd":[1,2],"advantages":[1,0]}'
        ;;
    cpu)
        export VISION_ENCODER_DEVICE=cpu
        export SKIP_GPU_VISION_ENCODER=0
        export VISION_DEVICE_MODE=fixed
        export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,4],"actor_fwd":[1,2],"advantages":[1,0],"vision_encoder":[1,0]}'
        ;;
    cpu_skip_gpu_encoder)
        export VISION_ENCODER_DEVICE=cpu
        export SKIP_GPU_VISION_ENCODER=1
        export VISION_DEVICE_MODE=fixed
        export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,4],"actor_fwd":[1,2],"advantages":[1,0],"vision_encoder":[1,0]}'
        ;;
    automatic)
        echo "Automatic Geo3K device choice requires runtime workload measurements, which are not implemented" >&2
        echo "Use gpu, cpu, or cpu_skip_gpu_encoder for the current real-task study" >&2
        exit 2
        ;;
    *)
        echo "VISION_RUN_MODE must be gpu, cpu, cpu_skip_gpu_encoder, or automatic; got ${VISION_RUN_MODE}" >&2
        exit 2
        ;;
esac

export RELAX_EXECUTION_MODE=fully_async
export USE_COLLOCATE=0
export MAX_STALENESS="${MAX_STALENESS:-4}"
export RAY_NUM_GPUS=8
export NUM_GPUS_PER_NODE=8
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
export ACTOR_RESOURCE_GPUS=2
export ACTOR_FWD_RESOURCE_GPUS=2
export ROLLOUT_RESOURCE_GPUS=4
export TRUE_ON_POLICY_MODE=FALSE
export TENSOR_MODEL_PARALLEL_SIZE=2
export PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1
export ENABLE_SEQUENCE_PARALLEL=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2
export SGLANG_PIPELINE_PARALLEL_SIZE=1
export SGLANG_DATA_PARALLEL_SIZE=1
export SGLANG_EXPERT_PARALLEL_SIZE=1
export SGLANG_ENABLE_DP_ATTENTION=0

export VISION_ENCODER_NUM_CPUS="${VISION_ENCODER_NUM_CPUS:-4}"
export VISION_ENCODER_NUM_REPLICAS="${VISION_ENCODER_NUM_REPLICAS:-1}"
export VISION_ENCODER_CACHE_MAX_BYTES="${VISION_ENCODER_CACHE_MAX_BYTES:-1}"
export SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${SGLANG_VISION_FEATURE_CACHE_MAX_BYTES:-0}"
export PRELOAD_VISION_FEATURES=0
export VISION_ENCODER_MAX_IMAGES_PER_REQUEST="${VISION_ENCODER_MAX_IMAGES_PER_REQUEST:-8}"
export VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0
export FREEZE_VISION_MODEL=1

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_ASSET_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)/relax_assets"
export ASSET_DIR="${ASSET_DIR:-${DEFAULT_ASSET_DIR}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"
export MODEL_CONFIG_NAME=qwen3-vl-4B
export MODEL_ASSET_NAME=Qwen3-VL-4B-Instruct
export MODEL_LOG_NAME="qwen3-vl-4b-geo3k-${VISION_RUN_MODE}"
export INPUT_KEY=prompt
export LABEL_KEY=reward_model
export MULTIMODAL_KEYS='{"image":"images"}'
if [ -z "${SYSTEM_PROMPT:-}" ]; then
    export SYSTEM_PROMPT='Solve the geometry problem step by step. Put the reasoning inside <think></think> and the final answer inside \boxed{}.'
else
    export SYSTEM_PROMPT
fi
export RM_TYPE=geo3k
export REWARD_KEY="${REWARD_KEY:-}"
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export LR="${LR:-1e-6}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-4096}"
export ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
export UPDATE_WEIGHTS_INTERVAL=1
export ENABLE_RECOMPUTE=1
export USE_STREAMING_DATASET=0
export ROLLOUT_SHUFFLE=0

if [ "${ENABLE_EVAL}" = "1" ]; then
    export EVAL_CONFIG
    export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
    export EVAL_MAX_CONTEXT_LEN=4096
    export EVAL_MAX_PROMPT_LEN=2048
    export EVAL_MAX_RESPONSE_LEN=2048
fi

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-32768}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.65}"

export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
if [ "${SAVE_CHECKPOINTS}" = "0" ]; then
    unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG LOAD_DIR
fi

export RUN_LOG="${RUN_LOG:-${ROOT_DIR}/log/qwen3-vl-4b-geo3k-${VISION_RUN_MODE}-${RUN_TAG}.log}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-relax-qwen3-vl-geo3k}"
export WANDB_GROUP="${WANDB_GROUP:-qwen3-vl-4b-geo3k-${VISION_RUN_MODE}-${RUN_TAG}}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/log/wandb}"

bash scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
