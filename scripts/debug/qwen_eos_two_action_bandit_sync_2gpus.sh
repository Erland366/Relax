#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the accepted joint EOS and A/B SFT model directory}"
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    echo "HF_CHECKPOINT does not contain config.json: ${HF_CHECKPOINT}" >&2
    exit 2
fi

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_ASSET_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)/relax_assets"

export ASSET_DIR="${ASSET_DIR:-${DEFAULT_ASSET_DIR}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"

export MODEL_CONFIG_NAME=qwen3-mock
export MODEL_ASSET_NAME=Qwen3-Mock-0.5B-EOS-Two-Action-Bandit-SFT
export MODEL_LOG_NAME=qwen3-mock-0.5b-eos-two-action-bandit-sft
export PROMPT_SET="${ROOT_DIR}/examples/eos_two_action_bandit/prompts.jsonl"
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

export RELAX_EXECUTION_MODE=sync
export USE_COLLOCATE=0
export RM_TYPE=eos_two_action_bandit
export REWARD_KEY=score
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export MAX_STALENESS=0

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export NUM_STEPS_PER_ROLLOUT=1
export MOCK_ROLLOUT_BATCH_SIZE=2
export MOCK_N_SAMPLES_PER_PROMPT=8
export MOCK_GLOBAL_BATCH_SIZE=16
export MICRO_BATCH_SIZE=8
export MOCK_ROLLOUT_MAX_RESPONSE_LEN=3
export UPDATE_WEIGHTS_INTERVAL=1

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1

export SAVE_CHECKPOINTS=0
unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG LOAD_DIR

export RUN_LOG="${ROOT_DIR}/log/eos-two-action-bandit-${RUN_TAG}.log"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-relax-amd-eos-two-action-bandit}"
export WANDB_GROUP="${WANDB_GROUP:-eos-two-action-bandit-${RUN_TAG}}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/log/wandb}"

bash scripts/training/multimodal/amd_qwen3_mock_2gpu_e2e.sh
