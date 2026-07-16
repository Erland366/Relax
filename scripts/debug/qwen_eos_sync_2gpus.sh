cd /vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610

RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"

export ASSET_DIR=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets
export CONDA_ENV_NAME=relaxrl_rocm_after_fix

export MODEL_CONFIG_NAME=qwen3-mock
export MODEL_ASSET_NAME=Qwen3-Mock-0.5B-EOS-SFT
export MODEL_LOG_NAME=qwen3-mock-0.5b-eos-sft
export HF_CHECKPOINT="${HF_CHECKPOINT:-${ASSET_DIR}/Qwen3-Mock-0.5B-EOS-SFT}"

export PROMPT_SET="${PWD}/examples/completion_length/prompts_system_eos.jsonl"
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

export RELAX_EXECUTION_MODE=sync
export USE_COLLOCATE=0
export RM_TYPE=completion_length
export REWARD_KEY=
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export MAX_STALENESS=0

export NUM_ROLLOUT=200
export NUM_STEPS_PER_ROLLOUT=1
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES_PER_PROMPT=4
export GLOBAL_BATCH_SIZE=16
export MICRO_BATCH_SIZE=4
export ROLLOUT_MAX_RESPONSE_LEN=4
export UPDATE_WEIGHTS_INTERVAL=5

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1

export SAVE_CHECKPOINTS=0
unset SAVE_DIR SAVE_INTERVAL CKPT_FORMAT NO_SAVE_OPTIM NO_SAVE_RNG
export RUN_LOG="${PWD}/log/completion-length-eos-sft-system-30-${RUN_TAG}.log"

export WANDB_MODE=online
export WANDB_ENTITY=
export WANDB_PROJECT=relax-amd-completion-length
export WANDB_GROUP="${WANDB_GROUP:-completion-length-eos-sft-system-30-${RUN_TAG}}"
export WANDB_DIR="${ASSET_DIR}/wandb"

bash scripts/training/multimodal/amd_qwen3_mock_2gpu_e2e.sh
