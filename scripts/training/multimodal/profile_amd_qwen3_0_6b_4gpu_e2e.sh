#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

cd "${ROOT_DIR}"

PROFILE_TIMESTAMP="${PROFILE_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
PROFILE_PASS="${PROFILE_PASS:-timeline}"
PROFILE_ROOT="${PROFILE_ROOT:-${ROOT_DIR}/profiling_results/qwen3-0.6b-fully-async-${PROFILE_TIMESTAMP}}"
PROFILE_ROOT="$(mkdir -p "${PROFILE_ROOT}" && cd "${PROFILE_ROOT}" && pwd)"
PROFILE_NAME="${RELAX_TB_EXPERIMENT_NAME:-qwen3-0.6b-fully-async-${PROFILE_PASS}-${PROFILE_TIMESTAMP}}"

case "${PROFILE_PASS}" in
    smoke | timeline | torch | sglang | all)
        ;;
    *)
        echo "PROFILE_PASS must be smoke, timeline, torch, sglang, or all; got ${PROFILE_PASS}" >&2
        exit 2
        ;;
esac

export WANDB_MODE="${WANDB_MODE:-offline}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm_after_fix}"
export RELAX_EXECUTION_MODE="${RELAX_EXECUTION_MODE:-fully_async}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_NUM_GPUS="${RAY_NUM_GPUS:-4}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
export ACTOR_RESOURCE_GPUS="${ACTOR_RESOURCE_GPUS:-2}"
export ROLLOUT_RESOURCE_GPUS="${ROLLOUT_RESOURCE_GPUS:-1}"
export ACTOR_FWD_RESOURCE_GPUS="${ACTOR_FWD_RESOURCE_GPUS:-1}"
export MAX_STALENESS="${MAX_STALENESS:-1}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-1024}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-128}"

export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-65536}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.2}"

export RELAX_USE_WANDB="${RELAX_USE_WANDB:-1}"
export RELAX_USE_METRICS_SERVICE="${RELAX_USE_METRICS_SERVICE:-1}"
export RELAX_DISABLE_SAVE="${RELAX_DISABLE_SAVE:-0}"
export RELAX_SKIP_INITIAL_FULLY_ASYNC_WEIGHT_UPDATE="${RELAX_SKIP_INITIAL_FULLY_ASYNC_WEIGHT_UPDATE:-0}"
export RELAX_TB_EXPERIMENT_NAME="${RELAX_TB_EXPERIMENT_NAME:-${PROFILE_NAME}}"

case "${PROFILE_PASS}" in
    timeline | torch | sglang | all)
        export RELAX_TIMELINE_DUMP_DIR="${RELAX_TIMELINE_DUMP_DIR:-${PROFILE_ROOT}/timeline}"
        ;;
esac

case "${PROFILE_PASS}" in
    torch | all)
        export RELAX_USE_PYTORCH_PROFILER="${RELAX_USE_PYTORCH_PROFILER:-1}"
        export RELAX_PROFILE_TARGETS="${RELAX_PROFILE_TARGETS:-train_overall train_actor train_log_probs}"
        export RELAX_PROFILE_STEP_START="${RELAX_PROFILE_STEP_START:-1}"
        export RELAX_PROFILE_STEP_END="${RELAX_PROFILE_STEP_END:-2}"
        ;;
esac

case "${PROFILE_PASS}" in
    sglang | all)
        export RELAX_SGLANG_PROFILE="${RELAX_SGLANG_PROFILE:-1}"
        export RELAX_SGLANG_PROFILE_STEP_START="${RELAX_SGLANG_PROFILE_STEP_START:-1}"
        export RELAX_SGLANG_PROFILE_STEP_END="${RELAX_SGLANG_PROFILE_STEP_END:-2}"
        export RELAX_SGLANG_PROFILE_NUM_STEPS="${RELAX_SGLANG_PROFILE_NUM_STEPS:-3}"
        export RELAX_SGLANG_PROFILE_OUTPUT_DIR="${RELAX_SGLANG_PROFILE_OUTPUT_DIR:-${PROFILE_ROOT}/sglang_trace}"
        ;;
esac

mkdir -p "${PROFILE_ROOT}" "${PROFILE_ROOT}/logs"

write_launch_env() {
    {
        printf 'PROFILE_ROOT=%q\n' "${PROFILE_ROOT}"
        printf 'PROFILE_PASS=%q\n' "${PROFILE_PASS}"
        printf 'RELAX_TB_EXPERIMENT_NAME=%q\n' "${RELAX_TB_EXPERIMENT_NAME}"
        printf 'WANDB_MODE=%q\n' "${WANDB_MODE}"
        printf 'CONDA_ENV_NAME=%q\n' "${CONDA_ENV_NAME}"
        env | grep -E '^(RELAX_|HIP_VISIBLE_DEVICES|RAY_NUM_GPUS|NUM_GPUS_PER_NODE|ACTOR_|ROLLOUT_|MAX_STALENESS|NUM_ROLLOUT|NUM_STEPS_PER_ROLLOUT|GLOBAL_BATCH_SIZE|MICRO_BATCH_SIZE|SEQ_LENGTH|SGLANG_|WANDB_)=' | sort || true
    } > "${PROFILE_ROOT}/launch.env"
}

sync_train_traces() {
    local source_dir="${ROOT_DIR}/traces/${RELAX_TB_EXPERIMENT_NAME}"
    local target_dir="${PROFILE_ROOT}/traces/${RELAX_TB_EXPERIMENT_NAME}"

    if [ ! -d "${source_dir}" ]; then
        return 0
    fi

    mkdir -p "${PROFILE_ROOT}/traces"
    rm -rf "${target_dir}"
    cp -a "${source_dir}" "${target_dir}"
}

run_summary() {
    local status="$1"
    python scripts/tools/summarize_fully_async_profile.py \
        --profile-root "${PROFILE_ROOT}" \
        --expected-pass "${PROFILE_PASS}" \
        --run-status "${status}"
}

write_launch_env

printf 'Profiling Qwen3-0.6B fully_async pass=%s\n' "${PROFILE_PASS}" | tee "${PROFILE_ROOT}/run.log"
printf '  profile_root=%s\n' "${PROFILE_ROOT}" | tee -a "${PROFILE_ROOT}/run.log"
printf '  tb_experiment=%s\n' "${RELAX_TB_EXPERIMENT_NAME}" | tee -a "${PROFILE_ROOT}/run.log"

set +e
"${SCRIPT_DIR}/run_amd_qwen3_0_6b_4gpu_e2e.sh" "$@" 2>&1 | tee -a "${PROFILE_ROOT}/run.log"
run_status="${PIPESTATUS[0]}"
set -e

sync_train_traces

if grep -Eq "Job 'raysubmit_[^']+' was stopped" "${PROFILE_ROOT}/run.log"; then
    run_summary stopped || true
    exit 130
elif [ "${run_status}" -eq 0 ]; then
    run_summary success
else
    run_summary "failed:${run_status}" || true
fi

exit "${run_status}"
