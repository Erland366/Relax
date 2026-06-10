#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="${RELAX_ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}"
ASSET_DIR="/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets"
MEGATRON_DIR="${MEGATRON_DIR:-/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM}"
MEGATRON_BRIDGE_DIR="${MEGATRON_BRIDGE_DIR:-/vast/users/qirong.ho/erland/Python_project/Megatron-Bridge}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/log}"
mkdir -p "${LOG_DIR}"

localhost_bypass() {
    env \
        -u http_proxy \
        -u https_proxy \
        -u HTTP_PROXY \
        -u HTTPS_PROXY \
        -u ALL_PROXY \
        -u all_proxy \
        "$@"
}

append_no_proxy() {
    local host_list="127.0.0.1,localhost,::1"

    if [ -n "${no_proxy:-}" ]; then
        export no_proxy="${host_list},${no_proxy}"
    else
        export no_proxy="${host_list}"
    fi

    if [ -n "${NO_PROXY:-}" ]; then
        export NO_PROXY="${host_list},${NO_PROXY}"
    else
        export NO_PROXY="${host_list}"
    fi
}

require_positive_integer() {
    local name="$1"
    local value="${!name}"

    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "${name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
}

require_boolean_flag() {
    local name="$1"
    local value="${!name}"

    case "${value}" in
        0 | 1)
            ;;
        *)
            echo "${name} must be 0 or 1, got ${value}" >&2
            exit 2
            ;;
    esac
}

require_execution_mode() {
    local value="${RELAX_EXECUTION_MODE}"

    case "${value}" in
        sync | hybrid | fully_async)
            ;;
        *)
            echo "RELAX_EXECUTION_MODE must be sync, hybrid, or fully_async; got ${value}" >&2
            exit 2
            ;;
    esac
}

require_nonnegative_integer() {
    local name="$1"
    local value="${!name}"

    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a non-negative integer, got ${value}" >&2
        exit 2
    fi
}

require_optional_positive_integer() {
    local name="$1"
    local value="${!name:-}"

    if [ -z "${value}" ]; then
        return 0
    fi

    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "${name} must be empty or a positive integer, got ${value}" >&2
        exit 2
    fi
}

activate_environment() {
    source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm}"
    conda activate "${CONDA_ENV_NAME}"
}

load_dotenv() {
    if [ -f "${ROOT_DIR}/.env" ]; then
        set -a
        source "${ROOT_DIR}/.env"
        set +a
    fi
}

configure_gpu_resources() {
    if [ -n "${RELAX_HIP_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
        export HIP_VISIBLE_DEVICES="${RELAX_HIP_VISIBLE_DEVICES_OVERRIDE}"
    fi

    export PYTHONUNBUFFERED=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"

    IFS=',' read -ra RELAX_VISIBLE_GPU_IDS <<< "${HIP_VISIBLE_DEVICES}"
    DEFAULT_VISIBLE_GPU_COUNT="${#RELAX_VISIBLE_GPU_IDS[@]}"

    RAY_NUM_GPUS="${RAY_NUM_GPUS:-${DEFAULT_VISIBLE_GPU_COUNT}}"
    NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${RAY_NUM_GPUS}}"
    ACTOR_RESOURCE_GPUS="${ACTOR_RESOURCE_GPUS:-1}"
    ACTOR_FWD_RESOURCE_GPUS="${ACTOR_FWD_RESOURCE_GPUS:-0}"
    require_positive_integer RAY_NUM_GPUS
    require_positive_integer NUM_GPUS_PER_NODE
    require_positive_integer ACTOR_RESOURCE_GPUS
    require_nonnegative_integer ACTOR_FWD_RESOURCE_GPUS

    ROLLOUT_RESOURCE_GPUS="${ROLLOUT_RESOURCE_GPUS:-$((RAY_NUM_GPUS - ACTOR_RESOURCE_GPUS))}"
    require_positive_integer ROLLOUT_RESOURCE_GPUS

    if [ "$((ACTOR_RESOURCE_GPUS + ROLLOUT_RESOURCE_GPUS + ACTOR_FWD_RESOURCE_GPUS))" -gt "${RAY_NUM_GPUS}" ]; then
        echo "ACTOR_RESOURCE_GPUS + ROLLOUT_RESOURCE_GPUS + ACTOR_FWD_RESOURCE_GPUS exceeds RAY_NUM_GPUS" >&2
        exit 2
    fi
}

configure_execution_mode() {
    RELAX_EXECUTION_MODE="${RELAX_EXECUTION_MODE:-${TRAINING_PIPELINE_MODE:-sync}}"
    if [ "${FULLY_ASYNC:-0}" = "1" ]; then
        RELAX_EXECUTION_MODE="fully_async"
    fi
    if [ "${HYBRID:-0}" = "1" ]; then
        RELAX_EXECUTION_MODE="hybrid"
    fi
    require_execution_mode
}

configure_sglang_parallelism() {
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
    SGLANG_PIPELINE_PARALLEL_SIZE="${SGLANG_PIPELINE_PARALLEL_SIZE:-1}"
    SGLANG_DATA_PARALLEL_SIZE="${SGLANG_DATA_PARALLEL_SIZE:-1}"
    SGLANG_EXPERT_PARALLEL_SIZE="${SGLANG_EXPERT_PARALLEL_SIZE:-1}"
    SGLANG_ENABLE_DP_ATTENTION="${SGLANG_ENABLE_DP_ATTENTION:-0}"
    SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-torch_native}"

    require_positive_integer ROLLOUT_NUM_GPUS_PER_ENGINE
    require_positive_integer SGLANG_PIPELINE_PARALLEL_SIZE
    require_positive_integer SGLANG_DATA_PARALLEL_SIZE
    require_positive_integer SGLANG_EXPERT_PARALLEL_SIZE
    require_boolean_flag SGLANG_ENABLE_DP_ATTENTION

    if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -gt "${ROLLOUT_RESOURCE_GPUS}" ]; then
        echo "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE} exceeds ROLLOUT_RESOURCE_GPUS=${ROLLOUT_RESOURCE_GPUS}" >&2
        exit 2
    fi
    if [ "$((ROLLOUT_RESOURCE_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE))" -ne 0 ]; then
        echo "ROLLOUT_RESOURCE_GPUS=${ROLLOUT_RESOURCE_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
        exit 2
    fi
    if [ "$((ROLLOUT_NUM_GPUS_PER_ENGINE % SGLANG_PIPELINE_PARALLEL_SIZE))" -ne 0 ]; then
        echo "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE} must be divisible by SGLANG_PIPELINE_PARALLEL_SIZE=${SGLANG_PIPELINE_PARALLEL_SIZE}" >&2
        exit 2
    fi
    if [ "${SGLANG_DATA_PARALLEL_SIZE}" -gt 1 ] && [ "${SGLANG_ENABLE_DP_ATTENTION}" != "1" ]; then
        echo "SGLANG_ENABLE_DP_ATTENTION=1 is required when SGLANG_DATA_PARALLEL_SIZE > 1" >&2
        exit 2
    fi
}

configure_megatron_parallelism() {
    TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${ACTOR_RESOURCE_GPUS}}"
    PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
    CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
    EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
    EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
    ENABLE_RECOMPUTE="${ENABLE_RECOMPUTE:-1}"

    require_positive_integer TENSOR_MODEL_PARALLEL_SIZE
    require_positive_integer PIPELINE_MODEL_PARALLEL_SIZE
    require_positive_integer CONTEXT_PARALLEL_SIZE
    require_positive_integer EXPERT_MODEL_PARALLEL_SIZE
    require_positive_integer EXPERT_TENSOR_PARALLEL_SIZE
    require_boolean_flag ENABLE_RECOMPUTE

    MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))

    if [ -z "${ENABLE_SEQUENCE_PARALLEL+x}" ]; then
        if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
            ENABLE_SEQUENCE_PARALLEL=1
        else
            ENABLE_SEQUENCE_PARALLEL=0
        fi
    fi
    require_boolean_flag ENABLE_SEQUENCE_PARALLEL

    if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ] && [ "${ENABLE_SEQUENCE_PARALLEL}" != "1" ]; then
        echo "ENABLE_SEQUENCE_PARALLEL=1 is required when TENSOR_MODEL_PARALLEL_SIZE > 1" >&2
        exit 2
    fi
    if [ "${ACTOR_RESOURCE_GPUS}" -lt "${MODEL_PARALLEL_SIZE}" ]; then
        echo "ACTOR_RESOURCE_GPUS=${ACTOR_RESOURCE_GPUS} is smaller than model parallel size ${MODEL_PARALLEL_SIZE}" >&2
        exit 2
    fi
    if [ "$((ACTOR_RESOURCE_GPUS % MODEL_PARALLEL_SIZE))" -ne 0 ]; then
        echo "ACTOR_RESOURCE_GPUS=${ACTOR_RESOURCE_GPUS} must be divisible by model parallel size ${MODEL_PARALLEL_SIZE}" >&2
        exit 2
    fi
}

configure_runtime_environment() {
    GPU_LABEL="${GPU_LABEL:-${RAY_NUM_GPUS}gpu}"
    case "${RELAX_EXECUTION_MODE}" in
        fully_async)
            if [ "${ACTOR_FWD_RESOURCE_GPUS}" -gt 0 ]; then
                DEFAULT_RESOURCE_JSON="{\"actor\": [1, ${ACTOR_RESOURCE_GPUS}], \"rollout\": [1, ${ROLLOUT_RESOURCE_GPUS}], \"actor_fwd\": [1, ${ACTOR_FWD_RESOURCE_GPUS}], \"advantages\": [1, 0]}"
            else
                DEFAULT_RESOURCE_JSON="{\"actor\": [1, ${ACTOR_RESOURCE_GPUS}], \"rollout\": [1, ${ROLLOUT_RESOURCE_GPUS}], \"advantages\": [1, 0]}"
            fi
            ;;
        sync | hybrid)
            DEFAULT_RESOURCE_JSON="{\"actor\": [1, ${ACTOR_RESOURCE_GPUS}], \"rollout\": [1, ${ROLLOUT_RESOURCE_GPUS}]}"
            ;;
    esac
    RESOURCE_JSON="${RESOURCE_JSON:-${DEFAULT_RESOURCE_JSON}}"

    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
    export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
    export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-300}"
    export RAY_py_gcs_connect_timeout_s="${RAY_py_gcs_connect_timeout_s:-300}"
    export RAY_nums_py_gcs_reconnect_retry="${RAY_nums_py_gcs_reconnect_retry:-80}"
    export RAY_task_events_report_interval_ms="${RAY_task_events_report_interval_ms:-0}"
    export RAY_grpc_client_keepalive_time_ms="${RAY_grpc_client_keepalive_time_ms:-600000}"
    export RAY_grpc_client_keepalive_timeout_ms="${RAY_grpc_client_keepalive_timeout_ms:-300000}"
    unset ROCR_VISIBLE_DEVICES

    if [ ! -d "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" ]; then
        echo "MEGATRON_BRIDGE_DIR does not contain src/megatron/bridge: ${MEGATRON_BRIDGE_DIR}" >&2
        exit 2
    fi

    SGLANG_PYTHON_DIR="${SGLANG_PYTHON_DIR:-/vast/users/qirong.ho/erland/Python_project/sglang/python}"
    if [ -n "${RELAX_SGL_KERNEL_BUILD_DIR:-}" ] && [ ! -d "${RELAX_SGL_KERNEL_BUILD_DIR}" ]; then
        echo "RELAX_SGL_KERNEL_BUILD_DIR does not exist: ${RELAX_SGL_KERNEL_BUILD_DIR}" >&2
        exit 2
    fi
    if [ ! -d "${SGLANG_PYTHON_DIR}" ]; then
        echo "SGLANG_PYTHON_DIR does not exist: ${SGLANG_PYTHON_DIR}" >&2
        exit 2
    fi

    if [ -n "${RELAX_SGL_KERNEL_BUILD_DIR:-}" ]; then
        export PYTHONPATH="${RELAX_SGL_KERNEL_BUILD_DIR}:${SGLANG_PYTHON_DIR}:${MEGATRON_BRIDGE_DIR}/src:${MEGATRON_DIR}:${ROOT_DIR}"
    else
        export PYTHONPATH="${SGLANG_PYTHON_DIR}:${MEGATRON_BRIDGE_DIR}/src:${MEGATRON_DIR}:${ROOT_DIR}"
    fi
    export MEGATRON="${MEGATRON_DIR}"
    export MEGATRON_BRIDGE_DIR
    export RELAX="${ROOT_DIR}"
    export MODEL_CONFIG_DIR="${ROOT_DIR}/scripts/models"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
}

configure_run_defaults() {
    NOW="$(date '+%Y%m%d_%H%M%S')"
    MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME:-qwen3-4B}"
    MODEL_ASSET_NAME="${MODEL_ASSET_NAME:-Qwen3-4B}"
    MODEL_LOG_NAME="${MODEL_LOG_NAME:-qwen3-4b}"
    PROJECT_NAME="${PROJECT_NAME:-Relax/amd/dapo-math}"
    WANDB_PROJECT="${WANDB_PROJECT:-relax-amd}"
    WANDB_GROUP="${WANDB_GROUP:-${MODEL_LOG_NAME}-mi210-${GPU_LABEL}-${NOW}}"
    WANDB_DIR="${WANDB_DIR:-${ASSET_DIR}/wandb}"
    SAVE_DIR="${SAVE_DIR:-${ASSET_DIR}/${MODEL_ASSET_NAME}_mcore_${GPU_LABEL}-${NOW}}"
    LOAD_DIR="${LOAD_DIR:-}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
    CKPT_FORMAT="${CKPT_FORMAT:-torch_dist}"
    NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
    NO_SAVE_RNG="${NO_SAVE_RNG:-0}"
    NO_LOAD_RNG="${NO_LOAD_RNG:-0}"
    NUM_DATA_STORAGE_UNITS="${NUM_DATA_STORAGE_UNITS:-1}"
    require_positive_integer NUM_DATA_STORAGE_UNITS
    SCHEDULER_RESUME_POLICY="${SCHEDULER_RESUME_POLICY:-strict}"
    case "${RELAX_EXECUTION_MODE}" in
        fully_async)
            if [ "${ACTOR_FWD_RESOURCE_GPUS}" -gt 0 ]; then
                MAX_STALENESS="${MAX_STALENESS:-1}"
                NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
                ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
            else
                MAX_STALENESS="${MAX_STALENESS:-0}"
                NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
                ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
            fi
            USE_BALANCE_DATA="${USE_BALANCE_DATA:-0}"
            USE_KL_LOSS="${USE_KL_LOSS:-0}"
            ;;
        hybrid)
            MAX_STALENESS="${MAX_STALENESS:-2}"
            NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
            ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
            USE_BALANCE_DATA="${USE_BALANCE_DATA:-1}"
            USE_KL_LOSS="${USE_KL_LOSS:-1}"
            ;;
        sync)
            MAX_STALENESS="${MAX_STALENESS:-0}"
            NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
            ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
            USE_BALANCE_DATA="${USE_BALANCE_DATA:-1}"
            USE_KL_LOSS="${USE_KL_LOSS:-1}"
            ;;
    esac
    NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-512}"
    SEQ_LENGTH="${SEQ_LENGTH:-4096}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-768}"
    require_nonnegative_integer MAX_STALENESS
    require_boolean_flag USE_BALANCE_DATA
    require_boolean_flag USE_KL_LOSS
    require_positive_integer NUM_ROLLOUT
    require_positive_integer NUM_STEPS_PER_ROLLOUT
    require_positive_integer ROLLOUT_BATCH_SIZE
    require_positive_integer N_SAMPLES_PER_PROMPT
    DEFAULT_GLOBAL_BATCH_SIZE="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}"
    require_positive_integer GLOBAL_BATCH_SIZE
    require_positive_integer MICRO_BATCH_SIZE
    require_positive_integer SGLANG_SERVER_CONCURRENCY
    require_positive_integer SEQ_LENGTH
    require_positive_integer ROLLOUT_MAX_RESPONSE_LEN
    require_optional_positive_integer ROLLOUT_MAX_CONTEXT_LEN
    require_optional_positive_integer ROLLOUT_MAX_PROMPT_LEN
    require_optional_positive_integer SGLANG_MAX_RUNNING_REQUESTS
    require_optional_positive_integer SGLANG_MAX_TOTAL_TOKENS

    if [ -n "${MAX_TOKENS_PER_GPU:-}" ] || [ -n "${LOG_PROBS_MAX_TOKENS_PER_GPU:-}" ]; then
        echo "MAX_TOKENS_PER_GPU and LOG_PROBS_MAX_TOKENS_PER_GPU require dynamic batch size, but this ROCm launcher uses qkv-format=bshd where Relax rejects dynamic batch size. Use MICRO_BATCH_SIZE instead." >&2
        exit 2
    fi

    ROLLOUT_SAMPLES_PER_STEP=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
    if [ "$((ROLLOUT_SAMPLES_PER_STEP % NUM_STEPS_PER_ROLLOUT))" -ne 0 ]; then
        echo "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be divisible by NUM_STEPS_PER_ROLLOUT." >&2
        echo "Current values: ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}, NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT}." >&2
        exit 2
    fi
    EXPECTED_GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))
    if [ "${GLOBAL_BATCH_SIZE}" -ne "${EXPECTED_GLOBAL_BATCH_SIZE}" ]; then
        echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT // NUM_STEPS_PER_ROLLOUT = ${EXPECTED_GLOBAL_BATCH_SIZE}" >&2
        exit 2
    fi
    TRAIN_DP_SIZE=$((ACTOR_RESOURCE_GPUS / MODEL_PARALLEL_SIZE))
    if [ "${USE_BALANCE_DATA}" = "1" ]; then
        if [ "$((GLOBAL_BATCH_SIZE % N_SAMPLES_PER_PROMPT))" -ne 0 ]; then
            echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT} when USE_BALANCE_DATA=1." >&2
            exit 2
        fi
        PROMPT_GROUPS_PER_TRAIN_STEP=$((GLOBAL_BATCH_SIZE / N_SAMPLES_PER_PROMPT))
        if [ "${PROMPT_GROUPS_PER_TRAIN_STEP}" -lt "${TRAIN_DP_SIZE}" ]; then
            echo "USE_BALANCE_DATA=1 with train DP=${TRAIN_DP_SIZE} requires at least ${TRAIN_DP_SIZE} prompt groups per train step." >&2
            echo "Current values give ${PROMPT_GROUPS_PER_TRAIN_STEP}: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}." >&2
            echo "Reduce NUM_STEPS_PER_ROLLOUT or increase GLOBAL_BATCH_SIZE/ROLLOUT_BATCH_SIZE." >&2
            exit 2
        fi
    fi
    if [ "${RELAX_EXECUTION_MODE}" = "fully_async" ] && [ "${ACTOR_FWD_RESOURCE_GPUS}" -eq 0 ] && [ "${RESOURCE_JSON}" = "${DEFAULT_RESOURCE_JSON}" ]; then
        if [ "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" -ne "${GLOBAL_BATCH_SIZE}" ]; then
            echo "Pure fully_async without actor_fwd is true-on-policy, so ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must equal GLOBAL_BATCH_SIZE." >&2
            echo "Current values: ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}." >&2
            echo "Set ACTOR_FWD_RESOURCE_GPUS>0 or provide RESOURCE_JSON with actor_fwd GPUs for bounded-staleness fully_async." >&2
            exit 2
        fi
        if [ "${MAX_STALENESS}" -ne 0 ]; then
            echo "Pure fully_async without actor_fwd requires MAX_STALENESS=0; got ${MAX_STALENESS}." >&2
            echo "Set ACTOR_FWD_RESOURCE_GPUS>0 for bounded-staleness fully_async." >&2
            exit 2
        fi
        if [ "${USE_KL_LOSS}" = "1" ]; then
            echo "Pure fully_async default resource graph has no reference service, so USE_KL_LOSS must be 0." >&2
            echo "Provide RESOURCE_JSON with reference GPUs to enable KL loss in pure fully_async mode." >&2
            exit 2
        fi
    fi

    if [ "${ROLLOUT_MAX_RESPONSE_LEN}" -gt "${SEQ_LENGTH}" ]; then
        echo "ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN} must be <= SEQ_LENGTH=${SEQ_LENGTH}" >&2
        exit 2
    fi
    if [ -n "${ROLLOUT_MAX_CONTEXT_LEN:-}" ] && [ "${ROLLOUT_MAX_CONTEXT_LEN}" -gt "${SEQ_LENGTH}" ]; then
        echo "ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN} must be <= SEQ_LENGTH=${SEQ_LENGTH}" >&2
        exit 2
    fi
    if [ -n "${ROLLOUT_MAX_PROMPT_LEN:-}" ] && [ -n "${ROLLOUT_MAX_CONTEXT_LEN:-}" ] && [ "${ROLLOUT_MAX_PROMPT_LEN}" -gt "$((ROLLOUT_MAX_CONTEXT_LEN - 1))" ]; then
        echo "ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN} must be <= ROLLOUT_MAX_CONTEXT_LEN - 1" >&2
        exit 2
    fi

    PROMPT_SET="${ASSET_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    HF_CHECKPOINT="${HF_CHECKPOINT:-${ASSET_DIR}/${MODEL_ASSET_NAME}}"
    RUN_LOG="${RUN_LOG:-${LOG_DIR}/amd-${MODEL_LOG_NAME}-${GPU_LABEL}-${NOW}.log}"
    mkdir -p "$(dirname "${RUN_LOG}")"
    RAY_DASHBOARD_URL="http://${MASTER_ADDR}:8265"
}

cleanup_stale_relax_processes() {
    local stale_pids=()
    while IFS= read -r pid; do
        [ -n "${pid}" ] && stale_pids+=("${pid}")
    done < <(
        python - "${ASSET_DIR}" "$$" "${PPID:-0}" <<'PY'
import os
import sys


asset_dir = sys.argv[1]
skip_pids = {int(pid) for pid in sys.argv[2:] if pid.isdigit()}
skip_pids.update({os.getpid(), os.getppid()})

for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if pid in skip_pids:
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode().strip()
    except OSError:
        continue
    if asset_dir not in cmdline:
        continue
    if "python3 -m relax.entrypoints.train" in cmdline:
        print(pid)
        continue
    if "ray job submit" in cmdline and "relax.entrypoints.train" in cmdline:
        print(pid)

proc_table = {}
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            stat_parts = fh.read().split()
        ppid = int(stat_parts[3])
    except (OSError, IndexError, ValueError):
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode().strip()
    except OSError:
        continue
    proc_table[pid] = {"ppid": ppid, "cmdline": cmdline}

orphan_sglang_names = {"sglang::scheduler", "sglang::detokenizer"}
stale_orphans = set()
for pid, info in proc_table.items():
    if pid in skip_pids:
        continue
    if info["cmdline"] not in orphan_sglang_names:
        continue
    parent = proc_table.get(info["ppid"])
    if parent is None:
        continue
    if parent["ppid"] != 1:
        continue
    if "from multiprocessing.spawn import spawn_main" not in parent["cmdline"]:
        continue
    stale_orphans.add(pid)
    stale_orphans.add(info["ppid"])

for pid in sorted(stale_orphans):
    print(pid)
PY
    )

    if [ "${#stale_pids[@]}" -eq 0 ]; then
        return 0
    fi

    echo "Stopping stale Relax launcher processes: ${stale_pids[*]}" >&2
    kill "${stale_pids[@]}" >/dev/null 2>&1 || true
    sleep 5

    local surviving_pids=()
    local pid
    for pid in "${stale_pids[@]}"; do
        if kill -0 "${pid}" >/dev/null 2>&1; then
            surviving_pids+=("${pid}")
        fi
    done

    if [ "${#surviving_pids[@]}" -gt 0 ]; then
        echo "Force killing stale Relax launcher processes: ${surviving_pids[*]}" >&2
        kill -9 "${surviving_pids[@]}" >/dev/null 2>&1 || true
    fi
}

wait_for_ray_dashboard() {
    local attempt
    for attempt in $(seq 1 30); do
        if localhost_bypass curl -fsS "${RAY_DASHBOARD_URL}/api/version" >/dev/null; then
            return 0
        fi
        sleep 2
    done

    echo "Ray dashboard did not become ready at ${RAY_DASHBOARD_URL}" >&2
    return 1
}

load_model_config() {
    MODEL_CONFIG_PATH="${MODEL_CONFIG_DIR}/${MODEL_CONFIG_NAME}"
    if [[ "${MODEL_CONFIG_PATH}" != *.sh ]]; then
        MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}.sh"
    fi
    source "${MODEL_CONFIG_PATH}"
}

start_ray_head() {
    cleanup_stale_relax_processes
    localhost_bypass python3 -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
    localhost_bypass python3 -m ray.scripts.scripts start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --num-cpus "${RAY_NUM_CPUS:-16}" \
        --num-gpus "${RAY_NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
    wait_for_ray_dashboard
}

configure_rocm_library_preload() {
    ROCM_HSA_RUNTIME_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD:-/opt/rocm-7.0.0/lib/libhsa-runtime64.so.1}"
    if [ ! -f "${ROCM_HSA_RUNTIME_PRELOAD}" ]; then
        echo "ROCM_HSA_RUNTIME_PRELOAD does not exist: ${ROCM_HSA_RUNTIME_PRELOAD}" >&2
        exit 2
    fi

    case ":${LD_PRELOAD:-}:" in
        *":${ROCM_HSA_RUNTIME_PRELOAD}:"*)
            ;;
        *)
            if [ -n "${LD_PRELOAD:-}" ]; then
                export LD_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD}:${LD_PRELOAD}"
            else
                export LD_PRELOAD="${ROCM_HSA_RUNTIME_PRELOAD}"
            fi
            ;;
    esac
    export ROCM_HSA_RUNTIME_PRELOAD
}

build_runtime_env_json() {
    python - <<'PY'
import json
import os
import socket


def resolve_host(addr: str) -> str:
    try:
        socket.inet_pton(socket.AF_INET, addr)
        return addr
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, addr)
        return addr
    except OSError:
        pass
    try:
        return socket.gethostbyname(addr)
    except socket.gaierror:
        return ""


def append_no_proxy_entries(value: str | None, entries: list[str]) -> str:
    combined = []
    seen = set()
    for item in [value or "", *entries]:
        for part in item.split(","):
            candidate = part.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            combined.append(candidate)
    return ",".join(combined)


keys = [
    "PYTHONUNBUFFERED",
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "MASTER_ADDR",
    "HIP_VISIBLE_DEVICES",
    "RELAX_SGL_KERNEL_BUILD_DIR",
    "MEGATRON_BRIDGE_DIR",
    "LD_PRELOAD",
    "ROCM_HSA_RUNTIME_PRELOAD",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "RAY_gcs_rpc_server_reconnect_timeout_s",
    "RAY_py_gcs_connect_timeout_s",
    "RAY_nums_py_gcs_reconnect_retry",
    "RAY_task_events_report_interval_ms",
    "RAY_grpc_client_keepalive_time_ms",
    "RAY_grpc_client_keepalive_timeout_ms",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_GROUP",
    "WANDB_DIR",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
]
env_vars = {k: os.environ[k] for k in keys if os.environ.get(k)}
local_entries = ["127.0.0.1", "localhost", "::1"]
master_addr = os.environ.get("MASTER_ADDR")
if master_addr:
    local_entries.extend([master_addr, resolve_host(master_addr)])
host_name = socket.gethostname()
if host_name:
    local_entries.extend([host_name, resolve_host(host_name)])
env_vars["no_proxy"] = append_no_proxy_entries(env_vars.get("no_proxy"), local_entries)
env_vars["NO_PROXY"] = append_no_proxy_entries(env_vars.get("NO_PROXY"), local_entries)
env_vars["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
env_vars["RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"] = "1"
print(json.dumps({"env_vars": env_vars}))
PY
}

build_asynchronous_rl_args() {
    ASYNC_RL_ARGS=(
        --max-staleness "${MAX_STALENESS}"
    )
    case "${RELAX_EXECUTION_MODE}" in
        fully_async)
            ASYNC_RL_ARGS+=(--fully-async)
            ;;
        hybrid)
            ASYNC_RL_ARGS+=(--hybrid)
            ;;
        sync)
            ;;
    esac
}

build_checkpoint_args() {
    CKPT_ARGS=(
        --hf-checkpoint "${HF_CHECKPOINT}"
        --ref-load "${HF_CHECKPOINT}"
        --megatron-to-hf-mode bridge
        --save "${SAVE_DIR}"
        --save-interval "${SAVE_INTERVAL}"
        --ckpt-format "${CKPT_FORMAT}"
    )

    if [ -n "${LOAD_DIR}" ]; then
        CKPT_ARGS+=(--load "${LOAD_DIR}")
    fi
    if [ "${NO_SAVE_OPTIM}" = "1" ]; then
        CKPT_ARGS+=(--no-save-optim)
    fi
    if [ "${NO_SAVE_RNG}" = "1" ]; then
        CKPT_ARGS+=(--no-save-rng)
    fi
    if [ "${NO_LOAD_RNG}" = "1" ]; then
        CKPT_ARGS+=(--no-load-rng)
    fi

    case "${SCHEDULER_RESUME_POLICY}" in
        strict)
            ;;
        override)
            CKPT_ARGS+=(--override-opt-param-scheduler)
            ;;
        checkpoint)
            CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
            ;;
        *)
            echo "Unsupported SCHEDULER_RESUME_POLICY=${SCHEDULER_RESUME_POLICY}; expected strict, override, or checkpoint" >&2
            exit 2
            ;;
    esac
}

append_rollout_arg() {
    local value="$1"
    local flag="$2"

    if [ -n "${value}" ]; then
        ROLLOUT_ARGS+=("${flag}" "${value}")
    fi
}

build_rollout_args() {
    ROLLOUT_ARGS=(
        --use-streaming-dataset
        --streaming-buffer-size 10000
        --prompt-data "${PROMPT_SET}"
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type dapo
        --reward-key score
        --num-rollout "${NUM_ROLLOUT}"
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
        --rollout-temperature 0.8
        --global-batch-size "${GLOBAL_BATCH_SIZE}"
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
        --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL:-1}"
        --use-fault-tolerance
    )
    if [ "${USE_BALANCE_DATA}" = "1" ]; then
        ROLLOUT_ARGS+=(--balance-data)
    fi
    append_rollout_arg "${ROLLOUT_MAX_CONTEXT_LEN:-}" --rollout-max-context-len
    append_rollout_arg "${ROLLOUT_MAX_PROMPT_LEN:-}" --rollout-max-prompt-len
}

build_megatron_args() {
    MEGATRON_PARALLEL_ARGS=(
        --seq-length "${SEQ_LENGTH}"
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
        --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
        --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
        --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
        --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
        --micro-batch-size "${MICRO_BATCH_SIZE}"
    )
    if [ "${ENABLE_RECOMPUTE}" = "1" ]; then
        MEGATRON_PARALLEL_ARGS+=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
    fi
    if [ "${ENABLE_SEQUENCE_PARALLEL}" = "1" ]; then
        MEGATRON_PARALLEL_ARGS+=(--sequence-parallel)
    fi
}

build_algorithm_args() {
    GRPO_ARGS=(
        --advantage-estimator grpo
        --entropy-coef 0.0
        --eps-clip 0.2
        --eps-clip-high 0.28
        --use-tis
    )
    if [ "${USE_KL_LOSS}" = "1" ]; then
        GRPO_ARGS+=(
            --use-kl-loss
            --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
            --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}"
        )
    fi
}

build_optimizer_args() {
    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr "${LR:-1e-6}"
        --lr-decay-style constant
        --weight-decay "${WEIGHT_DECAY:-0.1}"
        --adam-beta1 0.9
        --adam-beta2 0.98
    )
}

build_wandb_args() {
    WANDB_ARGS=(
        --use-wandb
        --wandb-mode "${WANDB_MODE:-online}"
        --wandb-team "${WANDB_ENTITY}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-dir "${WANDB_DIR}"
        --disable-wandb-random-suffix
    )
}

append_env_arg() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"

    if [ -n "${value}" ]; then
        DEBUG_ARGS+=("${flag}" "${value}")
    fi
}

build_debug_args() {
    DEBUG_ARGS=()
    append_env_arg RELAX_DUMP_DETAILS --dump-details
    append_env_arg RELAX_SAVE_DEBUG_ROLLOUT_DATA --save-debug-rollout-data
    append_env_arg RELAX_SAVE_DEBUG_TRAIN_DATA --save-debug-train-data
    append_env_arg RELAX_LOAD_DEBUG_ROLLOUT_DATA --load-debug-rollout-data
    append_env_arg RELAX_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE --load-debug-rollout-data-subsample
    append_env_arg RELAX_CUSTOM_MEGATRON_BEFORE_TRAIN_STEP_HOOK_PATH --custom-megatron-before-train-step-hook-path
}

append_profiling_arg() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"

    if [ -n "${value}" ]; then
        PROFILING_ARGS+=("${flag}" "${value}")
    fi
}

append_profiling_flag() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"

    case "${value}" in
        "")
            ;;
        0)
            ;;
        1)
            PROFILING_ARGS+=("${flag}")
            ;;
        *)
            echo "${env_name} must be 0 or 1 when set, got ${value}" >&2
            exit 2
            ;;
    esac
}

append_profiling_list_arg() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    local values=()

    if [ -z "${value}" ]; then
        return 0
    fi

    read -ra values <<< "${value}"
    if [ "${#values[@]}" -eq 0 ]; then
        return 0
    fi
    PROFILING_ARGS+=("${flag}" "${values[@]}")
}

build_profiling_args() {
    PROFILING_ARGS=()

    append_profiling_arg RELAX_TB_EXPERIMENT_NAME --tb-experiment-name
    append_profiling_arg RELAX_TIMELINE_DUMP_DIR --timeline-dump-dir

    append_profiling_flag RELAX_USE_PYTORCH_PROFILER --use-pytorch-profiler
    append_profiling_list_arg RELAX_PROFILE_TARGETS --profile-target
    append_profiling_arg RELAX_PROFILE_STEP_START --profile-step-start
    append_profiling_arg RELAX_PROFILE_STEP_END --profile-step-end
    append_profiling_flag RELAX_PROFILE_WITH_STACK --profile-with-stack
    append_profiling_flag RELAX_PROFILE_WITH_MEMORY --profile-with-memory
    append_profiling_flag RELAX_PROFILE_WITH_FLOPS --profile-with-flops

    append_profiling_flag RELAX_SGLANG_PROFILE --sglang-profile
    append_profiling_arg RELAX_SGLANG_PROFILE_STEP_START --sglang-profile-step-start
    append_profiling_arg RELAX_SGLANG_PROFILE_STEP_END --sglang-profile-step-end
    append_profiling_list_arg RELAX_SGLANG_PROFILE_STEPS --sglang-profile-steps
    append_profiling_arg RELAX_SGLANG_PROFILE_NUM_STEPS --sglang-profile-num-steps
    append_profiling_list_arg RELAX_SGLANG_PROFILE_ACTIVITIES --sglang-profile-activities
    append_profiling_flag RELAX_SGLANG_PROFILE_BY_STAGE --sglang-profile-by-stage
    append_profiling_flag RELAX_SGLANG_PROFILE_WITH_STACK --sglang-profile-with-stack
    append_profiling_flag RELAX_SGLANG_PROFILE_RECORD_SHAPES --sglang-profile-record-shapes
    append_profiling_arg RELAX_SGLANG_PROFILE_OUTPUT_DIR --sglang-profile-output-dir
}

build_sglang_args() {
    SGLANG_ARGS=(
        --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
        --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
        --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
        --sglang-pipeline-parallel-size "${SGLANG_PIPELINE_PARALLEL_SIZE}"
        --sglang-data-parallel-size "${SGLANG_DATA_PARALLEL_SIZE}"
        --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
        --sglang-model-impl transformers
        --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
        --sglang-sampling-backend pytorch
        --sglang-disable-custom-all-reduce
        --sglang-disable-cuda-graph
        --sglang-disable-overlap-schedule
    )

    if [ "${SGLANG_ENABLE_DP_ATTENTION}" = "1" ]; then
        SGLANG_ARGS+=(--sglang-enable-dp-attention)
    fi
    if [ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
        SGLANG_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
    fi
    if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
        SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
    fi
}

build_rocm_compat_args() {
    ROCM_COMPAT_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend unfused
        --no-masked-softmax-fusion
        --no-bias-dropout-fusion
        --no-gradient-accumulation-fusion
        --qkv-format bshd
        --no-rope-fusion
    )
}

build_training_args() {
    build_checkpoint_args
    build_rollout_args
    build_megatron_args
    build_algorithm_args
    build_optimizer_args
    build_wandb_args
    build_debug_args
    build_profiling_args
    build_sglang_args
    build_rocm_compat_args
    build_asynchronous_rl_args
}

log_launch_config() {
    echo "Launching Relax AMD ${MODEL_LOG_NAME} run" >&2
    echo "  mode: ${RELAX_EXECUTION_MODE}, max_staleness=${MAX_STALENESS}, balance_data=${USE_BALANCE_DATA}, use_kl_loss=${USE_KL_LOSS}" >&2
    echo "  resources: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}, RAY_NUM_GPUS=${RAY_NUM_GPUS}, actor_gpus=${ACTOR_RESOURCE_GPUS}, rollout_gpus=${ROLLOUT_RESOURCE_GPUS}, actor_fwd_gpus=${ACTOR_FWD_RESOURCE_GPUS}, RESOURCE_JSON=${RESOURCE_JSON}" >&2
    echo "  training: TP=${TENSOR_MODEL_PARALLEL_SIZE}, PP=${PIPELINE_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, micro_batch=${MICRO_BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}, recompute=${ENABLE_RECOMPUTE}" >&2
    echo "  rollout: num_rollout=${NUM_ROLLOUT}, steps_per_rollout=${NUM_STEPS_PER_ROLLOUT}, rollout_batch=${ROLLOUT_BATCH_SIZE}, samples_per_prompt=${N_SAMPLES_PER_PROMPT}" >&2
    echo "  transfer_queue: num_data_storage_units=${NUM_DATA_STORAGE_UNITS}" >&2
    echo "  sequence: seq_length=${SEQ_LENGTH}, rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}, rollout_max_context_len=${ROLLOUT_MAX_CONTEXT_LEN:-unset}, rollout_max_prompt_len=${ROLLOUT_MAX_PROMPT_LEN:-unset}" >&2
    echo "  sglang: gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}, pp=${SGLANG_PIPELINE_PARALLEL_SIZE}, dp=${SGLANG_DATA_PARALLEL_SIZE}, ep=${SGLANG_EXPERT_PARALLEL_SIZE}, attention_backend=${SGLANG_ATTENTION_BACKEND}, server_concurrency=${SGLANG_SERVER_CONCURRENCY}, max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS:-unset}, max_total_tokens=${SGLANG_MAX_TOTAL_TOKENS:-unset}" >&2
}

submit_training_job() {
    local tee_args=()
    if [ "${RUN_LOG_APPEND:-0}" = "1" ]; then
        tee_args=(-a)
    fi

    localhost_bypass python3 -m ray.scripts.scripts job submit --address="${RAY_DASHBOARD_URL}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 -m relax.entrypoints.train \
        --resource "${RESOURCE_JSON}" \
        --num-data-storage-units "${NUM_DATA_STORAGE_UNITS}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${DEBUG_ARGS[@]}" \
        "${PROFILING_ARGS[@]}" \
        "${MEGATRON_PARALLEL_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${ROCM_COMPAT_ARGS[@]}" \
        "${ASYNC_RL_ARGS[@]}" 2>&1 | tee "${tee_args[@]}" "${RUN_LOG}"
}

main() {
    activate_environment
    load_dotenv

    configure_gpu_resources
    configure_execution_mode
    configure_sglang_parallelism
    configure_megatron_parallelism
    configure_runtime_environment
    configure_run_defaults
    load_model_config
    append_no_proxy

    start_ray_head
    configure_rocm_library_preload
    RUNTIME_ENV_JSON="$(build_runtime_env_json)"
    build_training_args
    log_launch_config
    submit_training_job
}

main "$@"
