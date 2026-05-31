#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ASSET_DIR="/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets"
MEGATRON_DIR="${MEGATRON_DIR:-/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM}"
LOG_DIR="${ROOT_DIR}/log"
mkdir -p "${LOG_DIR}"

source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-relaxrl_rocm}"
conda activate "${CONDA_ENV_NAME}"

if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi
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
ROLLOUT_RESOURCE_GPUS="${ROLLOUT_RESOURCE_GPUS:-$((RAY_NUM_GPUS - ACTOR_RESOURCE_GPUS))}"
if [ "${RAY_NUM_GPUS}" -lt 1 ]; then
    echo "RAY_NUM_GPUS must be >= 1, got ${RAY_NUM_GPUS}" >&2
    exit 2
fi
if [ "${ACTOR_RESOURCE_GPUS}" -lt 1 ]; then
    echo "ACTOR_RESOURCE_GPUS must be >= 1, got ${ACTOR_RESOURCE_GPUS}" >&2
    exit 2
fi
if [ "${ROLLOUT_RESOURCE_GPUS}" -lt 1 ]; then
    echo "ROLLOUT_RESOURCE_GPUS must be >= 1, got ${ROLLOUT_RESOURCE_GPUS}" >&2
    exit 2
fi
if [ "$((ACTOR_RESOURCE_GPUS + ROLLOUT_RESOURCE_GPUS))" -gt "${RAY_NUM_GPUS}" ]; then
    echo "ACTOR_RESOURCE_GPUS + ROLLOUT_RESOURCE_GPUS exceeds RAY_NUM_GPUS" >&2
    exit 2
fi
GPU_LABEL="${GPU_LABEL:-${RAY_NUM_GPUS}gpu}"
RESOURCE_JSON="${RESOURCE_JSON:-{\"actor\": [1, ${ACTOR_RESOURCE_GPUS}], \"rollout\": [1, ${ROLLOUT_RESOURCE_GPUS}]}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${ACTOR_RESOURCE_GPUS}}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if [ -z "${ENABLE_SEQUENCE_PARALLEL+x}" ]; then
    if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
        ENABLE_SEQUENCE_PARALLEL=1
    else
        ENABLE_SEQUENCE_PARALLEL=0
    fi
fi
if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -lt 1 ]; then
    echo "TENSOR_MODEL_PARALLEL_SIZE must be >= 1, got ${TENSOR_MODEL_PARALLEL_SIZE}" >&2
    exit 2
fi
case "${ENABLE_SEQUENCE_PARALLEL}" in
    0 | 1)
        ;;
    *)
        echo "ENABLE_SEQUENCE_PARALLEL must be 0 or 1, got ${ENABLE_SEQUENCE_PARALLEL}" >&2
        exit 2
        ;;
esac
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
export PYTHONPATH="/vast/users/qirong.ho/erland/Python_project/sglang/python:${MEGATRON_DIR}:${ROOT_DIR}"
export MEGATRON="${MEGATRON_DIR}"
export RELAX="${ROOT_DIR}"
export MODEL_CONFIG_DIR="${ROOT_DIR}/scripts/models"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
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
# The single-rank MI210 actor must checkpoint. The ROCm hook keeps Megatron's
# torch_dist path on the PyTorch 2.6-compatible planner/writer flow.
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
CKPT_FORMAT="${CKPT_FORMAT:-torch_dist}"
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
SCHEDULER_RESUME_POLICY="${SCHEDULER_RESUME_POLICY:-strict}"
PROMPT_SET="${ASSET_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
HF_CHECKPOINT="${HF_CHECKPOINT:-${ASSET_DIR}/${MODEL_ASSET_NAME}}"
RUN_LOG="${LOG_DIR}/amd-${MODEL_LOG_NAME}-${GPU_LABEL}-${NOW}.log"
RAY_DASHBOARD_URL="http://${MASTER_ADDR}:8265"

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

MODEL_CONFIG_PATH="${MODEL_CONFIG_DIR}/${MODEL_CONFIG_NAME}"
if [[ "${MODEL_CONFIG_PATH}" != *.sh ]]; then
    MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}.sh"
fi
source "${MODEL_CONFIG_PATH}"

append_no_proxy

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

RUNTIME_ENV_JSON="$(python - <<'PY'
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
)"

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
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-768}"
    --rollout-temperature 0.8
    --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
    --num-steps-per-rollout 2
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
    --micro-batch-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)
if [ "${ENABLE_SEQUENCE_PARALLEL}" = "1" ]; then
    PERF_ARGS+=(--sequence-parallel)
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WAND_ARGS=(
    --use-wandb
    --wandb-mode online
    --wandb-team "${WANDB_ENTITY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
    --disable-wandb-random-suffix
)

DEBUG_ARGS=()
if [ -n "${RELAX_DUMP_DETAILS:-}" ]; then
    DEBUG_ARGS+=(--dump-details "${RELAX_DUMP_DETAILS}")
fi
if [ -n "${RELAX_SAVE_DEBUG_ROLLOUT_DATA:-}" ]; then
    DEBUG_ARGS+=(--save-debug-rollout-data "${RELAX_SAVE_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${RELAX_SAVE_DEBUG_TRAIN_DATA:-}" ]; then
    DEBUG_ARGS+=(--save-debug-train-data "${RELAX_SAVE_DEBUG_TRAIN_DATA}")
fi
if [ -n "${RELAX_LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
    DEBUG_ARGS+=(--load-debug-rollout-data "${RELAX_LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${RELAX_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}" ]; then
    DEBUG_ARGS+=(--load-debug-rollout-data-subsample "${RELAX_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}")
fi

SGLANG_ARGS=(
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    --sglang-model-impl transformers
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --sglang-disable-custom-all-reduce
    --sglang-disable-cuda-graph
    --sglang-disable-overlap-schedule
)
if [ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
    SGLANG_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
fi
if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
    SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi

echo "Launching Relax AMD ${MODEL_LOG_NAME} run with HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}, RAY_NUM_GPUS=${RAY_NUM_GPUS}, RESOURCE_JSON=${RESOURCE_JSON}, TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}" >&2

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend unfused
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --qkv-format bshd
    --no-rope-fusion
)

localhost_bypass python3 -m ray.scripts.scripts job submit --address="${RAY_DASHBOARD_URL}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WAND_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
