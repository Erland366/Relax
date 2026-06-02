#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${ROOT_DIR}/log"
RESULTS_FILE="${ROOT_DIR}/results.tsv"
LAUNCHER="${ROOT_DIR}/amd_qwen3_4b_2gpu_e2e.sh"
HEALTHY_SECONDS="${HEALTHY_SECONDS:-600}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"

mkdir -p "${LOG_DIR}"

if [ ! -f "${RESULTS_FILE}" ]; then
    printf 'commit\tstatus\tdescription\n' >"${RESULTS_FILE}"
fi

log_result() {
    local status="$1"
    local description="$2"
    local commit
    commit="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo no-git)"
    printf '%s\t%s\t%s\n' "${commit}" "${status}" "${description}" >>"${RESULTS_FILE}"
}

cleanup_child() {
    if [ -n "${child_pid:-}" ] && kill -0 "${child_pid}" >/dev/null 2>&1; then
        kill "${child_pid}" >/dev/null 2>&1 || true
    fi
}

trap cleanup_child EXIT INT TERM

attempt=0
while true; do
    attempt=$((attempt + 1))
    started_at="$(date '+%Y-%m-%d %H:%M:%S')"
    attempt_log="${LOG_DIR}/amd-qwen3-4b-overnight-attempt-${attempt}-$(date '+%Y%m%d_%H%M%S').log"

    echo "[loop] attempt=${attempt} started_at=${started_at} health_window=${HEALTHY_SECONDS}s log=${attempt_log}"

    set +e
    "${LAUNCHER}" 2>&1 | tee "${attempt_log}" &
    child_pid=$!
    set -e

    became_healthy=0
    attempt_started_epoch="$(date +%s)"

    while kill -0 "${child_pid}" >/dev/null 2>&1; do
        now_epoch="$(date +%s)"
        runtime_seconds=$((now_epoch - attempt_started_epoch))
        if [ "${runtime_seconds}" -ge "${HEALTHY_SECONDS}" ] && [ "${became_healthy}" -eq 0 ]; then
            became_healthy=1
            echo "[loop] attempt=${attempt} stayed alive for ${runtime_seconds}s and is considered healthy"
            log_result "healthy" "attempt=${attempt} runtime=${runtime_seconds}s log=${attempt_log}"
        fi
        sleep 15
    done

    set +e
    wait "${child_pid}"
    exit_code=$?
    set -e
    child_pid=""

    ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
    final_runtime_seconds=$(( $(date +%s) - attempt_started_epoch ))

    if [ "${exit_code}" -eq 0 ]; then
        echo "[loop] attempt=${attempt} completed successfully after ${final_runtime_seconds}s"
        log_result "success" "attempt=${attempt} runtime=${final_runtime_seconds}s log=${attempt_log}"
        exit 0
    fi

    if [ "${became_healthy}" -eq 1 ]; then
        echo "[loop] attempt=${attempt} failed after healthy runtime; exit_code=${exit_code}; restarting in ${RETRY_DELAY_SECONDS}s"
        log_result "retry" "attempt=${attempt} exit=${exit_code} runtime=${final_runtime_seconds}s started_at=${started_at} ended_at=${ended_at} log=${attempt_log}"
    else
        echo "[loop] attempt=${attempt} failed before healthy window; exit_code=${exit_code}; restarting in ${RETRY_DELAY_SECONDS}s"
        log_result "startup-failure" "attempt=${attempt} exit=${exit_code} runtime=${final_runtime_seconds}s started_at=${started_at} ended_at=${ended_at} log=${attempt_log}"
    fi

    sleep "${RETRY_DELAY_SECONDS}"
done
