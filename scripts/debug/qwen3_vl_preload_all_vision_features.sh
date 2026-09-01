#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PYTHON="$(command -v "${PYTHON}")"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REFINEMENT_DATA="${REFINEMENT_DATA:-}"
PROMPT_SET="${PROMPT_SET:-${REFINEMENT_DATA:+${REFINEMENT_DATA}/refinement_rl_train.parquet}}"
RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${VISION_PRELOAD_ARTIFACT_DIR:-${ROOT_DIR}/benchmark_results/cpu_vision/full_dataset_preload_${RUN_TAG}}"
MANIFEST_PATH="${ARTIFACT_DIR}/manifest.tsv"
ANALYSIS_PATH="${ARTIFACT_DIR}/full_dataset_preload.json"
LAUNCHER="${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"

orders=("none full_dataset" "full_dataset none" "none full_dataset" "full_dataset none")
train_seeds=(1234 1235 1236 1237)
rollout_seeds=(42 43 44 45)

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"

for required_command in git sha256sum lscpu rocm-smi ray timeout pgrep; do
    if ! command -v "${required_command}" >/dev/null; then
        echo "Required full-dataset preload command is unavailable: ${required_command}" >&2
        exit 2
    fi
done

for required_path in \
    "${PYTHON}" \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${PROMPT_SET}"; do
    if [ -z "${HF_CHECKPOINT}" ] || [ -z "${REFINEMENT_DATA}" ] || [ ! -e "${required_path}" ]; then
        echo "Required full-dataset preload input does not exist: ${required_path}" >&2
        exit 2
    fi
done

"${PYTHON}" -m examples.visual_xor.cpu_vision_capacity_preflight \
    --vision-encoder-num-replicas=1 \
    --vision-encoder-num-cpus=1 \
    --min-affinity-cpus=16 \
    --min-physical-cores=16

"${PYTHON}" - <<'PY'
import torch


count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"full-dataset preload study requires exactly four visible GPUs; found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if not all("MI210" in name for name in names):
    raise SystemExit(f"full-dataset preload study requires four MI210 GPUs; found {names}")
print(f"GPU preflight passed: {names}")
PY

git rev-parse HEAD > "${ARTIFACT_DIR}/git-head.txt"
git status --porcelain=v1 -uall > "${ARTIFACT_DIR}/git-status.txt"
sha256sum \
    "${PROMPT_SET}" \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    > "${ARTIFACT_DIR}/input-sha256.txt"
lscpu --extended=CPU,CORE,SOCKET,NODE,ONLINE > "${ARTIFACT_DIR}/cpu-topology.txt"
rocm-smi --showid --showproductname --showmeminfo vram --csv > "${ARTIFACT_DIR}/vram-before.csv"

owns_runtime=0

assert_clean_runtime() {
    local current_uid
    current_uid="$(id -u)"
    if pgrep -u "${current_uid}" -f '(raylet|gcs_server|ray::|sglang.*launch_server|relax\.entrypoints\.train)' \
        >/dev/null; then
        echo "Existing Ray, SGLang, or Relax processes detected; refusing to claim runtime ownership." >&2
        return 1
    fi
}

cleanup_owned_runtime() {
    if [ "${owns_runtime}" != "1" ]; then
        return
    fi
    owns_runtime=0
    (
        set +e
        timeout 60 ray serve shutdown -y
        ray stop --force
    )
}
trap cleanup_owned_runtime EXIT

assert_clean_runtime

printf 'repeat\torder_position\tpreload\ttrain_seed\trollout_seed\tpreload_enabled\tpreload_artifact\tstarted_at\tended_at\texit_status\tconsole_log\tinternal_run_log\n' \
    > "${MANIFEST_PATH}"

extract_preload_summary() {
    local console_log="$1"
    local output_path="$2"
    "${PYTHON}" - "${console_log}" "${output_path}" <<'PY'
import ast
import json
import math
import re
import sys
from pathlib import Path


console_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
prefix = "VISION_FEATURE_PRELOAD"
matches = [
    ast.literal_eval(clean_line.split(prefix, 1)[1].lstrip(": "))
    for line in console_path.read_text(errors="replace").splitlines()
    if prefix in (clean_line := re.sub(r"\x1b\[[0-9;]*m", "", line))
]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one preload record in {console_path}; found {len(matches)}")
metrics = matches[0]
if metrics.get("schema_version") != 2:
    raise SystemExit(f"preload summary requires schema_version 2: {metrics!r}")
wall_seconds = float(metrics["wall_seconds"])
if not math.isfinite(wall_seconds) or wall_seconds < 0:
    raise SystemExit(f"preload wall_seconds must be finite and non-negative: {wall_seconds}")
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
PY
}

validate_completed_run() {
    local repeat="$1"
    local preload="$2"
    local console_log="$3"
    "${PYTHON}" - "${repeat}" "${preload}" "${console_log}" <<'PY'
import sys

from examples.visual_xor.measure_full_dataset_preload import _analyze_preload_run


repeat, preload, console_log = sys.argv[1:]
_analyze_preload_run(
    repeat,
    preload,
    console_log,
    expected_cycles=22,
    expected_responses=64,
    optimizer_steps_per_cycle=2,
)
PY
}

analysis_args=()
preload_args=()
for repeat_index in "${!orders[@]}"; do
    repeat="repeat_$((repeat_index + 1))"
    train_seed="${train_seeds[repeat_index]}"
    rollout_seed="${rollout_seeds[repeat_index]}"
    read -ra preload_settings <<< "${orders[repeat_index]}"
    for order_position in "${!preload_settings[@]}"; do
        preload="${preload_settings[order_position]}"
        case "${preload}" in
            none)
                cpu_cache_bytes=1
                sglang_bytes=0
                preload_enabled=0
                preload_artifact="-"
                ;;
            full_dataset)
                cpu_cache_bytes=1073741824
                sglang_bytes=1073741824
                preload_enabled=1
                preload_artifact="${ARTIFACT_DIR}/${repeat}_preload.json"
                ;;
            *)
                echo "Unknown preload setting: ${preload}" >&2
                exit 2
                ;;
        esac

        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        console_log="${ARTIFACT_DIR}/${repeat}_${preload}.console.log"
        internal_run_log="${ARTIFACT_DIR}/${repeat}_${preload}.internal.log"
        echo "=== full-dataset preload ${repeat} position=${order_position} preload=${preload} ==="
        assert_clean_runtime
        owns_runtime=1
        set +e
        env \
            ENABLE_EVAL=0 \
            SAVE_CHECKPOINTS=0 \
            HF_CHECKPOINT="${HF_CHECKPOINT}" \
            REFINEMENT_DATA="${REFINEMENT_DATA}" \
            PROMPT_SET="${PROMPT_SET}" \
            NUM_ROLLOUT=22 \
            NUM_STEPS_PER_ROLLOUT=2 \
            ROLLOUT_BATCH_SIZE=32 \
            N_SAMPLES_PER_PROMPT=2 \
            GLOBAL_BATCH_SIZE=32 \
            TRAIN_SEED="${train_seed}" \
            ROLLOUT_SEED="${rollout_seed}" \
            RAY_NUM_CPUS=16 \
            CPU_VISION_CAPACITY_MODE=1 \
            MAX_STALENESS=4 \
            VISION_ENCODER_NUM_REPLICAS=1 \
            VISION_ENCODER_NUM_CPUS=1 \
            SKIP_GPU_VISION_ENCODER=1 \
            VISION_ENCODER_CACHE_MAX_BYTES="${cpu_cache_bytes}" \
            VISION_ENCODER_MAX_IMAGES_PER_REQUEST=8 \
            VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0 \
            SGLANG_VISION_FEATURE_CACHE_MAX_BYTES="${sglang_bytes}" \
            PRELOAD_VISION_FEATURES="${preload_enabled}" \
            RUN_LOG="${internal_run_log}" \
            bash "${LAUNCHER}" 2>&1 | tee "${console_log}"
        exit_status="${PIPESTATUS[0]}"
        set -e
        cleanup_owned_runtime
        ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${repeat}" "${order_position}" "${preload}" "${train_seed}" "${rollout_seed}" \
            "${preload_enabled}" "${preload_artifact}" "${started_at}" "${ended_at}" \
            "${exit_status}" "${console_log}" "${internal_run_log}" >> "${MANIFEST_PATH}"
        if [ "${exit_status}" -ne 0 ]; then
            echo "Full-dataset preload run ${repeat}:${preload} failed with exit status ${exit_status}" >&2
            exit "${exit_status}"
        fi

        validate_completed_run "${repeat}" "${preload}" "${console_log}"
        analysis_args+=(--run "${repeat}:${preload}=${console_log}")
        if [ "${preload}" = "full_dataset" ]; then
            extract_preload_summary "${console_log}" "${preload_artifact}"
            preload_args+=(--preload-artifact "${repeat}=${preload_artifact}")
        fi
    done
done

"${PYTHON}" -m examples.visual_xor.measure_full_dataset_preload \
    "${analysis_args[@]}" \
    "${preload_args[@]}" \
    --expected-cycles 22 \
    --expected-responses 64 \
    --optimizer-steps-per-cycle 2 \
    --output "${ANALYSIS_PATH}"

rocm-smi --showid --showproductname --showmeminfo vram --csv > "${ARTIFACT_DIR}/vram-after.csv"
echo "Full-dataset preload study completed; manifest=${MANIFEST_PATH}; analysis=${ANALYSIS_PATH}"
