#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PYTHON="$(command -v "${PYTHON}")"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REFINEMENT_DATA="${REFINEMENT_DATA:-}"
PROMPT_SET="${PROMPT_SET:-}"
VISION_DEVICE_PLAN="${VISION_DEVICE_PLAN:-}"
RUN_TAG="$(date -u +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${VISION_DEVICE_ARTIFACT_DIR:-${ROOT_DIR}/benchmark_results/cpu_vision/compare_devices_${RUN_TAG}}"
MANIFEST_TSV="${ARTIFACT_DIR}/manifest.tsv"
MANIFEST_JSON="${ARTIFACT_DIR}/analysis_manifest.json"
ANALYSIS_JSON="${ARTIFACT_DIR}/device_comparison.json"
PROFILE_TIMEOUT_SECONDS="${PROFILE_TIMEOUT_SECONDS:-5400}"
GPU_LAUNCHER="${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh"
CPU_LAUNCHER="${ROOT_DIR}/scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh"

orders=(
    "gpu cpu automatic"
    "cpu automatic gpu"
    "automatic gpu cpu"
    "gpu automatic cpu"
    "cpu gpu automatic"
)
train_seeds=(1234 1235 1236 1237 1238)
rollout_seeds=(42 43 44 45 46)

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}/log"
for command in flock git sha256sum lscpu rocm-smi ray timeout pgrep; do
    if ! command -v "${command}" >/dev/null; then
        echo "Required vision-device comparison command is unavailable: ${command}" >&2
        exit 2
    fi
done
if [ -n "${RAY_ADDRESS:-}" ]; then
    echo "RAY_ADDRESS must be unset for the direct vision-device comparison" >&2
    exit 2
fi
if ! [[ "${PROFILE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_TIMEOUT_SECONDS must be a positive integer, got ${PROFILE_TIMEOUT_SECONDS}" >&2
    exit 2
fi
exec 9>"/tmp/relax-ray-${UID}.lock"
if ! flock -n 9; then
    echo "Another same-user Relax runner holds the node-local Ray ownership lock" >&2
    exit 2
fi
for path in \
    "${PYTHON}" \
    "${HF_CHECKPOINT}/config.json" \
    "${HF_CHECKPOINT}/model.safetensors" \
    "${PROMPT_SET}" \
    "${VISION_DEVICE_PLAN}"; do
    if [ -z "${HF_CHECKPOINT}" ] || [ -z "${REFINEMENT_DATA}" ] || [ ! -e "${path}" ]; then
        echo "Required vision-device comparison input does not exist: ${path:-unset}" >&2
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
    raise SystemExit(f"vision-device comparison requires exactly four visible GPUs; found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
if not all("MI210" in name for name in names):
    raise SystemExit(f"vision-device comparison requires four MI210 GPUs; found {names}")
print(f"GPU preflight passed: {names}")
PY
"${PYTHON}" - "${VISION_DEVICE_PLAN}" <<'PY'
import sys

from relax.engine.rollout.vision_device import load_vision_device_plan


plan = load_vision_device_plan(sys.argv[1])
devices = {plan.choose(cycle).device for cycle in range(24)}
if devices != {"cpu", "gpu"}:
    raise SystemExit(f"device plan must select CPU and GPU over cycles 0-23; got {devices}")
PY

git rev-parse HEAD > "${ARTIFACT_DIR}/git-head.txt"
git status --porcelain=v1 -uall > "${ARTIFACT_DIR}/git-status.txt"
sha256sum \
    "${PROMPT_SET}" \
    "${VISION_DEVICE_PLAN}" \
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
printf 'repeat\torder_position\trun\ttrain_seed\trollout_seed\tstarted_at\tended_at\texit_status\tconsole_log\tinternal_run_log\n' \
    > "${MANIFEST_TSV}"

for repeat_index in "${!orders[@]}"; do
    repeat="repeat_$((repeat_index + 1))"
    train_seed="${train_seeds[repeat_index]}"
    rollout_seed="${rollout_seeds[repeat_index]}"
    read -ra runs <<< "${orders[repeat_index]}"
    for order_position in "${!runs[@]}"; do
        run="${runs[order_position]}"
        launcher="${GPU_LAUNCHER}"
        device_mode=fixed
        run_plan=""
        if [ "${run}" != "gpu" ]; then
            launcher="${CPU_LAUNCHER}"
        fi
        if [ "${run}" = "automatic" ]; then
            device_mode=automatic
            run_plan="${VISION_DEVICE_PLAN}"
        fi
        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        console_log="${ARTIFACT_DIR}/${repeat}_${run}.console.log"
        internal_run_log="${ARTIFACT_DIR}/${repeat}_${run}.internal.log"
        echo "=== vision-device comparison ${repeat} position=${order_position} run=${run} ==="
        assert_clean_runtime
        owns_runtime=1
        set +e
        timeout --signal=TERM --kill-after=60s "${PROFILE_TIMEOUT_SECONDS}" env \
            ENABLE_EVAL=0 \
            SAVE_CHECKPOINTS=0 \
            HF_CHECKPOINT="${HF_CHECKPOINT}" \
            REFINEMENT_DATA="${REFINEMENT_DATA}" \
            PROMPT_SET="${PROMPT_SET}" \
            NUM_ROLLOUT=24 \
            NUM_STEPS_PER_ROLLOUT=2 \
            ROLLOUT_BATCH_SIZE=32 \
            N_SAMPLES_PER_PROMPT=2 \
            GLOBAL_BATCH_SIZE=32 \
            TRAIN_SEED="${train_seed}" \
            ROLLOUT_SEED="${rollout_seed}" \
            RAY_NUM_CPUS=16 \
            MAX_STALENESS=0 \
            VISION_ENCODER_NUM_REPLICAS=1 \
            VISION_ENCODER_NUM_CPUS=1 \
            SKIP_GPU_VISION_ENCODER=0 \
            VISION_ENCODER_CACHE_MAX_BYTES=1 \
            SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=0 \
            PRELOAD_VISION_FEATURES=0 \
            VISION_DEVICE_MODE="${device_mode}" \
            VISION_DEVICE_PLAN="${run_plan}" \
            RUN_LOG="${internal_run_log}" \
            bash "${launcher}" 2>&1 | tee "${console_log}"
        exit_status="${PIPESTATUS[0]}"
        set -e
        cleanup_owned_runtime
        ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${repeat}" "${order_position}" "${run}" "${train_seed}" "${rollout_seed}" \
            "${started_at}" "${ended_at}" "${exit_status}" "${console_log}" "${internal_run_log}" \
            >> "${MANIFEST_TSV}"
        if [ "${exit_status}" -ne 0 ]; then
            echo "Vision-device run ${repeat}:${run} failed with exit status ${exit_status}" >&2
            exit "${exit_status}"
        fi
    done
done

"${PYTHON}" - "${MANIFEST_TSV}" "${MANIFEST_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path


manifest_tsv, output_path = map(Path, sys.argv[1:])
repeats = {}
with manifest_tsv.open() as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if int(row["exit_status"]) != 0:
            raise SystemExit(f"refusing to analyze failed run: {row}")
        repeat = repeats.setdefault(row["repeat"], {"name": row["repeat"], "runs": {}})
        repeat["runs"][row["run"]] = row["console_log"]
output_path.write_text(
    json.dumps(
        {
            "schema_version": 2,
            "measured_cycles": list(range(24)),
            "expected_responses_per_cycle": 64,
            "expected_generation_requests_per_cycle": 32,
            "require_both_devices": True,
            "repeats": [repeats[name] for name in sorted(repeats)],
        },
        indent=2,
    )
    + "\n"
)
PY
"${PYTHON}" -m examples.visual_xor.compare_vision_device_choices \
    --manifest "${MANIFEST_JSON}" \
    --output "${ANALYSIS_JSON}"

rocm-smi --showid --showproductname --showmeminfo vram --csv > "${ARTIFACT_DIR}/vram-after.csv"
echo "Vision-device comparison completed; manifest=${MANIFEST_JSON}; analysis=${ANALYSIS_JSON}"
