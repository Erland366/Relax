#!/usr/bin/env bash

# Explicit ROCm runtime contract for the Relax ROCm/Megatron stack.
#
# Source this after conda activation and after loading .env:
#   source scripts/setup/rocm_runtime_env.sh
#
# This keeps ROCm's HSA runtime ahead of PyTorch's bundled HSA runtime, which is
# required when ROCm TransformerEngine loads /opt/rocm-7.0.0/lib/libamdhip64.so.7
# after `import torch`.

_relax_rocm_runtime_fail() {
    echo "error: $*" >&2
    return 1 2>/dev/null || exit 1
}

_relax_rocm_prepend_colon_path() {
    local name="$1"
    local path_entry="$2"
    local current_value="${!name:-}"

    case ":${current_value}:" in
        *":${path_entry}:"*)
            ;;
        *)
            export "${name}=${path_entry}${current_value:+:${current_value}}"
            ;;
    esac
}

ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.0.0}"
if [ ! -d "${ROCM_PATH}" ]; then
    _relax_rocm_runtime_fail "ROCM_PATH=${ROCM_PATH} does not exist"
fi

export ROCM_PATH
export ROCM_HOME="${ROCM_HOME:-${ROCM_PATH}}"
export HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"

_relax_rocm_prepend_colon_path LD_LIBRARY_PATH "${ROCM_PATH}/lib"
_relax_rocm_prepend_colon_path LD_LIBRARY_PATH "${ROCM_PATH}/lib64"

ROCM_PRELOAD_HSA="${ROCM_PRELOAD_HSA:-1}"
if [ "${ROCM_PRELOAD_HSA}" = "1" ]; then
    ROCM_HSA_RUNTIME_LIB="${ROCM_HSA_RUNTIME_LIB:-${ROCM_PATH}/lib/libhsa-runtime64.so.1}"
    if [ ! -f "${ROCM_HSA_RUNTIME_LIB}" ]; then
        _relax_rocm_runtime_fail "ROCM_HSA_RUNTIME_LIB=${ROCM_HSA_RUNTIME_LIB} does not exist"
    fi
    if command -v readelf >/dev/null 2>&1; then
        if ! readelf -Ws "${ROCM_HSA_RUNTIME_LIB}" | grep -q "hsa_amd_memory_get_preferred_copy_engine.*ROCR_1"; then
            _relax_rocm_runtime_fail "ROCM_HSA_RUNTIME_LIB=${ROCM_HSA_RUNTIME_LIB} does not export hsa_amd_memory_get_preferred_copy_engine@@ROCR_1"
        fi
    fi
    export ROCM_HSA_RUNTIME_LIB
    _relax_rocm_prepend_colon_path LD_PRELOAD "${ROCM_HSA_RUNTIME_LIB}"
fi

export NVTE_FUSED_ATTN_CK="${NVTE_FUSED_ATTN_CK:-0}"
export RELAX_ROCM_RUNTIME_ENV=1

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
unset -f _relax_rocm_prepend_colon_path
unset -f _relax_rocm_runtime_fail
