#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install ROCm Megatron dependencies used by this Relax fork.

Required:
  scripts/setup/install_rocm_megatron_deps.sh

Optional environment variables:
  CONDA_ENV_NAME                    Conda env to create or reuse. Default: relaxrl_rocm
  CREATE_CONDA_ENV                  1 to create/activate CONDA_ENV_NAME. Default: 1
  CONDA_PYTHON_VERSION              Python version for a new conda env. Default: 3.12
  CONDA_SH                          Conda shell hook. Default: /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
  EXPECTED_CONDA_ENV                Required active env when CREATE_CONDA_ENV=0. Default: CONDA_ENV_NAME
  ALLOW_DIFFERENT_CONDA_ENV         1 to allow another active conda env. Default: 0
  INSTALL_UV                        1 to install uv if no uv executable exists. Default: 1
  CONDA_UV_CHANNEL                  Conda channel used to install uv. Default: conda-forge
  INSTALL_ROCM_PYTORCH              1 to install ROCm PyTorch when missing. Default: 1
  PYTORCH_ROCM_INDEX_URL            Default: https://download.pytorch.org/whl/rocm6.3
  TORCH_PACKAGES                    Default: torch==2.9.1+rocm6.3
  INSTALL_RELAX_REQUIREMENTS        1 to install requirements.txt. Default: 1
  INSTALL_RELAX_EDITABLE            1 to install this checkout as editable. Default: 1
  ROCM_DEPS_SRC_DIR                 Clone/cache directory. Default: .deps/rocm-megatron
  ROCM_TRANSFORMER_ENGINE_REPO      Default: https://github.com/ROCm/TransformerEngine.git
  ROCM_TRANSFORMER_ENGINE_REF       Default: v2.10_rocm
  ROCM_APEX_REPO                    Default: https://github.com/ROCm/apex.git
  ROCM_APEX_REF                     Default: release/1.9.0
  ROCM_PATH                         Default: detected from ROCm CMake packages.
  ROCM_RUNTIME_ENV_SH               Default: scripts/setup/rocm_runtime_env.sh
  ROCM_PRELOAD_HSA                  1 to preload ROCm's HSA runtime. Default: 1
  ROCM_HSA_RUNTIME_LIB              Default: ${ROCM_PATH}/lib/libhsa-runtime64.so.1
  RECREATE_FAILED_CHECKOUT          Move a failed generated checkout aside and retry once. Default: 1
  INSTALL_BUILD_TOOLS               1/0, install Python build helpers. Default: 1
  INSTALL_TRANSFORMER_ENGINE        1/0. Default: 1
  INSTALL_APEX                      1/0. Default: 1
  VALIDATE_TRANSFORMER_ENGINE       1/0. Default: INSTALL_TRANSFORMER_ENGINE
  VALIDATE_APEX                     1/0. Default: INSTALL_APEX
  NVTE_ROCM_ARCH                    Comma-separated ROCm GPU arch list.
                                    Default: detected from active ROCm PyTorch.
  NVTE_FRAMEWORK                    Default: pytorch
  NVTE_FUSED_ATTN_CK                Default: 0 when NVTE_ROCM_ARCH has no gfx942/gfx950 target.
                                    Must also be set at runtime if disabled at build time.
  NVTE_CMAKE_BUILD_DIR              Default: TransformerEngine/build/cmake-${NVTE_ROCM_ARCH}-ck${NVTE_FUSED_ATTN_CK}
  NVTE_BUILD_THREADS_PER_JOB        Default: 1
  MAX_JOBS                          Default: 4
  CMAKE_BUILD_PARALLEL_LEVEL        Default: MAX_JOBS
  APEX_PREBUILD_OPS                 Comma-separated Apex ops to prebuild.
                                    Default: amp_c,fused_adam,fused_layer_norm

Examples:
  scripts/setup/install_rocm_megatron_deps.sh
  CREATE_CONDA_ENV=0 scripts/setup/install_rocm_megatron_deps.sh
  ROCM_TRANSFORMER_ENGINE_REF=dev scripts/setup/install_rocm_megatron_deps.sh
  APEX_PREBUILD_OPS=amp_c,fused_adam,fused_layer_norm,apex_c scripts/setup/install_rocm_megatron_deps.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-${EXPECTED_CONDA_ENV:-relaxrl_rocm}}"
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-${CONDA_ENV_NAME}}"
ALLOW_DIFFERENT_CONDA_ENV="${ALLOW_DIFFERENT_CONDA_ENV:-0}"
CREATE_CONDA_ENV="${CREATE_CONDA_ENV:-1}"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.12}"
CONDA_SH="${CONDA_SH:-/vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh}"
INSTALL_UV="${INSTALL_UV:-1}"
INSTALL_ROCM_PYTORCH="${INSTALL_ROCM_PYTORCH:-1}"
PYTORCH_ROCM_INDEX_URL="${PYTORCH_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.9.1+rocm6.3}"
INSTALL_RELAX_REQUIREMENTS="${INSTALL_RELAX_REQUIREMENTS:-1}"
INSTALL_RELAX_EDITABLE="${INSTALL_RELAX_EDITABLE:-1}"
CONDA_UV_CHANNEL="${CONDA_UV_CHANNEL:-conda-forge}"

if ! command -v git >/dev/null 2>&1; then
    echo "error: git is required to clone ROCm dependency repositories" >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
UV_BIN=""

load_conda() {
    local conda_base

    if declare -F conda >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f "${CONDA_SH}" ]]; then
        # shellcheck source=/dev/null
        source "${CONDA_SH}"
        if declare -F conda >/dev/null 2>&1; then
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
        if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "${conda_base}/etc/profile.d/conda.sh"
            if declare -F conda >/dev/null 2>&1; then
                return 0
            fi
        fi
    fi

    echo "error: could not load conda. Set CONDA_SH=/path/to/conda.sh" >&2
    exit 1
}

conda_env_exists() {
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx -- "${CONDA_ENV_NAME}" >/dev/null
}

ensure_conda_env() {
    load_conda

    if [[ "${CREATE_CONDA_ENV}" == "1" ]]; then
        EXPECTED_CONDA_ENV="${CONDA_ENV_NAME}"
        if ! conda_env_exists; then
            echo "Creating conda env '${CONDA_ENV_NAME}' with Python ${CONDA_PYTHON_VERSION}"
            conda create -y -n "${CONDA_ENV_NAME}" "python=${CONDA_PYTHON_VERSION}"
        fi
        conda activate "${CONDA_ENV_NAME}"
    elif [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "error: activate a conda env first, or keep CREATE_CONDA_ENV=1" >&2
        exit 1
    fi

    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT

    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "${EXPECTED_CONDA_ENV}" && "${ALLOW_DIFFERENT_CONDA_ENV}" != "1" ]]; then
        echo "error: active conda env is ${CONDA_DEFAULT_ENV}; expected ${EXPECTED_CONDA_ENV}" >&2
        echo "set ALLOW_DIFFERENT_CONDA_ENV=1 only when intentionally using another ROCm env" >&2
        exit 1
    fi
}

ensure_python_bin() {
    PYTHON_BIN="${PYTHON_BIN:-python}"
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "error: PYTHON_BIN=${PYTHON_BIN} is not executable" >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return 0
    fi

    if [[ "${INSTALL_UV}" != "1" ]]; then
        echo "error: uv is required for dependency installation" >&2
        exit 1
    fi

    conda install -y -n "${CONDA_DEFAULT_ENV}" -c "${CONDA_UV_CHANNEL}" uv
    hash -r

    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "${CONDA_PREFIX}/bin/uv" ]]; then
        UV_BIN="${CONDA_PREFIX}/bin/uv"
    else
        echo "error: uv installation completed but no uv executable was found" >&2
        exit 1
    fi
}

rocm_torch_available() {
    "${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception:
    sys.exit(1)

sys.exit(0 if getattr(torch.version, "hip", None) else 1)
PY
}

install_rocm_pytorch_if_needed() {
    local torch_packages=()

    if rocm_torch_available; then
        return 0
    fi

    if [[ "${INSTALL_ROCM_PYTORCH}" != "1" ]]; then
        echo "error: ROCm PyTorch is not available in this conda env" >&2
        echo "set INSTALL_ROCM_PYTORCH=1 or install ROCm PyTorch manually first" >&2
        exit 1
    fi

    read -r -a torch_packages <<<"${TORCH_PACKAGES}"
    echo "Installing ROCm PyTorch from ${PYTORCH_ROCM_INDEX_URL}: ${TORCH_PACKAGES}"
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" --index-url "${PYTORCH_ROCM_INDEX_URL}" "${torch_packages[@]}"
}

install_relax_if_requested() {
    if [[ "${INSTALL_RELAX_REQUIREMENTS}" == "1" ]]; then
        "${UV_BIN}" pip install --python "${PYTHON_BIN}" -r "${REPO_ROOT}/requirements.txt"
    fi

    if [[ "${INSTALL_RELAX_EDITABLE}" == "1" ]]; then
        "${UV_BIN}" pip install --python "${PYTHON_BIN}" -e "${REPO_ROOT}"
    fi
}

ensure_conda_env
ensure_python_bin
ensure_uv
install_rocm_pytorch_if_needed
install_relax_if_requested

"${PYTHON_BIN}" - <<'PY'
import os

import torch

if torch.version.hip is None:
    raise SystemExit("error: this installer must run inside a ROCm PyTorch environment")

print(f"Detected ROCm PyTorch: torch={torch.__version__}, hip={torch.version.hip}")
print(f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV', '<not set>')}")
print(f"CONDA_PREFIX={os.environ.get('CONDA_PREFIX', '<not set>')}")
print(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '<not set>')}")
PY

ROCM_DEPS_SRC_DIR="${ROCM_DEPS_SRC_DIR:-${REPO_ROOT}/.deps/rocm-megatron}"
ROCM_TRANSFORMER_ENGINE_REPO="${ROCM_TRANSFORMER_ENGINE_REPO:-https://github.com/ROCm/TransformerEngine.git}"
ROCM_TRANSFORMER_ENGINE_REF="${ROCM_TRANSFORMER_ENGINE_REF:-v2.10_rocm}"
ROCM_APEX_REPO="${ROCM_APEX_REPO:-https://github.com/ROCm/apex.git}"
ROCM_APEX_REF="${ROCM_APEX_REF:-release/1.9.0}"
RECREATE_FAILED_CHECKOUT="${RECREATE_FAILED_CHECKOUT:-1}"
INSTALL_BUILD_TOOLS="${INSTALL_BUILD_TOOLS:-1}"
INSTALL_TRANSFORMER_ENGINE="${INSTALL_TRANSFORMER_ENGINE:-1}"
INSTALL_APEX="${INSTALL_APEX:-1}"
VALIDATE_TRANSFORMER_ENGINE="${VALIDATE_TRANSFORMER_ENGINE:-${INSTALL_TRANSFORMER_ENGINE}}"
VALIDATE_APEX="${VALIDATE_APEX:-${INSTALL_APEX}}"
ROCM_PRELOAD_HSA="${ROCM_PRELOAD_HSA:-1}"
ROCM_RUNTIME_ENV_SH="${ROCM_RUNTIME_ENV_SH:-${REPO_ROOT}/scripts/setup/rocm_runtime_env.sh}"
NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
MAX_JOBS="${MAX_JOBS:-4}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
APEX_PREBUILD_OPS="${APEX_PREBUILD_OPS:-amp_c,fused_adam,fused_layer_norm}"

detect_rocm_path() {
    local hipcc_path
    local candidate
    local candidates=()

    if [[ -n "${ROCM_PATH:-}" ]]; then
        candidates+=("${ROCM_PATH}")
    fi
    if [[ -n "${ROCM_HOME:-}" ]]; then
        candidates+=("${ROCM_HOME}")
    fi
    if command -v hipcc >/dev/null 2>&1; then
        hipcc_path="$(readlink -f "$(command -v hipcc)")"
        candidates+=("$(dirname "$(dirname "${hipcc_path}")")")
    fi
    candidates+=("/opt/rocm" "/opt/rocm-7.0.0" "/opt/rocm-6.3.0")

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/lib/cmake/hip/hip-config.cmake" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "error: could not find ROCm CMake packages; set ROCM_PATH to the ROCm install prefix" >&2
    return 1
}

detect_nvte_rocm_arch() {
    "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("error: set NVTE_ROCM_ARCH because ROCm PyTorch cannot see a GPU")

archs: list[str] = []
for device_idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(device_idx)
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0].strip()
    if not arch:
        raise SystemExit(f"error: could not detect ROCm arch for GPU {device_idx}")
    if arch not in archs:
        archs.append(arch)

print(",".join(archs))
PY
}

nvte_arch_has_aiter_v3_target() {
    case ",${NVTE_ROCM_ARCH}," in
        *,gfx942,* | *,gfx950,*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

checkout_repo_once() {
    local repo_url="$1"
    local repo_ref="$2"
    local checkout_dir="$3"
    local recurse_submodules="$4"

    mkdir -p "$(dirname "${checkout_dir}")"
    if [[ -d "${checkout_dir}/.git" ]]; then
        git -C "${checkout_dir}" fetch --tags origin
    else
        git clone "${repo_url}" "${checkout_dir}"
    fi

    git -C "${checkout_dir}" checkout "${repo_ref}"
    if [[ "${recurse_submodules}" == "1" ]]; then
        git -C "${checkout_dir}" submodule update --init --recursive
    fi
}

checkout_repo() {
    local repo_url="$1"
    local repo_ref="$2"
    local checkout_dir="$3"
    local recurse_submodules="$4"
    local failed_checkout

    if checkout_repo_once "${repo_url}" "${repo_ref}" "${checkout_dir}" "${recurse_submodules}"; then
        return 0
    fi

    if [[ "${RECREATE_FAILED_CHECKOUT}" != "1" ]]; then
        return 1
    fi

    failed_checkout="${checkout_dir}.failed-$(date +%Y%m%d-%H%M%S)"
    echo "warning: checkout failed; moving ${checkout_dir} to ${failed_checkout} and retrying once" >&2
    mv "${checkout_dir}" "${failed_checkout}"
    checkout_repo_once "${repo_url}" "${repo_ref}" "${checkout_dir}" "${recurse_submodules}"
}

configure_apex_prebuild_ops() {
    local op
    IFS=',' read -ra ops <<<"${APEX_PREBUILD_OPS}"
    for op in "${ops[@]}"; do
        op="${op//[[:space:]]/}"
        case "${op}" in
            "")
                ;;
            amp_c)
                export APEX_BUILD_AMP_C=1
                ;;
            apex_c)
                export APEX_BUILD_APEX_C=1
                ;;
            fused_adam)
                export APEX_BUILD_FUSED_ADAM=1
                ;;
            fused_layer_norm)
                export APEX_BUILD_FUSED_LAYER_NORM=1
                ;;
            fused_dense)
                export APEX_BUILD_FUSED_DENSE=1
                ;;
            fused_rope)
                export APEX_BUILD_FUSED_ROPE=1
                ;;
            cpp_ops)
                export APEX_BUILD_CPP_OPS=1
                ;;
            cuda_ops)
                export APEX_BUILD_CUDA_OPS=1
                ;;
            *)
                echo "error: unsupported APEX_PREBUILD_OPS entry: ${op}" >&2
                exit 1
                ;;
        esac
    done
}

unset_apex_zero_flag() {
    local flag_name="$1"
    if [[ "${!flag_name:-}" == "0" ]]; then
        unset "${flag_name}"
    fi
}

move_stale_cmake_build_dir() {
    local build_dir="$1"
    local expected_source_dir="$2"
    local cache_file="${build_dir}/CMakeCache.txt"
    local cached_source_dir=""
    local stale_dir

    if [[ ! -f "${cache_file}" ]]; then
        return 0
    fi

    cached_source_dir="$(awk -F= '/^CMAKE_HOME_DIRECTORY:INTERNAL=/ {print $2; exit}' "${cache_file}")"
    if [[ -z "${cached_source_dir}" || "${cached_source_dir}" == "${expected_source_dir}" ]]; then
        return 0
    fi

    stale_dir="${build_dir}.stale-$(date +%Y%m%d-%H%M%S)"
    echo "warning: moving stale CMake build dir ${build_dir} to ${stale_dir}" >&2
    echo "warning: cached source dir was ${cached_source_dir}; expected ${expected_source_dir}" >&2
    mv "${build_dir}" "${stale_dir}"
}

configure_rocm_runtime_env() {
    if [[ ! -f "${ROCM_RUNTIME_ENV_SH}" ]]; then
        echo "error: ROCM_RUNTIME_ENV_SH=${ROCM_RUNTIME_ENV_SH} does not exist" >&2
        exit 1
    fi

    # shellcheck source=/dev/null
    source "${ROCM_RUNTIME_ENV_SH}"
}

ROCM_PATH="$(detect_rocm_path)"
export ROCM_PATH

if [[ -z "${NVTE_ROCM_ARCH:-}" ]]; then
    NVTE_ROCM_ARCH="$(detect_nvte_rocm_arch)"
fi
export NVTE_ROCM_ARCH
echo "NVTE_ROCM_ARCH=${NVTE_ROCM_ARCH}"

if ! nvte_arch_has_aiter_v3_target; then
    if [[ -n "${NVTE_FUSED_ATTN_CK:-}" && "${NVTE_FUSED_ATTN_CK}" != "0" ]]; then
        echo "error: ROCm TransformerEngine CK/AITER v3 build supports gfx942/gfx950, not NVTE_ROCM_ARCH=${NVTE_ROCM_ARCH}" >&2
        echo "set NVTE_FUSED_ATTN_CK=0 or build on a supported CK/AITER v3 target" >&2
        exit 1
    fi
    NVTE_FUSED_ATTN_CK=0
fi
export NVTE_FUSED_ATTN_CK="${NVTE_FUSED_ATTN_CK:-1}"
echo "NVTE_FUSED_ATTN_CK=${NVTE_FUSED_ATTN_CK}"

configure_rocm_runtime_env
export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_BUILD_PARALLEL_LEVEL
echo "ROCM_PATH=${ROCM_PATH}"
echo "ROCM_HOME=${ROCM_HOME}"
echo "HIP_PATH=${HIP_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
if [[ "${ROCM_PRELOAD_HSA}" == "1" ]]; then
    echo "LD_PRELOAD=${LD_PRELOAD}"
fi
echo "ROCM_RUNTIME_ENV_SH=${ROCM_RUNTIME_ENV_SH}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
echo "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"

if [[ "${INSTALL_BUILD_TOOLS}" == "1" ]]; then
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" --upgrade pip setuptools wheel packaging cmake ninja pybind11
fi

if [[ "${INSTALL_TRANSFORMER_ENGINE}" == "1" ]]; then
    TE_DIR="${ROCM_DEPS_SRC_DIR}/TransformerEngine"
    checkout_repo "${ROCM_TRANSFORMER_ENGINE_REPO}" "${ROCM_TRANSFORMER_ENGINE_REF}" "${TE_DIR}" "1"

    export NVTE_FRAMEWORK
    export NVTE_BUILD_THREADS_PER_JOB
    export MAX_JOBS

    te_build_arch_key="${NVTE_ROCM_ARCH//,/_}"
    te_build_arch_key="${te_build_arch_key//:/_}"
    te_build_arch_key="${te_build_arch_key//;/_}"
    export NVTE_CMAKE_BUILD_DIR="${NVTE_CMAKE_BUILD_DIR:-${TE_DIR}/build/cmake-${te_build_arch_key}-ck${NVTE_FUSED_ATTN_CK}}"
    echo "NVTE_CMAKE_BUILD_DIR=${NVTE_CMAKE_BUILD_DIR}"
    move_stale_cmake_build_dir "${NVTE_CMAKE_BUILD_DIR}" "${TE_DIR}/transformer_engine/common"

    "${UV_BIN}" pip install --python "${PYTHON_BIN}" -v --no-build-isolation "${TE_DIR}"
fi

if [[ "${INSTALL_APEX}" == "1" ]]; then
    APEX_DIR="${ROCM_DEPS_SRC_DIR}/apex"
    checkout_repo "${ROCM_APEX_REPO}" "${ROCM_APEX_REF}" "${APEX_DIR}" "0"

    unset_apex_zero_flag APEX_BUILD_CPP_OPS
    unset_apex_zero_flag APEX_BUILD_CUDA_OPS
    configure_apex_prebuild_ops
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" -v --no-build-isolation "${APEX_DIR}"
fi

if [[ "${VALIDATE_TRANSFORMER_ENGINE}" == "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import torch
import transformer_engine.pytorch as te

print(f"ROCm PyTorch import OK: torch={torch.__version__}, hip={torch.version.hip}")
print(f"TransformerEngine import OK after torch import: {te.__name__}")
PY
fi

if [[ "${VALIDATE_APEX}" == "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import apex
from apex.normalization import FusedLayerNorm
from apex.optimizers import FusedAdam

print(f"Apex import OK: {apex.__file__}")
print(f"Apex FusedAdam import OK: {FusedAdam.__name__}")
print(f"Apex FusedLayerNorm import OK: {FusedLayerNorm.__name__}")
PY
fi
