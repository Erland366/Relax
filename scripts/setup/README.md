# ROCm Megatron Dependency Setup

`install_rocm_megatron_deps.sh` installs the ROCm-specific dependencies that
the Megatron backend expects on AMD machines:

- ROCm TransformerEngine from `https://github.com/ROCm/TransformerEngine`
- ROCm Apex from `https://github.com/ROCm/apex`

Run it from the Relax checkout. By default it creates or reuses a conda
environment named `relaxrl_rocm`, installs ROCm PyTorch, installs this Relax
checkout, and then builds the ROCm Megatron dependencies:

```bash
scripts/setup/install_rocm_megatron_deps.sh
```

The default conda shell hook is
`/vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh`. Set `CONDA_SH` if
conda lives somewhere else. The default environment can also be changed:

```bash
CONDA_ENV_NAME=my_relax_rocm scripts/setup/install_rocm_megatron_deps.sh
```

When `CREATE_CONDA_ENV=1` (the default), the script creates the conda env if it
does not already exist and activates it before installing anything. When
`CREATE_CONDA_ENV=0`, it expects the current conda env to match
`EXPECTED_CONDA_ENV` and fails otherwise. Set `ALLOW_DIFFERENT_CONDA_ENV=1` only
when intentionally installing into another ROCm Python environment.
After conda activation, the installer clears inherited `VIRTUAL_ENV` markers so
an old shell-activated `.venv` cannot make the setup look like it is using a
virtualenv.

The ROCm runtime contract is versioned in the repo instead of being hidden in
conda activation:

```bash
source scripts/setup/rocm_runtime_env.sh
```

The installer and AMD launchers source this file explicitly after conda
activation and `.env` loading. It sets the runtime settings needed by the ROCm
Megatron stack:
`ROCM_PATH`, `ROCM_HOME`, `HIP_PATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, and
`NVTE_FUSED_ATTN_CK`. `LD_PRELOAD` defaults to
`${ROCM_PATH}/lib/libhsa-runtime64.so.1` because importing PyTorch before
TransformerEngine can otherwise load PyTorch's bundled `libhsa-runtime64.so`
first. ROCm TransformerEngine then loads `/opt/rocm-7.0.0/lib/libamdhip64.so.7`
against that already-loaded HSA runtime and fails with a missing ROCR symbol.

The installer does not write conda activation hooks. This is intentional:
runtime behavior must stay visible in the repo and in launch scripts, not hidden
inside a mutable conda environment.

The script installs `uv` through conda if no `uv` executable exists. It then
uses `uv pip --python "$(command -v python)"` for Python package installation.
ROCm PyTorch is installed when missing using:

```bash
PYTORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
TORCH_PACKAGES=torch==2.9.1+rocm6.3
```

Set `INSTALL_ROCM_PYTORCH=0` only when the active environment already has a ROCm
PyTorch build. The script intentionally fails if PyTorch is still missing or if
the current PyTorch build is not a ROCm build (`torch.version.hip is None`).

By default, the installer also installs `requirements.txt` and this checkout as
an editable package. Disable those steps when only rebuilding the ROCm Megatron
dependencies:

```bash
INSTALL_RELAX_REQUIREMENTS=0 INSTALL_RELAX_EDITABLE=0 scripts/setup/install_rocm_megatron_deps.sh
```

The installer clones source repositories under `.deps/rocm-megatron`, installs
with `uv pip`, and validates these imports:

```python
import torch
import transformer_engine.pytorch
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
```

The TransformerEngine validation intentionally imports `torch` before
`transformer_engine.pytorch` so the setup catches ROCm runtime ordering issues
that only appear in the normal training import order.

Defaults are chosen for reproducibility:

- `CONDA_ENV_NAME=relaxrl_rocm`
- `CREATE_CONDA_ENV=1`
- `CONDA_PYTHON_VERSION=3.12`
- `CONDA_UV_CHANNEL=conda-forge`
- `INSTALL_ROCM_PYTORCH=1`
- `PYTORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.3`
- `TORCH_PACKAGES=torch==2.9.1+rocm6.3`
- `INSTALL_RELAX_REQUIREMENTS=1`
- `INSTALL_RELAX_EDITABLE=1`
- `ROCM_TRANSFORMER_ENGINE_REF=v2.10_rocm`
- `ROCM_APEX_REF=release/1.9.0`
- `APEX_PREBUILD_OPS=amp_c,fused_adam,fused_layer_norm`
- `ROCM_PRELOAD_HSA=1`
- `ROCM_RUNTIME_ENV_SH=scripts/setup/rocm_runtime_env.sh`
- `RECREATE_FAILED_CHECKOUT=1`
- `CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}` so nested CMake builds obey the same
  parallelism limit as the outer TE build
- `ROCM_PATH` is detected from installed ROCm CMake packages, such as
  `/opt/rocm-7.0.0/lib/cmake/hip/hip-config.cmake`
- `NVTE_ROCM_ARCH` is detected from the active ROCm PyTorch GPUs

If a generated source checkout under `.deps/rocm-megatron` is left half-written
by an interrupted clone or build, the installer moves it aside to a
`.failed-YYYYmmdd-HHMMSS` path and retries once. Set
`RECREATE_FAILED_CHECKOUT=0` to inspect and fix the checkout manually instead.
When reusing a TransformerEngine source checkout that has moved path, the
installer also moves aside a stale `NVTE_CMAKE_BUILD_DIR` if its `CMakeCache.txt`
points at the previous source path.

On MI210 (`gfx90a`), ROCm TransformerEngine must not use its default build arch
list because upstream defaults to `gfx942,gfx950`. The installer detects
`gfx90a` and exports `NVTE_ROCM_ARCH=gfx90a` before building TE. It also sets
`NVTE_FUSED_ATTN_CK=0` when the target arch list has no `gfx942` or `gfx950`
entry, because ROCm TE's AITER v3 CK fused-attention path only supports those
targets. When TE is built with `NVTE_FUSED_ATTN_CK=0`, keep the same environment
variable set at runtime; TE documents that a backend disabled at compilation
must also be disabled during runtime.

Use environment variables when testing a different upstream ref:

```bash
ROCM_TRANSFORMER_ENGINE_REF=dev scripts/setup/install_rocm_megatron_deps.sh
```

Install or validate one dependency at a time when needed:

```bash
INSTALL_APEX=0 VALIDATE_APEX=0 scripts/setup/install_rocm_megatron_deps.sh
INSTALL_TRANSFORMER_ENGINE=0 VALIDATE_TRANSFORMER_ENGINE=0 scripts/setup/install_rocm_megatron_deps.sh
```

ROCm Apex treats `APEX_BUILD_CUDA_OPS=0` and `APEX_BUILD_CPP_OPS=0` as set
environment variables, which makes its setup logic include all compatible
extensions. The installer unsets those flags when they are `0` and enables only
the requested entries from `APEX_PREBUILD_OPS`. `amp_c` is part of the default
set because `apex.optimizers.FusedAdam` imports `amp_C` through
`apex.multi_tensor_apply`; `fused_layer_norm` provides the
`fused_layer_norm_cuda` extension used by `apex.normalization.FusedLayerNorm`.

This setup replaces the previous Relax fork-only SGLang Megatron import
restriction. Rollout and SGLang processes no longer hide Megatron from
`PYTHONPATH` or set `RELAX_SGLANG_BLOCK_MEGATRON_IMPORTS`; if Megatron,
TransformerEngine, or Apex imports are broken, startup should fail with the real
dependency error.
