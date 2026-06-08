---
name: rocm-megatron-conda-runtime
description: >
  Validate and repair the explicit ROCm runtime contract for Megatron dependencies.
  Use when: ROCm TransformerEngine or Apex builds successfully, but normal Relax
  training imports fail after importing torch first.
metadata:
  short-description: "ROCm conda runtime and HSA preload validation"
  tags:
    - rocm
    - megatron
    - transformer-engine
    - apex
    - conda
  domain: research
  created: 2026-06-08
  author: Codex
---

# ROCm Megatron Runtime Contract

## General Description

This skill captures the explicit runtime validation needed after building ROCm
TransformerEngine and ROCm Apex for Relax. A successful build is not enough:
the launched process must also source the repo-owned ROCm runtime contract and
survive the normal training import order, where PyTorch loads before
TransformerEngine.

The critical failure mode is a ROCm runtime mismatch inside one Python process.
If PyTorch's bundled HSA runtime is loaded first, ROCm 7's HIP library can fail
when TransformerEngine imports later.

## When to Apply

Use this knowledge when:
- Creating a fresh ROCm Relax conda env.
- Rebuilding ROCm TransformerEngine or Apex.
- `import transformer_engine.pytorch` works alone, but fails after `import torch`.
- The error mentions `libamdhip64.so.7` and a missing `ROCR_1` symbol.

Do NOT use when:
- TransformerEngine or Apex has not built yet.
- The failure is inside Relax rollout, checkpointing, or training after all
  dependency imports already pass.
- The machine is intentionally using a different ROCm prefix and has its own
  validated runtime activation policy.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Validated env | `relaxrl_rocm_after_fix` | Fresh conda env, not cloned |
| PyTorch | `2.9.1+rocm6.3` | HIP `6.3.42134-a9a80e791` |
| ROCm runtime | `/opt/rocm-7.0.0` | Needed for TE/Apex build and runtime |
| TE arch | `gfx90a` | MI210 |
| TE CK fused attention | `NVTE_FUSED_ATTN_CK=0` | CK/AITER v3 is not valid for gfx90a |
| Apex ops | `amp_c,fused_adam,fused_layer_norm` | Required for Megatron optimizer/norm imports |
| Required runtime preload | `/opt/rocm-7.0.0/lib/libhsa-runtime64.so.1` | Keeps ROCm 7 HSA ahead of PyTorch's bundled HSA |
| Runtime contract | `scripts/setup/rocm_runtime_env.sh` | Source explicitly after conda activation and `.env` loading |

## Recommended Practice

### Step 1: Validate the Problem in the Normal Import Order

Always test `torch` before TransformerEngine, because that is the order normal
training processes can hit:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix
source scripts/setup/rocm_runtime_env.sh
python - <<'PY'
import torch
import transformer_engine.pytorch as te
print(torch.__version__, torch.version.hip, te.__name__)
PY
```

If that fails with the `libamdhip64.so.7` / `ROCR_1` symbol error, inspect the
loaded libraries:

```bash
LD_DEBUG=libs python - <<'PY' 2>&1 | rg 'libhsa-runtime64|libamdhip64'
import torch
import transformer_engine.pytorch
PY
```

A broken trace loads `site-packages/torch/lib/libhsa-runtime64.so` before
TransformerEngine reaches the ROCm 7 HIP library.

### Step 2: Pin the ROCm Runtime Before Python Starts

Use the versioned repo runtime contract:

```bash
source scripts/setup/rocm_runtime_env.sh
```

It sets ROCm's matching HSA runtime explicitly:

```bash
export ROCM_PATH=/opt/rocm-7.0.0
export ROCM_HOME=/opt/rocm-7.0.0
export HIP_PATH=/opt/rocm-7.0.0
export LD_LIBRARY_PATH=/opt/rocm-7.0.0/lib64:/opt/rocm-7.0.0/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/rocm-7.0.0/lib/libhsa-runtime64.so.1
export NVTE_FUSED_ATTN_CK=0
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
```

Then rerun the same torch-before-TE import test.

### Step 3: Pass the Runtime Into Ray

Do not rely on conda activation hooks as the normal path. Launchers should
source `scripts/setup/rocm_runtime_env.sh` after conda activation and pass these
variables into the Ray runtime env:

```text
ROCM_PATH
ROCM_HOME
HIP_PATH
ROCM_HSA_RUNTIME_LIB
LD_LIBRARY_PATH
LD_PRELOAD
NVTE_FUSED_ATTN_CK
```

For this Relax fork, `scripts/setup/install_rocm_megatron_deps.sh` sources the
same runtime contract for validation. Conda activation hooks are not part of the
supported setup path.

### Step 4: Validate Apex Fused Imports Too

The Megatron path needs Apex fused optimizer and norm imports:

```bash
python - <<'PY'
import torch
import transformer_engine.pytorch as te
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
print(torch.__version__, torch.version.hip, te.__name__)
print(FusedAdam.__name__, FusedLayerNorm.__name__)
PY
```

Treat this as the minimum import gate before running a Relax training smoke.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| TE imported alone but failed after torch | PyTorch loaded bundled HSA first | Validate `import torch` before TE |
| Build reused moved TE source tree | Stale CMake cache pointed at old path | Move stale `NVTE_CMAKE_BUILD_DIR` aside before rebuilding |
| Diagnostics showed an old `.venv` | Shell inherited stale `VIRTUAL_ENV` | Clear virtualenv markers after conda activation |
| Installer validation passed but user shell failed | The installer did not validate the training import order | Make validation import torch before TransformerEngine |
| Runtime fix was hidden in conda activation | The env worked only after mutable activation hooks were installed | Keep the runtime contract in `scripts/setup/rocm_runtime_env.sh` and source it explicitly |

## Configuration

```yaml
conda_env_name: relaxrl_rocm_after_fix
rocm_path: /opt/rocm-7.0.0
nvte_rocm_arch: gfx90a
nvte_fused_attn_ck: 0
apex_prebuild_ops: amp_c,fused_adam,fused_layer_norm
ld_preload: /opt/rocm-7.0.0/lib/libhsa-runtime64.so.1
runtime_env_script: scripts/setup/rocm_runtime_env.sh
```

## References

- Experiment log: `references/experiment-log.md`
- Troubleshooting: `references/troubleshooting.md`
- Runtime contract: `scripts/setup/rocm_runtime_env.sh`
- Setup script: `scripts/setup/install_rocm_megatron_deps.sh`
- Related skills: `rocm-relax-bringup`, `rocm-megatron-tp2-checkpoint-resume`
