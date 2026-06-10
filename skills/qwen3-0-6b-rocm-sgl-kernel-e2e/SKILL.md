---
name: qwen3-0-6b-rocm-sgl-kernel-e2e
description: >
  Validate the Qwen3-0.6B 2-GPU ROCm Relax e2e path when SGLang requires a
  local built sgl_kernel rather than an installed sglang-kernel package.
  Use when: after_fix must prove SGLang rollout, Megatron actor training,
  TE-less/Apex-less fallback, offline W&B, and torch_dist checkpoints.
metadata:
  short-description: "Qwen3-0.6B ROCm e2e with local sgl_kernel"
  tags:
    - rocm
    - qwen3
    - sglang
    - sgl-kernel
    - megatron
    - e2e
    - wandb
  domain: research
  created: 2026-06-10
  author: Codex
---

# Qwen3-0.6B ROCm SGL Kernel E2E

## General Description

This skill captures the validated Qwen3-0.6B two-GPU Relax e2e smoke path on
ROCm using a local built SGLang kernel artifact. The important lesson is that
`sgl_kernel` may exist in the SGLang checkout but still be invisible to Ray
workers unless the built library path is present in the job runtime
`PYTHONPATH`.

The run also validates the TE-less and Apex-less actor path in
`relaxrl_rocm_after_fix`: Megatron-Bridge can force the local layer spec, use
Torch/local fallbacks, complete rollout/training, update SGLang weights, and
write optimizer-inclusive `torch_dist` checkpoints.

## When to Apply

Use this knowledge when:
- SGLang startup fails with `ModuleNotFoundError: No module named 'sgl_kernel'`.
- `sglang` is installed but `sglang-kernel` is not installed in the conda env.
- The local SGLang checkout has `sgl-kernel/build/lib.linux-x86_64-cpython-312`.
- Testing the reverted or clean `after_fix` path in `relaxrl_rocm_after_fix`.
- Verifying that removing Transformer Engine and Apex still leaves a viable
  ROCm Megatron actor smoke path.

Do not use this as proof of model quality. The two-step smoke proves
infrastructure boundaries: SGLang startup, rollout, actor training,
optimizer step, weight sync, W&B offline logging, and checkpointing.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Successful job | `raysubmit_vQKenPVHTm4HiKL9` | Completed two rollout/training steps |
| Log | `log/amd-qwen3-0.6b-2gpu-20260610_111842.log` | W&B offline |
| Model asset | `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B` | HF-compatible Qwen3-0.6B checkpoint |
| Save dir | `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260610_111842` | Contains checkpoints and rollout dumps |
| SGLang kernel path | `/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312` | Must precede `sglang/python` |
| SGLang startup | `[engine-init-barrier:default] all 1 engines ready` | No `No module named 'sgl_kernel'` in the successful log |
| Actor startup | `Actor initialized with starting step 0` | TE/Apex absent but non-fatal |
| Training proof | `Actor training completed step 0/2` and `Actor training completed step 1/2` | Both steps finished optimizer step |
| Checkpoints | Iterations `0` and `1` | `torch_dist`; latest checkpoint file contains `1` |
| Dataset state | `global_dataset_state_dict_0.pt`, `global_dataset_state_dict_1.pt` | Present |
| W&B mode | `offline` | Logs stored under `relax_e2e_assets/wandb` |
| Final status | `Job 'raysubmit_vQKenPVHTm4HiKL9' succeeded` | Ray job completed successfully |

## Recommended Practice

### Step 1: Verify the built kernel path before launching

The pure source path can find `sgl_kernel` but fail to import the compiled
extension. Prefer the built library path:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix

PYTHONPATH=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312 \
python - <<'PY'
import sgl_kernel

print(sgl_kernel.__file__)
print(hasattr(sgl_kernel, "moe_align_block_size"))
PY
```

Healthy output should point into `sgl-kernel/build/lib.linux-x86_64-cpython-312`
and report `True` for `moe_align_block_size`.

### Step 2: Launch with the kernel path in Ray runtime env

The launcher builds `PYTHONPATH` internally, so either install
`sglang-kernel` into the environment or make sure the launched script prepends
the built path before `/vast/users/qirong.ho/erland/Python_project/sglang/python`.
For a no-code-change smoke, patch the script at shell-launch time:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix
unset ROCR_VISIBLE_DEVICES

export WANDB_MODE=offline
export WANDB_ENTITY=""
export RELAX_ROOT_DIR="$PWD"
export CONDA_ENV_NAME=relaxrl_rocm_after_fix
export MODEL_CONFIG_NAME=qwen3-0.6B
export MODEL_ASSET_NAME=Qwen3-0.6B
export MODEL_LOG_NAME=qwen3-0.6b
export NUM_ROLLOUT=2
export SAVE_INTERVAL=1
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_MAX_RESPONSE_LEN=128
export SGLANG_SERVER_CONCURRENCY=16

bash <(
  sed \
    -e "s#--wandb-mode online#--wandb-mode offline#g" \
    -e "s#/vast/users/qirong.ho/erland/Python_project/sglang/python:#/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312:/vast/users/qirong.ho/erland/Python_project/sglang/python:#g" \
    scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
)
```

Keep `RELAX_ROOT_DIR="$PWD"` when using process substitution. Otherwise the
script can infer its root from `/dev/fd`.

### Step 3: Check the right success markers

For a two-rollout smoke, require all of these:

```text
wandb_mode ...................................... offline
wandb: W&B syncing is set to `offline`
[engine-init-barrier:default] all 1 engines ready
Transformer Engine unavailable; force Megatron-Bridge provider to use local layer spec
Actor initialized with starting step 0
All 2 services registered successfully
Actor training completed step 0/2
Actor training completed step 1/2
successfully saved checkpoint from iteration       0
successfully saved checkpoint from iteration       1
All training steps finished
Job '<job-id>' succeeded
```

Also verify the checkpoint boundary:

```bash
SAVE_DIR=/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260610_111842
cat "$SAVE_DIR/latest_checkpointed_iteration.txt"
test -f "$SAVE_DIR/dataset/global_dataset_state_dict_0.pt"
test -f "$SAVE_DIR/dataset/global_dataset_state_dict_1.pt"
test -f "$SAVE_DIR/iter_0000000/.metadata"
test -f "$SAVE_DIR/iter_0000001/.metadata"
```

### Step 4: Use local Ray CLI safely

If querying the local Ray dashboard from a shell with proxy variables, force
localhost and unset proxy variables:

```bash
env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  ray job status --address=http://127.0.0.1:8265 <job-id>
```

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| `No module named 'sgl_kernel'` | Built `sgl-kernel` path was not visible to Ray workers | Add `sgl-kernel/build/lib...` before `sglang/python` |
| Import from `sgl-kernel/python` only | Source path lacks `common_ops` compiled extension | Use the build output path or install `sglang-kernel` |
| `WANDB_ENTITY: unbound variable` | Launcher uses `set -u` and the variable was unset | Export `WANDB_ENTITY=""` |
| W&B online run | Reverted launcher hardcodes `--wandb-mode online` | Replace with `--wandb-mode offline` at launch or edit intentionally |
| Ray CLI hit Squid proxy | Status command used node IP through proxy | Use `http://127.0.0.1:8265` and unset proxy variables |
| `failed to import relax.models, error=No module named 'transformer_engine'` | TE intentionally uninstalled | Non-fatal when Megatron-Bridge forces local layer spec |
| Missing `gguf`, `aiter`, `amdsmi`, or `torchaudio` warnings | Optional SGLang/ROCm dependencies are absent | Non-fatal for this Qwen3-0.6B transformers smoke |

## Configuration

```yaml
environment: relaxrl_rocm_after_fix
hardware: 2x MI210
model:
  config: qwen3-0.6B
  asset: /vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B
runtime:
  sglang_kernel_path: /vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
  sglang_source_path: /vast/users/qirong.ho/erland/Python_project/sglang/python
  megatron_dir: /vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM
  wandb_mode: offline
training:
  num_rollout: 2
  save_interval: 1
  n_samples_per_prompt: 1
  rollout_max_response_len: 128
  sglang_server_concurrency: 16
checkpoint:
  format: torch_dist
  latest_validated_iteration: 1
```
