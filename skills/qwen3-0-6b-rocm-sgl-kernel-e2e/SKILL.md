---
name: qwen3-0-6b-rocm-sgl-kernel-e2e
description: >
  Validate the command-only Qwen3-0.6B Relax ROCm e2e path with SGLang's built
  local `sgl_kernel` artifact and offline W&B. Use when: checking that a clean
  after_fix checkout can run the two-GPU Qwen3-0.6B stack without code edits,
  or when SGLang imports fail because `sgl_kernel` is present locally but not
  importable in the Ray runtime.
metadata:
  short-description: "Qwen3-0.6B ROCm e2e with local sgl_kernel"
  tags:
    - rocm
    - qwen3
    - sglang
    - sgl-kernel
    - e2e
    - wandb-offline
    - relax
  domain: research
  created: 2026-06-10
  author: Codex
---

# Qwen3 0.6B ROCm SGL Kernel E2E

## General Description

This skill captures the clean `after_fix` Qwen3-0.6B ROCm e2e validation path
that succeeded without code changes after the runtime `PYTHONPATH` was pointed
at SGLang's built local `sgl_kernel` artifact. It is meant for quick
two-GPU infrastructure validation of Relax, SGLang, Megatron-Bridge, offline
W&B, and optimizer-inclusive `torch_dist` checkpointing.

The important lesson is that the local SGLang checkout can be present and still
fail at runtime if only the SGLang Python source directory is exported. On this
machine, the working import path must include the built kernel directory:

```text
/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
```

Do not install CUDA-only packages such as `cuda-python` to solve this on ROCm.

## When to Apply

Use this knowledge when:
- The user asks for a fast Qwen3-0.6B Relax e2e run on the `after_fix` ROCm
  branch.
- A run fails with `No module named 'sgl_kernel'`, even though the SGLang source
  checkout exists under `/vast/users/qirong.ho/erland/Python_project/sglang`.
- The validation must avoid new code edits and must keep W&B offline.
- You need to prove real rollout, actor training, optimizer step, and
  `torch_dist` checkpoint save on two GPUs.

Do NOT use when:
- The goal is full Qwen3-4B TP2 validation. Use
  `rocm-megatron-tp2-checkpoint-resume` for that path.
- The goal is the random 0.5B mock checkpoint. Use
  `qwen3-mock-0-5b-rocm-e2e` for that path.
- The proposed fix is to install NVIDIA CUDA runtime packages in the ROCm
  environment.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Validated date | 2026-06-10 | Clean detached `after_fix` worktree |
| Conda env | `relaxrl_rocm_after_fix` | TE/Apex uninstalled; warnings are non-fatal |
| Ray job | `raysubmit_vQKenPVHTm4HiKL9` | Completed successfully |
| Log | `log/amd-qwen3-0.6b-2gpu-20260610_111842.log` | Contains rollout, train, save, and Ray success markers |
| Save dir | `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260610_111842` | `latest_checkpointed_iteration.txt` contains `1` |
| W&B mode | `offline` | Launch changed `--wandb-mode online` to `--wandb-mode offline`; do not log online |
| SGLang source path | `/vast/users/qirong.ho/erland/Python_project/sglang/python` | Still needed |
| Required kernel path | `/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312` | Must precede the SGLang source path in `PYTHONPATH` |
| Checkpoint proof | Iterations `0` and `1` saved | `.metadata` files and dataset states were present |

## Recommended Practice

### Step 1: Start From a Clean after_fix Worktree

Do not carry debug edits into this validation. Use the clean `after_fix`
checkout and confirm the branch/worktree state before launch:

```bash
git status --short --branch
```

If the checkout is detached, that is acceptable for validation, but report it
when committing or preserving results.

### Step 2: Activate the after_fix ROCm Environment

Use the conda environment that was validated for this path:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix
```

The TE/Apex-free state is expected. These warnings are not by themselves a
failure:

```text
No module named 'transformer_engine'
Transformer Engine unavailable; force Megatron-Bridge provider to use local layer spec
```

### Step 3: Verify SGLang and Built sgl_kernel Importability

Check both the SGLang package and the built `sgl_kernel` module from the same
environment the Ray job will use:

```bash
SGL_KERNEL_BUILD=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
SGLANG_PYTHON=/vast/users/qirong.ho/erland/Python_project/sglang/python
PYTHONPATH="${SGL_KERNEL_BUILD}:${SGLANG_PYTHON}:${PYTHONPATH}" \
  python -c "import sglang, sgl_kernel; print('sglang and sgl_kernel import ok')"
```

The source directory alone is not enough on this machine. The built artifact
contains the importable pieces that Ray workers need.

### Step 4: Launch Without Editing the Script

For a command-only validation, use process substitution to keep the repository
unchanged while forcing W&B offline and prepending the built kernel path:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix

SGL_KERNEL_BUILD=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
SGLANG_PYTHON=/vast/users/qirong.ho/erland/Python_project/sglang/python

WANDB_MODE=offline \
WANDB_ENTITY="" \
RELAX_ROOT_DIR="$PWD" \
CONDA_ENV_NAME=relaxrl_rocm_after_fix \
MODEL_CONFIG_NAME=qwen3-0.6B \
MODEL_ASSET_NAME=Qwen3-0.6B \
MODEL_LOG_NAME=qwen3-0.6b \
NUM_ROLLOUT=2 \
SAVE_INTERVAL=1 \
N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
SGLANG_SERVER_CONCURRENCY=16 \
bash <(
  sed \
    -e 's/--wandb-mode online/--wandb-mode offline/g' \
    -e "s#${SGLANG_PYTHON}:#${SGL_KERNEL_BUILD}:${SGLANG_PYTHON}:#g" \
    amd_qwen3_0_6b_2gpu_e2e.sh
)
```

If running in tmux, keep the same environment and command shape. Leave the
tmux pane intact after completion for inspection.

### Step 5: Verify the Success Signature

The run is healthy only when it reaches rollout, real actor training, optimizer
step, checkpoint save, and Ray job success. Expected markers include:

```text
[engine-init-barrier:default] all 1 engines ready
Actor initialized with starting step 0
All 2 services registered successfully
Start rollout 0/2
Finish rollout 1/2
train_one_step rollout=0 step=0: finished forward_backward
finished optimizer.step (update_successful=True
successfully saved checkpoint from iteration       0
successfully saved checkpoint from iteration       1
Actor training completed step 1/2
All training steps finished
Job 'raysubmit_vQKenPVHTm4HiKL9' succeeded
```

Also verify that the successful log does not contain:

```text
No module named 'sgl_kernel'
```

### Step 6: Audit the Checkpoint Boundary

The validated save directory had `latest_checkpointed_iteration.txt` set to
`1`, dataset state for iterations `0` and `1`, and `.metadata` under both
checkpoint iterations:

```bash
SAVE_DIR=/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260610_111842
cat "$SAVE_DIR/latest_checkpointed_iteration.txt"
test -f "$SAVE_DIR/dataset/global_dataset_state_dict_0.pt"
test -f "$SAVE_DIR/dataset/global_dataset_state_dict_1.pt"
test -f "$SAVE_DIR/iter_0000000/.metadata"
test -f "$SAVE_DIR/iter_0000001/.metadata"
```

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| `No module named 'sgl_kernel'` even though SGLang exists locally | The Ray runtime saw the SGLang source tree but not the built `sgl_kernel` artifact | Prepend `sgl-kernel/build/lib.linux-x86_64-cpython-312` before `sglang/python` in `PYTHONPATH` |
| Importing from the source tree alone failed on `common_ops` | The source path is not the importable runtime artifact for this build | Use the built library directory instead of adding only `sgl-kernel/python` |
| Debugging drifted into TE/Apex or `torch_optimizer` | The successful before/after path does not require TE, Apex, or CPU-offload optimizer packages | Keep the TE/Apex absence as expected and validate Megatron's local layer-spec fallback |
| W&B could log online by default | The launcher default can be `--wandb-mode online` | Force `WANDB_MODE=offline` and replace `--wandb-mode online` with `--wandb-mode offline` |
| CUDA package installation looked tempting | Some SGLang packaging errors mention CUDA-centric module names | Do not install `cuda-python` in the ROCm environment; fix the ROCm SGLang import path instead |

## Configuration

```yaml
environment: relaxrl_rocm_after_fix
hardware: 2x MI210 gfx90a
launcher: amd_qwen3_0_6b_2gpu_e2e.sh
validated_worktree: clean detached after_fix
model:
  config_name: qwen3-0.6B
  asset_name: Qwen3-0.6B
  asset_root: /vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B
runtime_paths:
  sgl_kernel_build: /vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
  sglang_python: /vast/users/qirong.ho/erland/Python_project/sglang/python
training:
  num_rollout: 2
  save_interval: 1
  n_samples_per_prompt: 1
  rollout_max_response_len: 128
  sglang_server_concurrency: 16
wandb:
  mode: offline
  entity: ""
checkpoint:
  latest_checkpointed_iteration: 1
  dataset_state_iterations:
    - 0
    - 1
successful_run:
  ray_job: raysubmit_vQKenPVHTm4HiKL9
  log: log/amd-qwen3-0.6b-2gpu-20260610_111842.log
  save_dir: /vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260610_111842
```

## References

- Related log: `references/experiment-log.md`
- Related troubleshooting: `references/troubleshooting.md`
- Related skills: `rocm-relax-bringup`, `rocm-megatron-tp2-checkpoint-resume`, `qwen3-mock-0-5b-rocm-e2e`
