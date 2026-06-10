---
name: qwen3-0-6b-rocm-fully-async-e2e
description: >
  Validate the Qwen3-0.6B Relax ROCm fully_async e2e path with actor,
  rollout, actor_fwd, offline W&B, SGLang's built local `sgl_kernel`, and
  optimizer-inclusive torch_dist checkpoints. Use when: checking bounded
  staleness fully_async on MI210 after the sync Qwen3-0.6B path works.
metadata:
  short-description: "Qwen3-0.6B ROCm fully_async e2e"
  tags:
    - rocm
    - qwen3
    - fully-async
    - actor-fwd
    - sglang
    - checkpoint
    - wandb-offline
    - relax
  domain: research
  created: 2026-06-10
  author: Codex
---

# Qwen3 0.6B ROCm Fully Async E2E

## General Description

This skill captures the validated Qwen3-0.6B fully_async e2e path on four
MI210 GPUs. It proves that the clean `after_fix` stack can run separate actor,
rollout, and actor_fwd services, perform async weight updates through DCS,
compute actor_fwd log probabilities, train the actor, and save
optimizer-inclusive `torch_dist` checkpoints with W&B offline.

This is distinct from the two-GPU sync smoke. Fully_async needs a different
resource graph and a longer startup window: actor uses two GPUs, rollout uses
one GPU, actor_fwd uses one GPU, and the first five minutes may only prove
startup health.

## When to Apply

Use this knowledge when:
- The user asks whether fully_async e2e works on `after_fix`.
- The sync Qwen3-0.6B ROCm path already works and the next boundary is
  actor/rollout/actor_fwd coordination.
- You need a small fully_async validation with real rollout, async weight
  update, actor_fwd log-prob computation, actor training, W&B application
  metrics, and optimizer checkpoint save.
- The run must keep W&B offline and use the built local SGLang `sgl_kernel`
  artifact.

Do NOT use when:
- The target is pure true-on-policy fully_async without actor_fwd. That path
  has different invariants and requires `MAX_STALENESS=0`.
- The goal is the full Qwen3-4B TP2 path. Use
  `rocm-megatron-tp2-checkpoint-resume`.
- The goal is only the two-GPU sync/import smoke. Use
  `qwen3-0-6b-rocm-sgl-kernel-e2e`.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Validated date | 2026-06-10 | Clean detached `after_fix` worktree |
| Conda env | `relaxrl_rocm_after_fix` | TE/Apex absent and non-fatal |
| Foreground gate | Ray job `raysubmit_SZFD4qZvxFU8q2v2`, exit `124` | Timed out at 300s after healthy startup; not an e2e failure |
| Full tmux run | `tmux-3` | Left intact for inspection |
| Successful Ray job | `raysubmit_cvLQL4xbyhdB9L4d` | Completed successfully |
| Log | `log/amd-qwen3-0.6b-fully-async-4gpu-fully-async-20260610_115729.log` | Contains fully_async startup, actor_fwd, train, save, and Ray success markers |
| Save dir | `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_4gpu-fully-async-20260610_115729` | `latest_checkpointed_iteration.txt` contains `1` |
| Resource graph | `{"actor": [1, 2], "rollout": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}` | Actor GPUs 0-1, rollout GPU 2, actor_fwd GPU 3 |
| Fully_async settings | `--fully-async`, `MAX_STALENESS=1`, `NUM_STEPS_PER_ROLLOUT=2` | Bounded staleness requires actor_fwd in this launcher |
| W&B mode | `offline` | Launcher `--wandb-mode online` was replaced at execution time |
| Checkpoint proof | Iterations `0` and `1` saved | Each iteration has `.metadata`, four `.distcp` shards, `common.pt`, and `metadata.json` |
| Optimizer proof | Distributed optimizer state saved | Metadata contains optimizer/exp_avg entries; log says `Storing distributed optimizer sharded state of type dp_reshardable` |

## Recommended Practice

### Step 1: Preflight the Environment

Use the same ROCm environment and verify the built SGLang kernel artifact:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix

SGL_KERNEL_BUILD=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
SGLANG_PYTHON=/vast/users/qirong.ho/erland/Python_project/sglang/python
PYTHONPATH="${SGL_KERNEL_BUILD}:${SGLANG_PYTHON}:${PYTHONPATH}" \
  python -c "import torch, sglang, sgl_kernel; print(torch.version.hip, torch.cuda.device_count())"
```

Healthy output should show ROCm/HIP, four visible GPUs, and successful
`sgl_kernel` import.

### Step 2: Keep the Four-GPU Fully Async Topology

The validated bounded-staleness topology is:

```bash
RELAX_EXECUTION_MODE=fully_async
HIP_VISIBLE_DEVICES=0,1,2,3
RAY_NUM_GPUS=4
NUM_GPUS_PER_NODE=4
ACTOR_RESOURCE_GPUS=2
ROLLOUT_RESOURCE_GPUS=1
ACTOR_FWD_RESOURCE_GPUS=1
TENSOR_MODEL_PARALLEL_SIZE=1
ENABLE_SEQUENCE_PARALLEL=0
MAX_STALENESS=1
```

Do not remove `actor_fwd` while keeping `MAX_STALENESS=1`. Pure fully_async
without actor_fwd is true-on-policy and must use `MAX_STALENESS=0`.

### Step 3: Use a Short Fully Async Regression Shape

The validated short run used:

```bash
NUM_ROLLOUT=2
NUM_STEPS_PER_ROLLOUT=2
ROLLOUT_BATCH_SIZE=2
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
MICRO_BATCH_SIZE=1
SAVE_INTERVAL=1
CKPT_FORMAT=torch_dist
NO_SAVE_OPTIM=0
SEQ_LENGTH=1024
ROLLOUT_MAX_RESPONSE_LEN=128
SGLANG_SERVER_CONCURRENCY=16
SGLANG_MAX_RUNNING_REQUESTS=128
SGLANG_MAX_TOTAL_TOKENS=65536
SGLANG_MEM_FRACTION_STATIC=0.2
```

This is intentionally small but still exercises the fully_async surfaces:
rollout, DCS weight update, actor_fwd logprobs, two actor train steps per
rollout, and checkpointing.

### Step 4: Launch With W&B Offline and Built sgl_kernel

Use process substitution to keep the repository unchanged while forcing W&B
offline and injecting the built `sgl_kernel` artifact before `sglang/python`:

```bash
SGL_KERNEL_BUILD=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
SGLANG_PYTHON=/vast/users/qirong.ho/erland/Python_project/sglang/python

WANDB_MODE=offline \
WANDB_ENTITY="" \
CONDA_ENV_NAME=relaxrl_rocm_after_fix \
MODEL_CONFIG_NAME=qwen3-0.6B \
MODEL_ASSET_NAME=Qwen3-0.6B \
MODEL_LOG_NAME=qwen3-0.6b-fully-async \
RELAX_EXECUTION_MODE=fully_async \
RELAX_HIP_VISIBLE_DEVICES_OVERRIDE=0,1,2,3 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
RAY_NUM_GPUS=4 \
NUM_GPUS_PER_NODE=4 \
ACTOR_RESOURCE_GPUS=2 \
ROLLOUT_RESOURCE_GPUS=1 \
ACTOR_FWD_RESOURCE_GPUS=1 \
MAX_STALENESS=1 \
NUM_ROLLOUT=2 \
NUM_STEPS_PER_ROLLOUT=2 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
SAVE_INTERVAL=1 \
CKPT_FORMAT=torch_dist \
NO_SAVE_OPTIM=0 \
SEQ_LENGTH=1024 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
SGLANG_SERVER_CONCURRENCY=16 \
SGLANG_MAX_RUNNING_REQUESTS=128 \
SGLANG_MAX_TOTAL_TOKENS=65536 \
SGLANG_MEM_FRACTION_STATIC=0.2 \
GPU_LABEL=4gpu-fully-async \
bash <(
  sed \
    -e 's/--wandb-mode online/--wandb-mode offline/g' \
    -e "s#${SGLANG_PYTHON}:#${SGL_KERNEL_BUILD}:${SGLANG_PYTHON}:#g" \
    scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
)
```

Run the foreground gate first. If it reaches healthy startup and exits with
`124` due to the 300-second timeout, restart the same command in tmux for the
actual e2e run.

### Step 5: Verify Fully Async Success Markers

The run is healthy only if it crosses all of these boundaries:

```text
Using parallel creation mode (fully_async=True)
Service actor_fwd has been created successfully
[engine-init-barrier:default] all 1 engines ready
Weights updated for rollout role.
actor_fwd model computed log prob for step 0/2
actor_fwd model computed log prob for step 1/2
train_one_step rollout=0 step=0: finished optimizer.step
train_one_step rollout=0 step=1: finished optimizer.step
train_one_step rollout=1 step=0: finished optimizer.step
train_one_step rollout=1 step=1: finished optimizer.step
Actor training completed step 0/2
Actor training completed step 1/2
All training steps finished
Job 'raysubmit_cvLQL4xbyhdB9L4d' succeeded
```

### Step 6: Audit Checkpoints

Check both checkpoint and dataset state:

```bash
SAVE_DIR=/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_4gpu-fully-async-20260610_115729
cat "$SAVE_DIR/latest_checkpointed_iteration.txt"
find "$SAVE_DIR" -maxdepth 2 -type f \
  \( -name '.metadata' -o -name 'metadata.json' -o -name 'common.pt' -o -name '*.distcp' -o -name 'global_dataset_state_dict_*.pt' \) |
  sort
```

For optimizer state, inspect the Megatron metadata with the ROCm Megatron path
available:

```bash
PYTHONPATH="/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM:$PWD:${PYTHONPATH}" \
python - <<'PY'
from pathlib import Path
import pickle

metadata = Path("/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_4gpu-fully-async-20260610_115729/iter_0000001/.metadata")
obj = pickle.loads(metadata.read_bytes())
for attr in ["state_dict_metadata", "storage_data"]:
    data = getattr(obj, attr)
    keys = [str(k) for k in data.keys()]
    print(attr, "optimizer", sum("optimizer" in k for k in keys), "exp_avg", sum("exp_avg" in k for k in keys))
PY
```

The validated run found optimizer and `exp_avg` entries in both metadata and
storage data.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Foreground validation exited `124` | Fully_async startup exceeded the five-minute foreground gate but had already reached healthy service/model startup | Treat timeout as a pass-to-tmux signal only after checking startup markers; do not call it e2e success |
| `actor_fwd` removed while `MAX_STALENESS=1` stayed enabled | Bounded staleness fully_async needs actor_fwd to compute logprobs for stale rollout data | Keep `ACTOR_FWD_RESOURCE_GPUS=1` for this recipe, or switch to pure true-on-policy invariants |
| W&B would log online by default | The base launcher still emits `--wandb-mode online` | Force `WANDB_MODE=offline` and replace launcher `--wandb-mode online` at execution time |
| `sgl_kernel` missing in Ray workers | Runtime `PYTHONPATH` included SGLang source but not the built kernel artifact | Prepend `sgl-kernel/build/lib.linux-x86_64-cpython-312` before `sglang/python` |
| Checkpoint looked weights-only from a simple string grep | Megatron checkpoint metadata requires the Megatron classes on `PYTHONPATH` to inspect correctly | Use the ROCm Megatron path when unpickling `.metadata`, and verify optimizer entries in `state_dict_metadata` / `storage_data` |

## Configuration

```yaml
environment: relaxrl_rocm_after_fix
hardware: 4x MI210 gfx90a
launcher: scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
mode: fully_async
resource:
  actor: [1, 2]
  rollout: [1, 1]
  actor_fwd: [1, 1]
  advantages: [1, 0]
runtime_paths:
  sgl_kernel_build: /vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
  sglang_python: /vast/users/qirong.ho/erland/Python_project/sglang/python
training:
  num_rollout: 2
  num_steps_per_rollout: 2
  rollout_batch_size: 2
  n_samples_per_prompt: 2
  global_batch_size: 2
  micro_batch_size: 1
  max_staleness: 1
checkpoint:
  ckpt_format: torch_dist
  save_interval: 1
  no_save_optim: false
  latest_checkpointed_iteration: 1
wandb:
  mode: offline
successful_run:
  foreground_ray_job: raysubmit_SZFD4qZvxFU8q2v2
  tmux_session: tmux-3
  ray_job: raysubmit_cvLQL4xbyhdB9L4d
  log: log/amd-qwen3-0.6b-fully-async-4gpu-fully-async-20260610_115729.log
  save_dir: /vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_4gpu-fully-async-20260610_115729
```

## References

- Related log: `references/experiment-log.md`
- Related troubleshooting: `references/troubleshooting.md`
- Related skills: `qwen3-0-6b-rocm-sgl-kernel-e2e`, `rocm-relax-bringup`, `rocm-megatron-tp2-checkpoint-resume`
