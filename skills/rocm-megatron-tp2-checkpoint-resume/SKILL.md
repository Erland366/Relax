---
name: rocm-megatron-tp2-checkpoint-resume
description: >
  Preserve and validate the current AMD Qwen3-4B Relax path that uses TP2 GPU
  optimizer state, W&B application metrics, and Megatron torch_dist save/resume
  with optimizer state enabled.
  Use when: the MI210 Relax run must prove logging, checkpointing, and resume
  without CPU optimizer offload or checkpoint shortcuts.
metadata:
  short-description: "TP2 GPU optimizer checkpoint/resume validation on ROCm"
  tags:
    - rocm
    - megatron
    - checkpoint
    - wandb
    - relax
  domain: research
  created: 2026-05-31
  author: Codex
---

# ROCm Megatron TP2 Checkpoint Resume

## General Description

This skill captures the current no-cheating AMD Qwen3-4B e2e path: use four
MI210 GPUs, give two GPUs to the Megatron actor, use tensor parallel size 2
with sequence parallel, keep optimizer state on GPU, and keep Megatron
`torch_dist` checkpointing with optimizer state enabled. It supersedes the
historical CPU optimizer offload workaround for this workload.

## When to Apply

Use this knowledge when:
- The user asks whether the Relax ROCm Megatron e2e run is really working.
- The run must prove W&B application metrics, checkpoint save, and checkpoint
  resume.
- A proposed fix tries to use CPU optimizer offload, disable checkpointing,
  disable optimizer save, or treat a partial run as final e2e success.

Do NOT use when:
- The run is intentionally testing a legacy CPU-offload optimizer path.
- The task is only SGLang startup or import isolation before actor training.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Current topology | 4 visible MI210 GPUs, actor 2 GPUs, rollout 2 GPUs | `HIP_VISIBLE_DEVICES=0,1,2,3`, `ACTOR_RESOURCE_GPUS=2`, `ROLLOUT_RESOURCE_GPUS=2` |
| Actor parallelism | `--tensor-model-parallel-size 2` plus `--sequence-parallel` | Required to fit GPU optimizer state without CPU offload |
| CPU optimizer offload | Disabled and forbidden | `optimizer_cpu_offload=False`; Relax raises if ROCm Megatron actor requests CPU offload |
| Checkpoint format | `CKPT_FORMAT=torch_dist` | Uses Relax ROCm checkpoint hook and Megatron DCP artifacts |
| Optimizer checkpointing | Enabled | `NO_SAVE_OPTIM=0`; do not add `--no-save-optim` |
| Save interval used for long run | `SAVE_INTERVAL=20` | Saves zero-based iterations 19, 39, 59, 79, 99, 119, 139, ... |
| First long-run proof | Reached step 100 and saved checkpoints through iteration 99 | W&B run `gult6g9a` |
| Resume proof | Loaded iteration 99, started at actor step 100, loaded dataset state, saved 119 and 139, and continued through completed step 154 | W&B run `6c3wrymd`; manual stop at step 155 |
| Current completion run | Resumed from latest iteration 139 with same settings | Ray job `raysubmit_npbaWGLVyyJMxhD8`; W&B run `bkzsnt9k` |
| Final e2e status | Not proven until final 200-step completion and final checkpoint audit | A partial healthy run is progress, not completion |

## Recommended Practice

### Step 1: Use the TP2 GPU optimizer path

Launch the AMD Qwen3-4B run with the actor spread across two GPUs:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3
RAY_NUM_GPUS=4
NUM_GPUS_PER_NODE=4
ACTOR_RESOURCE_GPUS=2
ROLLOUT_RESOURCE_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_LABEL=4gpu-tp2
```

Keep sequence parallel enabled whenever TP is greater than 1. The launcher does
this automatically unless `ENABLE_SEQUENCE_PARALLEL=0` is forced, which should
fail fast for TP2.

### Step 2: Reject CPU optimizer offload for this path

Do not fix this run by adding:

```text
--optimizer-cpu-offload
--optimizer-offload-fraction
--use-torch-optimizer-for-cpu-offload
--use-precision-aware-optimizer
--no-pin-cpu-grads
--no-pin-cpu-params
```

Those are historical debugging tools for a different path. For the current
AMD Qwen3-4B objective, CPU optimizer offload is a cheating solution and the
code should fail loudly if it is requested.

### Step 3: Keep checkpointing real

The required checkpoint settings are:

```bash
SAVE_INTERVAL=20
CKPT_FORMAT=torch_dist
NO_SAVE_OPTIM=0
```

A valid checkpoint directory must contain:

```text
.metadata
__0_0.distcp
__0_1.distcp
__1_0.distcp
__1_1.distcp
common.pt
metadata.json
latest_checkpointed_iteration.txt
dataset/global_dataset_state_dict_<iteration>.pt
```

For optimizer-state proof, inspect `.metadata` for keys such as:

```text
optimizer.state.exp_avg.*
optimizer.state.exp_avg_sq.*
optimizer.state.fp32_param.*
```

### Step 4: Resume explicitly with LOAD_DIR

Using an existing `SAVE_DIR` is not a resume. A real resume must pass
Megatron `--load`, which the AMD launcher does through `LOAD_DIR`:

```bash
SAVE_DIR=/path/to/checkpoint
LOAD_DIR=/path/to/checkpoint
SCHEDULER_RESUME_POLICY=strict
```

Proof of resume requires all of these signals:

```text
successfully loaded checkpoint ... at iteration <n>
Loaded streaming dataset state: epoch=<e>, position=<p>
Actor initialized with starting step <n + 1>
```

After resume, the run must train beyond the loaded iteration and save a newer
checkpoint. Loading iteration 99 and later saving 119 and 139 is a strong
resume proof.

### Step 5: Verify W&B application metrics, not only system charts

W&B is healthy only if the run contains application metrics such as:

```text
train/step
train/loss
train/entropy_loss
rollout/step
rollout/raw_reward
perf/step_time
perf/save_model_time
```

System metrics alone are not enough. If logs show `POST /metrics/log_metrics_batch`
but W&B is system-only, inspect `MetricsService` run joining and end-of-step
flush behavior before changing training.

### Step 6: Completion audit must match the full objective

Do not mark the e2e objective complete until the evidence covers every item:

- training reaches the configured final rollout count, currently `NUM_ROLLOUT=200`
- W&B contains application metrics through the final steps
- checkpointing remains enabled and saves the final expected boundary
- final checkpoint contains optimizer state and dataset state
- a fresh resume from the latest checkpoint starts at `latest + 1` and continues
  without loss spiking
- Ray/Serve/GPU state is clean after the run finishes or is intentionally stopped

For `SAVE_INTERVAL=20` with zero-based iterations, the expected final save
boundary before `NUM_ROLLOUT=200` is iteration 199.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Single-GPU actor OOMed on Adam state | Model weights fit but GPU optimizer moments did not | Use TP2 actor GPUs, not CPU optimizer offload |
| CPU optimizer offload looked attractive | It avoided the first OOM but repeatedly introduced ROCm HDO crashes and hidden state-placement bugs | Keep CPU offload rejected for this objective |
| Checkpoint was treated as optional | Disabling checkpointing or optimizer save would make the validation meaningless | Keep `torch_dist` and `NO_SAVE_OPTIM=0` |
| Resume restarted from step 0 | Only `SAVE_DIR` was set, so Megatron never received `--load` | Always set `LOAD_DIR` for resume |
| W&B showed only system metrics | MetricsService either split into its own run or buffered metrics until actor flush | Ensure MetricsService joins the primary run and reports namespaced `/step` metrics |
| Healthy partial run was treated as final | Manual stop after step 154 proved progress but not final 200-step completion | Keep the goal active until the final checkpoint and resume audit pass |

## Configuration

```yaml
environment: relaxrl_rocm
hardware: 4x MI210 gfx90a
launcher: amd_qwen3_4b_2gpu_e2e.sh
megatron_checkout: /vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM
topology:
  hip_visible_devices: "0,1,2,3"
  ray_num_gpus: 4
  actor_resource_gpus: 2
  rollout_resource_gpus: 2
  tensor_model_parallel_size: 2
  sequence_parallel: true
optimizer:
  optimizer_cpu_offload: false
  no_save_optim: false
checkpoint:
  ckpt_format: torch_dist
  save_interval: 20
  scheduler_resume_policy: strict
  require_load_dir_for_resume: true
wandb:
  project: relax-amd
  require_application_metrics: true
validated_resume:
  source_checkpoint_iteration: 99
  resumed_starting_step: 100
  latest_durable_checkpoint_seen: 139
  completed_training_step_seen: 154
  completion_run_pending: raysubmit_npbaWGLVyyJMxhD8
```

## References

- Experiment log: `references/experiment-log.md`
- Troubleshooting: `references/troubleshooting.md`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Related skills: `rocm-relax-bringup`, `ray-stale-live-state-triage`
