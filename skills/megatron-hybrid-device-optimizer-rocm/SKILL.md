---
name: megatron-hybrid-device-optimizer-rocm
description: >
  Debug, patch, and validate Megatron's HybridDeviceOptimizer on ROCm when Relax reaches
  real training but wedges at optimizer step or needs the bf16 CPU-offload safe path.
  Use when: the MI210 path reaches `finished forward_backward` and then stalls or dies at
  `starting optimizer.step`, or when maintaining the validated bf16 HDO path after repeated
  actor training steps complete.
metadata:
  short-description: "Megatron CPU-offload optimizer debugging on ROCm"
  tags:
    - megatron
    - optimizer
    - rocm
    - cpu-offload
    - relax
  domain: research
  created: 2026-04-20
  author: Codex
---

# Megatron Hybrid Device Optimizer Rocm

## General Description

This skill captures the ROCm-specific debugging and validation pattern for Megatron's `HybridDeviceOptimizer` after Relax has already reached real training. It started from the narrow boundary where rollout and forward/backward succeeded but the actor died during the first optimizer update on MI210; it now also records the bf16 CPU-offload safe path that completed repeated actor steps.

## When to Apply

Use this knowledge when:
- The run reaches `train_one_step ... finished forward_backward`.
- The next visible line is `train_one_step ... starting optimizer.step`.
- CPU optimizer offload is enabled on a single-rank ROCm actor path.
- A later run needs to preserve the validated bf16 HDO path that completed
  many actor training steps before a later checkpoint/save boundary.

Do NOT use when:
- The run is still failing in rollout startup, actor initialization, or checkpoint load.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Narrowed failure point | `optimizer.step()` on step 0 | Forward/backward already completed |
| Proven downstream bug | `HybridDeviceOptimizer` ignored `pin_cpu_params=False` | Offloaded params still used pinned host copies |
| Relax-side safety knobs already in place | `pin_cpu_grads=False`, `pin_cpu_params=False`, `overlap_cpu_optimizer_d2h_h2d=False` | These were not sufficient until Megatron respected them |
| Later narrowed boundary | `cpu sub-optimizer 12 step begin` | Grouped CPU sub-optimizers on MI210 still died inside `cpu_optimizer.step()` |
| Current safest fallback | `params_per_optimizer=1` | Single-parameter CPU sub-optimizers passed the clean 5-minute gate |
| Latest validated optimizer state | Actor steps 0-98 completed and rollout 99 completed `optimizer.step`, scheduler step, loss reduction, and metric logging | `log/amd-qwen3-4b-2gpu-20260424_134200.log` |
| Latest non-optimizer boundary | Actor died after `saving checkpoint at iteration 99` | Ray worker exited with `SYSTEM_ERROR`; rollout wedged on `train_99` |
| Foreground validation exit | `124` | External 10-minute timeout while rollout 6 was decoding, not a traceback |
| CPU-copy dtype | `torch.bfloat16` | Confirmed in HDO logs and direct smoke |
| Direct HDO smoke | `updated_mean 0.8984375` | Confirms bf16 CPU-offload update path |

## Recommended Practice

Treat this as a downstream optimizer-implementation problem, not a generic Relax failure.

### Step 1: Prove the boundary is really inside optimizer step

Look for this exact sequence in the actor log:

```text
train_one_step rollout=0 step=0: finished forward_backward
train_one_step rollout=0 step=0: starting optimizer.step
```

If that boundary appears, stop debugging rollout and startup code.

### Step 2: Verify Megatron actually consumes the Relax safety knobs

Inspect the local Megatron checkout, not just Relax. On this stack, the important knobs were:

```text
pin_cpu_grads=False
pin_cpu_params=False
overlap_cpu_optimizer_d2h_h2d=False
```

But `HybridDeviceOptimizer` still did:

```python
param.detach().clone().cpu().pin_memory()
```

for offloaded parameter copies unconditionally.

### Step 3: Patch downstream code when the knob is ignored

Make the local Megatron implementation honor the config:

```python
param = param.detach().clone().cpu()
if self.pin_cpu_params:
    param = param.pin_memory()
```

Do not assume a Relax-side config change is enough if the downstream implementation hardcodes a conflicting behavior.

### Step 4: Revalidate on the real launcher path

After patching Megatron, rerun:

```bash
python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py
bash -n amd_qwen3_4b_2gpu_e2e.sh
timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh
```

The only success signal that matters is that the actor gets farther than the old `starting optimizer.step` boundary on the real MI210 launcher path.

### Step 5: Treat grouped CPU sub-optimizers as optional optimization, not baseline

On this MI210 path, a grouped fallback such as:

```python
params_per_optimizer=4
```

may reduce Python overhead, but it is not the safe default. If the live actor
log narrows the death to:

```text
HybridDeviceOptimizer step: cpu sub-optimizer <n> step begin
```

with no corresponding `step done`, revert the ROCm fallback back to:

```python
params_per_optimizer=1
```

before changing unrelated rollout or Ray code.

### Step 6: Do not overinterpret a clean 5-minute gate

A clean foreground timeout is necessary, but on this stack it is not enough to
claim end-to-end stability. The long-run failure pattern can still be:

- rollout produces `train_<n>`
- the actor disappears later
- `Rollout` loops forever waiting for `train_<n>` to drain

So after the gate passes, always verify the long run by checking both:

- live `MegatronTrainRayActor` presence in `ps`
- whether `train_<n>` is actually drained rather than polled forever

### Step 7: Keep the ROCm safe path bf16 end-to-end

On the single-rank MI210 actor CPU-offload path, do not allow Megatron wrappers
to rebuild HDO around fp32 main params. The stable baseline currently requires:

```python
bf16 = False
fp16 = False
use_precision_aware_optimizer = False
pin_cpu_grads = False
pin_cpu_params = False
overlap_cpu_optimizer_d2h_h2d = False
params_per_optimizer = 1
```

This looks counterintuitive because the model itself is bf16. The point is to
avoid Megatron's fp32 main-param wrapper around the HDO CPU-offload fallback,
while HDO itself keeps CPU copies in bf16.

Patch the local ROCm Megatron checkout so:

- `HybridDeviceOptimizer` disables `param_update_in_fp32` on the unpinned,
  non-overlap HIP safe path.
- `FP32Optimizer.prepare_grads()` casts fp32 `main_grad` to the live parameter
  dtype before assigning `param.grad`.
- `clip_grad_by_total_norm_fp32()` accepts CUDA bf16 grads.

### Step 8: Verify with the full training-step boundary, not only direct smokes

Direct smokes are necessary but not sufficient. The real success signature is:

```text
train_one_step rollout=<n> step=0: finished optimizer.step (update_successful=True, ...)
train_one_step rollout=<n> step=0: scheduler step completed
train_one_step rollout=<n> step=0: loss reduction completed
Actor training completed step <n>/...
```

The 2026-04-24 foreground validation reached this signature for steps 0 through
5 before the external 10-minute validation timeout fired during rollout 6. The
later W&B tmux run reached this signature through actor step 98 and also
completed `optimizer.step`, scheduler step, loss reduction, metric logging, and
`train_actor rollout=99: megatron train completed` for rollout 99.

### Step 9: If the actor dies after Megatron train completes, leave optimizer debugging

If the log reaches:

```text
train_one_step rollout=99 step=0: loss reduction completed
step 99: {...}
train_actor rollout=99: megatron train completed
saving checkpoint at iteration      99 ...
```

then the original HDO optimizer boundary has been cleared. A later
`ActorDiedError` with rollout stuck on `train_99` should be triaged as
checkpoint/save, Ray worker lifecycle, GCS/resource pressure, W&B/metric flush,
or late memory/process death. Do not keep changing HDO unless the next run
regresses to `starting optimizer.step` without `finished optimizer.step`.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Relax disabled pinned CPU params but the optimizer path still wedged | Local Megatron ignored `pin_cpu_params=False` | Verify downstream implementation, not just upstream config |
| Ray still said `RUNNING` after the actor was gone | The job record outlived the useful workers | Use actor process state and log freshness, not Ray status alone |
| Startup fixes were mistaken for end-to-end success | Actor init and rollout were healthy, but optimizer step was still broken | Once `forward_backward` succeeds, move debugging to optimizer internals only |
| Grouped CPU sub-optimizers still died inside `cpu_optimizer.step()` | The grouped AdamW unit was still too large or fragile for this ROCm path | Use single-parameter CPU sub-optimizers as the baseline ROCm-safe fallback |
| A long run wedged on `train_<n>` after the 5-minute gate passed | The actor disappeared later even though startup and initial rollout were healthy | Gate success only proves short-horizon health; inspect the later dead worker log before changing higher-level services |
| fp32 CPU-offload params survived setup but failed in real updates | AdamW state and large CPU steps were too heavy or fragile | Keep HDO CPU copies bf16 on MI210 |
| `FP32Optimizer.prepare_grads()` assigned fp32 `main_grad` into bf16 params | PyTorch rejects gradient assignment when dtype differs from the parameter | Cast `main_grad` to `param.dtype` before assigning `param.grad` |
| Clip-grad asserted `torch.cuda.FloatTensor` | The ROCm safe path can produce bf16 CUDA grads | Accept CUDA bf16 grads in the clipping helper |
| Launcher flags looked correct but HDO still used fp32 copies | Megatron optimizer wrappers rebuilt internals under Relax | Check the actual HDO CPU-copy dtype in logs or direct smokes |
| W&B e2e died after `train_actor rollout=99: megatron train completed` | The actor exited during or after checkpoint save, not during HDO `optimizer.step` | Treat this as a new post-train/checkpoint boundary instead of reworking the bf16 HDO path |

## Configuration

```yaml
scope:
  hardware: MI210 gfx90a
  world_size: 1
  actor_backend: megatron
  optimizer_cpu_offload: true
required_relax_knobs:
  pin_cpu_grads: false
  pin_cpu_params: false
  overlap_cpu_optimizer_d2h_h2d: false
required_megatron_check:
  hybrid_device_optimizer_must_respect_pin_cpu_params: true
current_safe_fallback:
  params_per_optimizer: 1
  cpu_copy_dtype: bfloat16
  disable_megatron_fp32_main_param_wrapper: true
validation_boundary:
  before: "finished forward_backward"
  target: "finished optimizer.step and loss reduction completed"
latest_validation:
  date: 2026-04-24
  foreground_log: "log/amd-qwen3-4b-2gpu-20260424_094405.log"
  wandb_e2e_log: "log/amd-qwen3-4b-2gpu-20260424_134200.log"
  foreground_completed_actor_steps: 6
  wandb_e2e_last_completed_actor_step: 98
  wandb_e2e_optimizer_steps_completed: 100
  wandb_e2e_boundary: "rollout 99 completed optimizer/loss/metrics and then died after checkpoint save began"
  timeout_exit_code: 124
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`
- Troubleshooting: `references/troubleshooting.md`
