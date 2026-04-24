---
name: rocm-relax-bringup
description: >
  Bring up the Relax Qwen3-4B GRPO stack on AMD MI210 ROCm machines and push the run
  from install-time failures into real rollout and training execution.
  Use when: validating Relax, SGLang, and Megatron together on MI210 or similar ROCm hardware,
  especially when proxy behavior, actor memory limits, and TE-less Megatron paths are involved.
metadata:
  short-description: "Relax ROCm bring-up recipe for MI210"
  tags:
    - rocm
    - amd
    - relax
    - sglang
    - megatron
  domain: research
  created: 2026-04-15
  author: Codex
---

# Rocm Relax Bringup

## General Description

This skill captures the practical bring-up sequence that moved Relax on MI210 from failing during environment and startup work into a real end-to-end execution path. It is specifically about ROCm compatibility across Relax, SGLang, Megatron, and Ray Serve, and it records the concrete settings that advanced the run boundary on `gfx90a`.

## When to Apply

Use this knowledge when:
- You are bringing up Relax on AMD MI210 or a close ROCm-equivalent machine.
- The run uses the Qwen3-4B GRPO path with Ray Serve, SGLang, and Megatron.
- Early failures are still in import, kernel build, weight sync, or initial training startup.

Do NOT use when:
- The system already trains stably and you are tuning reward logic, data mixtures, or hyperparameters.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Hardware | 2x MI210 (`gfx90a`) | ROCm bring-up target |
| Cleared failure boundaries | Proxy stall, PPO HIP compile helper crash, Adam-state OOM, TE-version crash, several rollout/control-plane Megatron import leaks | All were reproduced and either removed or narrowed in the MI210 pass |
| Latest validated state | W&B e2e completed actor steps 0-98 and Megatron train/loss reduction for rollout 99 | `log/amd-qwen3-4b-2gpu-20260424_134200.log` |
| Latest long-run boundary | Actor died after `saving checkpoint at iteration 99` | Ray stayed `RUNNING`, rollout wedged on `train_99`, and `MegatronTrainRayActor` disappeared |
| Foreground validation exit | `124` | External 10-minute foreground timeout while rollout 6 was decoding, not a traceback |
| Required Megatron checkout | `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM` | Do not inherit a stale non-ROCm `Megatron-LM` in `PYTHONPATH` |
| W&B mode | `online` | Logged under `relax-amd` |
| Long-run misleading state | Ray and rollout can stay alive after the actor is gone | Judge health by actor liveness and partition drain, not by Ray `RUNNING` alone |

## Recommended Practice

Start by treating ROCm bring-up as a stack-integration problem, not a single package install. Validate each layer in order: environment activation, SGLang import and kernels, Relax weight conversion, Megatron attention path, then the first actor training step. On MI210, keep the launcher explicit rather than relying on defaults that were likely tuned for NVIDIA.

### Step 1: Lock the launcher to the known AMD-safe settings

Use these key launcher settings:

```bash
--qkv-format bshd
--no-masked-softmax-fusion
--no-rope-fusion
--sglang-attention-backend triton
--sglang-sampling-backend pytorch
--sglang-disable-custom-all-reduce
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### Step 2: Validate the stack in the same environment that will run the job

Activate the dedicated environment, source `.env`, and run the short foreground validation first. Confirm that the system reaches rollout generation and reward execution before moving to a long retry loop.

On the ROCm Megatron fork, use the dedicated environment:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm
```

The launcher must put the ROCm Megatron checkout ahead of any inherited
non-ROCm Megatron path:

```bash
MEGATRON_DIR=/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM
```

### Step 3: Fix local-address proxy bypass before debugging service startup

On managed clusters, do not assume localhost-only proxy bypass is enough. Add `MASTER_ADDR`, the resolved `MASTER_ADDR`, the hostname, the resolved hostname IP, plus the localhost entries to both `no_proxy` and `NO_PROXY` for the Ray runtime env and any child-process env propagation. If SGLang says it is ready but rollout health checks hang, validate this first.

### Step 4: Use CPU optimizer offload on the single-GPU actor rank

The bf16 model weights fit on one MI210 actor rank, but Adam state allocation did not. On this 2-GPU split layout, treat optimizer-state placement as part of bring-up. CPU offload was required to remove the first-update HIP OOM.

### Step 5: Keep TE absence an expected ROCm configuration

When Transformer Engine is not installed, capability checks must degrade to `False`, not crash. If the offload path hits `is_te_min_version()` or similar optional-dependency probes, patch them toward explicit unsupported behavior instead of allowing type errors to abort actor initialization.

### Step 6: Interpret failures by boundary, not by component name

If the run fails before Serve deployment, focus on environment and import issues. If it fails after rollout generation, stop calling it bring-up and move to Megatron or TorchInductor runtime debugging. This distinction mattered in the MI210 overnight loop because the later failures were no longer SGLang startup problems.

### Step 7: Do not trust a stale Ray `RUNNING` status

On this stack, the Ray job record can still say `RUNNING` after the useful workers are gone. Always cross-check:

- the latest actor-side timestamps in the log
- whether `MegatronTrainRayActor` is still alive in `ps`
- whether only the driver, rollout, and Ray control plane remain

If the actor is gone and the log is stale, treat the run as wedged or orphaned even if Ray still says `RUNNING`.

### Step 7.1: Use the `train_<n>` drain state as the practical health signal

On this MI210 path, a long run can pass the 5-minute gate and still fail later
by producing a training partition and then losing the actor. The signature is:

- `Rollout` repeatedly waits on `Current partitions: ['train_<n>']`
- `SGLangEngine` only serves health / tiny keepalive traffic
- `MegatronTrainRayActor` is missing from `ps`

Treat that as a dead training path, not a slow run.

### Step 8: Clean the import surface before assuming a ROCm runtime bug

On this stack, several failures that looked like ROCm runtime issues were actually control-plane or rollout-side import leaks. Check these layers in order:

1. `relax.distributed.checkpoint_service` package init
2. `relax.core.registry` through `relax.components.advantages`
3. `Rollout.__init__`
4. `RolloutManager.__init__`
5. `SGLangEngine` worker startup

If a plain interpreter import of a control-plane module already imports `megatron` or `sglang`, fix that import surface first. The MI210 path advanced materially only after converting DCS exports to lazy lookups, moving Megatron imports out of `advantages.py`, and installing transformers-mode isolation before rollout-side function loading.

### Step 9: Verify that downstream Megatron code actually respects the ROCm safety knobs

It is not enough to set `pin_cpu_grads=False` or `pin_cpu_params=False` in Relax. The local Megatron implementation must consume those knobs correctly. On this stack, `HybridDeviceOptimizer` still forced `.pin_memory()` for offloaded parameter copies even after the Relax-side actor path disabled pinned CPU params. Fixing that local Megatron bug was necessary to make the ROCm optimizer safety settings real.

### Step 10: Keep the ROCm CPU-offload fallback conservative until long runs prove otherwise

The MI210 path narrowed the optimizer death to grouped CPU sub-optimizer steps.
If the actor log dies after:

```text
HybridDeviceOptimizer step: cpu sub-optimizer <n> step begin
```

revert the unpinned, non-overlap ROCm fallback back to:

```text
params_per_optimizer=1
```

before trying new higher-level changes. Grouped CPU sub-optimizers are an
optimization, not the stability baseline.

### Step 11: Treat repeated completed actor steps as the new foreground gate

The old boundary was the first real `optimizer.step()`. That is no longer the
current validated boundary. A useful MI210 foreground gate should now confirm
at least one complete training cycle:

```text
rollout generation
train batch transfer
forward_backward
optimizer.step
scheduler step
loss reduction
Actor training completed step
```

The 2026-04-24 foreground validation completed this cycle for actor steps 0
through 5 and timed out only because the external foreground timeout fired
during rollout 6. The later W&B tmux run completed this same cycle through
actor step 98 and completed Megatron train/loss reduction for rollout 99.
After a foreground timeout, still stop Ray and confirm that no Ray/SGLang/Relax
workers or GPU memory allocations remain before starting tmux.

### Step 12: Treat checkpoint-save boundaries separately from optimizer bring-up

The W&B e2e run no longer died at the first optimizer step. It reached:

```text
train_one_step rollout=99 step=0: loss reduction completed
step 99: {...}
train_actor rollout=99: megatron train completed
saving checkpoint at iteration      99 ...
```

Then the `MegatronTrainRayActor` exited with Ray `SYSTEM_ERROR`, while Ray still
reported `RUNNING` and rollout waited forever on `train_99`. Future debugging
should start at checkpoint/save, GCS/resource pressure, worker-exit logs, or
late memory/process failure, not at the original optimizer-step boundary.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| SGLang ROCm import and JIT/kernel bring-up | CUDA-centric assumptions in the local SGLang stack | Patch SGLang for ROCm and build the local kernel package for `gfx90a` before chasing higher-level bugs |
| Qwen weight conversion during sync | Relax expected older layernorm names | Accept current Megatron layernorm aliases in the converter |
| Packed-sequence attention startup | Plain `DotProductAttention` on ROCm does not support the packed path | Force `bshd` in the AMD launcher |
| Rollout init hung after SGLang was healthy | Node-local HTTP traffic was routed through the cluster proxy | Include resolved local addresses in `no_proxy` and `NO_PROXY` |
| First actor update OOMed in Adam state allocation | The single-GPU actor rank could not also hold the optimizer moments on device | Use Megatron CPU optimizer offload on MI210 |
| CPU offload path crashed before training | Megatron compared a missing TE version as if it were a real version object | Make TE absence return `False` in capability checks |
| Control-plane services emitted Megatron warnings before rollout startup | Package init and registry imports were pulling Megatron in at file scope | Treat package `__init__` and registry imports as bring-up boundaries, not harmless convenience imports |
| Rollout-side startup still looked contaminated after controller cleanup | Isolation was only installed in `SGLangEngine`, but rollout functions and helpers loaded earlier | Install transformers-mode isolation in `Rollout` and `RolloutManager` before loading rollout functions |
| Step-0 training wedged at `optimizer.step()` even after Relax disabled pinned CPU params | The local Megatron `HybridDeviceOptimizer` ignored `pin_cpu_params=False` and still forced pinned host parameter copies | Verify downstream optimizer code honors the ROCm safety knobs; patch local Megatron when it does not |
| Ray still reported `RUNNING` long after the actor was gone | The job record outlived the useful workers | Use worker/process state and log freshness, not Ray job status alone, to decide whether a run is healthy |
| A long run passed the 5-minute gate and still wedged on `train_<n>` | The actor disappeared later while rollout and SGLang survived | A clean short gate only proves short-horizon health; long-run health requires actor liveness and partition drain |
| Grouped CPU sub-optimizers still died on MI210 | The grouped `cpu_optimizer.step()` unit remained unstable on this ROCm path | Keep `params_per_optimizer=1` as the ROCm-safe fallback unless a long run proves grouping is stable |
| The skill still described optimizer step as the current boundary | The 2026-04-24 validation completed actor steps 0-5 | Future debugging should start from the next long-run boundary, not redo the first-step optimizer work |
| W&B tmux run died after step-99 Megatron train completed | The actor emitted `saving checkpoint at iteration 99`, then Ray reported the worker exited with `SYSTEM_ERROR` and rollout wedged on `train_99` | The next boundary is checkpoint/save or late worker/resource failure, not first-step optimizer stability |
| Foreground validation exited with code `124` | The external timeout fired while rollout 6 was decoding | Interpret timeout status with log context before treating it as failure |
| Launcher inherited the old Megatron path | `MEGATRON_DIR` was set, but inherited `PYTHONPATH` could still point at non-ROCm Megatron | Construct `PYTHONPATH` explicitly with ROCm Megatron ahead of stale paths |

## Configuration

```yaml
environment: relaxrl_rocm
hardware: MI210 gfx90a
megatron_checkout: /vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM
launcher_flags:
  qkv_format: bshd
  masked_softmax_fusion: false
  rope_fusion: false
  sglang_attention_backend: triton
  sglang_sampling_backend: pytorch
  sglang_disable_custom_all_reduce: true
  optimizer_cpu_offload: true
  optimizer_offload_fraction: 1.0
  overlap_cpu_optimizer_d2h_h2d: true
  use_precision_aware_optimizer: true
runtime_env:
  extend_no_proxy_with:
    - 127.0.0.1
    - localhost
    - ::1
    - MASTER_ADDR
    - resolved_MASTER_ADDR
    - hostname
    - resolved_hostname
import_surface_checks:
  - relax.distributed.checkpoint_service.coordinator.service
  - relax.core.registry
  - relax.components.rollout
  - relax.distributed.ray.rollout
  - relax.backends.sglang.sglang_engine
wandb:
  enabled: true
  mode: online
  project: relax-amd
health_checks:
  require_live_actor_process: true
  do_not_trust_ray_running_alone: true
optimizer_debugging:
  verify_downstream_knobs:
    - pin_cpu_grads
    - pin_cpu_params
    - overlap_cpu_optimizer_d2h_h2d
  conservative_cpu_sub_optimizer_chunking: 1
  keep_hdo_cpu_copies_bf16: true
  avoid_fp32_main_param_wrapper_on_cpu_offload: true
long_run_health_checks:
  require_train_partition_to_drain: true
  require_live_megatron_actor: true
latest_validation:
  date: 2026-04-24
  foreground_log: log/amd-qwen3-4b-2gpu-20260424_094405.log
  wandb_e2e_log: log/amd-qwen3-4b-2gpu-20260424_134200.log
  foreground_completed_actor_steps: 6
  wandb_e2e_completed_actor_steps: 99
  wandb_e2e_last_completed_actor_step: 98
  wandb_e2e_boundary: "rollout 99 completed Megatron train/loss reduction, emitted checkpoint save, then actor died and rollout wedged on train_99"
  foreground_timeout_exit_code: 124
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `megatron-bridge-rocm-overrides`, `rocm-inductor-triton-cluster-dims`, `ray-rollout-import-isolation`
- Troubleshooting: `references/troubleshooting.md`
