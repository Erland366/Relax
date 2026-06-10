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

Current AMD Qwen3-4B e2e validation uses the TP2 GPU optimizer path, not CPU
optimizer offload: four visible GPUs, two actor GPUs, two rollout GPUs,
`--tensor-model-parallel-size 2`, `--sequence-parallel`, `CKPT_FORMAT=torch_dist`,
and `NO_SAVE_OPTIM=0`.

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
| Hardware | MI210 (`gfx90a`), current validation uses 4 GPUs | Earlier 2-GPU work is historical context |
| Cleared failure boundaries | Proxy stall, PPO HIP compile helper crash, Adam-state OOM, TE-version crash, SGLang HIP JIT failures, torch_dist save/resume failures, several rollout/control-plane Megatron import leaks | All were reproduced and either removed or narrowed in the MI210 pass |
| Current validated state | TP2 GPU optimizer path resumed from iteration 99, started at actor step 100, saved iterations 119 and 139, and continued through completed step 154 | W&B `6c3wrymd`; checkpoint dir `Qwen3-4B_mcore_4gpu-tp2-normpatch-overnight-20260530_233548` |
| Latest checkpoint boundary | Megatron `torch_dist` save and explicit load work on ROCm with optimizer state on the TP2 GPU optimizer path | Save requires the Relax ROCm checkpoint hook; load requires explicit `LOAD_DIR` and trusted `common.pt` loader |
| Qwen3-0.6B after_fix smoke | Ray job `raysubmit_vQKenPVHTm4HiKL9` succeeded with W&B offline | Clean command-only run used `relaxrl_rocm_after_fix` and prepended SGLang's built `sgl_kernel` artifact to `PYTHONPATH` |
| Qwen3-0.6B fully_async smoke | Ray job `raysubmit_cvLQL4xbyhdB9L4d` succeeded with W&B offline | Four-GPU bounded-staleness run used actor, rollout, actor_fwd, async DCS weight update, and torch_dist optimizer checkpoints |
| Foreground validation exit | `124` | External 300-second checkpoint-enabled smoke timeout before actor checkpoint save; cleanup was required before tmux |
| Required Megatron checkout | `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM` | Do not inherit a stale non-ROCm `Megatron-LM` in `PYTHONPATH` |
| W&B mode | `online` | Logged under `relax-amd` |
| Long-run misleading state | Ray and rollout can stay alive after the actor is gone | Judge health by actor liveness and partition drain, not by Ray `RUNNING` alone |

## Recommended Practice

Start by treating ROCm bring-up as a stack-integration problem, not a single package install. Validate each layer in order: environment activation, SGLang import and kernels, Relax weight conversion, Megatron attention path, then the first actor training step. On MI210, keep the launcher explicit rather than relying on defaults that were likely tuned for NVIDIA.

### Step 1: Lock the launcher to the known AMD-safe settings

Use these key launcher settings:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3
RAY_NUM_GPUS=4
NUM_GPUS_PER_NODE=4
ACTOR_RESOURCE_GPUS=2
ROLLOUT_RESOURCE_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_LABEL=4gpu-tp2
SAVE_INTERVAL=20
CKPT_FORMAT=torch_dist
NO_SAVE_OPTIM=0
SCHEDULER_RESUME_POLICY=strict
```

Keep these Megatron/SGLang flags on the AMD path:

```bash
--sequence-parallel
--qkv-format bshd
--no-masked-softmax-fusion
--no-rope-fusion
--sglang-attention-backend triton
--sglang-sampling-backend pytorch
--sglang-disable-custom-all-reduce
--sglang-disable-cuda-graph
--sglang-disable-overlap-schedule
```

Do not add CPU optimizer offload flags for the current Qwen3-4B objective.

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

### Step 2.1: Verify the built SGLang kernel path, not only the source checkout

The local SGLang checkout can be present while `sgl_kernel` is still missing
from the Ray runtime. For the validated Qwen3-0.6B `after_fix` smoke, the fix
was to prepend the built kernel artifact before the SGLang source path:

```bash
SGL_KERNEL_BUILD=/vast/users/qirong.ho/erland/Python_project/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-312
SGLANG_PYTHON=/vast/users/qirong.ho/erland/Python_project/sglang/python
PYTHONPATH="${SGL_KERNEL_BUILD}:${SGLANG_PYTHON}:${PYTHONPATH}" \
  python -c "import sglang, sgl_kernel"
```

Do not install `cuda-python` to solve this on ROCm. If only the source tree is
added, imports can still fail on missing built pieces such as `common_ops`.

### Step 3: Fix local-address proxy bypass before debugging service startup

On managed clusters, do not assume localhost-only proxy bypass is enough. Add `MASTER_ADDR`, the resolved `MASTER_ADDR`, the hostname, the resolved hostname IP, plus the localhost entries to both `no_proxy` and `NO_PROXY` for the Ray runtime env and any child-process env propagation. If SGLang says it is ready but rollout health checks hang, validate this first.

### Step 4: Use TP2 GPU optimizer state, not CPU optimizer offload

The bf16 model weights fit on one MI210 actor rank, but Adam state allocation did not. CPU optimizer offload was tried historically, but it repeatedly introduced ROCm HDO crashes and is no longer acceptable for the current AMD Qwen3-4B objective.

The current fit path is to give the actor two GPUs and use tensor parallelism:

```bash
ACTOR_RESOURCE_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2
ENABLE_SEQUENCE_PARALLEL=1
```

Relax should fail loudly if a ROCm Megatron actor requests
`optimizer_cpu_offload=True` on this path.

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

### Step 9: Treat CPU-offload debugging as historical context for this run

It is not enough to set `pin_cpu_grads=False` or `pin_cpu_params=False` in Relax. The local Megatron implementation must consume those knobs correctly. On this stack, `HybridDeviceOptimizer` still forced `.pin_memory()` for offloaded parameter copies even after the Relax-side actor path disabled pinned CPU params. Fixing that local Megatron bug was necessary for the old CPU-offload path, but that path is now superseded for the current AMD Qwen3-4B run.

Do not re-enable CPU optimizer offload just because those historical patches
exist. Use them only when intentionally debugging a legacy HDO path.

### Step 10: Keep the ROCm CPU-offload fallback conservative when testing legacy HDO

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
optimization, not the stability baseline. This is legacy guidance; the current
e2e path should reject CPU optimizer offload entirely.

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

For the AMD MI210 path, keep checkpointing enabled and use Megatron
`torch_dist` checkpoint format by default:

```bash
SAVE_INTERVAL=100
CKPT_FORMAT=torch_dist
NO_SAVE_OPTIM=0
```

The launcher appends `--save`, `--save-interval`, and `--ckpt-format`.
`CKPT_FORMAT=torch_dist` is validated on this ROCm path only when the Relax
ROCm checkpoint hook is active. The hook disables the stale
`dist_ckpt_save_pre_mcore_014` override on PyTorch >= 2.6, patches both
Megatron torch-dist writer aliases, forces `flatten_sharded_tensors=False` for
Megatron DCP planners, and streams checkpoint tensors through blocking
per-tensor CPU staging instead of Megatron's original async fork/preload path.

The checkpoint-enabled torch-dist path was validated on 2026-05-30 with
`SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=2` and
optimizer checkpointing enabled. The tmux run completed live rollout/training
steps 0/2 and 1/2, saved `iter_0000000` and `iter_0000001` with `.metadata`,
two `.distcp` files, `common.pt`, and `metadata.json` per iteration, wrote
`latest_checkpointed_iteration.txt`, logged `All training steps finished`, and
Ray reported job `raysubmit_FeKYagrKwzrfrcPU` as succeeded.

The current four-GPU TP2 run extends that proof. It saved checkpoints at
iterations 19, 39, 59, 79, 99, 119, and 139 with optimizer state enabled, then
resumed from iteration 99 and continued through completed actor step 154 before
manual stop. The remaining absolute e2e proof is the final 200-step completion
and final checkpoint/resume audit.

For fresh-launch resume tests, `SAVE_DIR` is not enough. `SAVE_DIR` maps to
Megatron `--save`; use `LOAD_DIR=/path/to/checkpoint` to pass `--load`.
Verify the actor starts at `latest_checkpointed_iteration + 1` before allowing
the run to continue, otherwise stop it before it overwrites iteration 0.
On PyTorch 2.6, the same ROCm checkpoint hook must also patch Megatron's
common-state loader so trusted local `common.pt` uses `weights_only=False`;
without that, explicit `LOAD_DIR` fails on OmegaConf `DictConfig` metadata
before the actor can report its restored starting step.

If the resumed run intentionally changes scheduler-driving args such as
`NUM_ROLLOUT`, choose the scheduler policy explicitly:

```bash
SCHEDULER_RESUME_POLICY=strict      # default Megatron mismatch check
SCHEDULER_RESUME_POLICY=override    # keep the new run's scheduler horizon
SCHEDULER_RESUME_POLICY=checkpoint  # keep checkpoint scheduler values
```

The post-resume train validation extended the short checkpoint smoke from
`NUM_ROLLOUT=2` to `NUM_ROLLOUT=3`, so it used
`SCHEDULER_RESUME_POLICY=override`.

On the single-rank HIP actor CPU-offload path, explicit `LOAD_DIR` also needs
the ROCm HDO restore hook in `relax/backends/megatron/optimizer_utils.py`.
There are two separate optimizer restore boundaries:

- `FP32Optimizer.sharded_state_dict(is_loading=True)` needs an HDO Adam-state
  initializer; otherwise it fails with `TypeError: 'NoneType' object is not
  callable`.
- `FP32Optimizer.load_state_dict()` must load checkpoint state onto HDO inner
  CPU-offload params, not the public live GPU model params; otherwise the actor
  can die immediately after `checkpoint version 3.0`.

The validated explicit-resume signature is:

```text
Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore
loading distributed checkpoint from ... at iteration 1
Initialized 290 HybridDeviceOptimizer Adam states for ROCm checkpoint restore
checkpoint version 3.0
Actor initialized with starting step 2
All training steps finished
Job 'raysubmit_KiRTCAx7v8RCU9L6' succeeded
```

This is not a valid place to disable optimizer checkpointing. The validated
run used `NO_SAVE_OPTIM=0`; if this path regresses, inspect HDO state placement
before changing checkpoint intervals or save/load flags.

The stronger post-resume training signature is:

```text
Actor initialized with starting step 2
Actor training step 2/3
saving checkpoint at iteration       2 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration       2
Actor training completed step 2/3
All training steps finished
Job 'raysubmit_tD6jkVCbk6t7cZ21' succeeded
TMUX_STATUS=0
```

### Step 13: Re-apply the ROCm JIT checklist after syncing upstream

After merging a newer Relax upstream, revalidate the SGLang and Megatron
runtime boundaries before assuming the older MI210 recipe still holds. The
2026-05-29 post-sync run needed all of these checks:

- keep `--sglang-disable-overlap-schedule` enabled on MI210;
- on HIP, disable SGLang's JIT KV-cache `store_cache` path so SGLang uses the
  tensor assignment fallback;
- on HIP, route SGLang clamp-position through `_clamp_position_native` instead
  of `clamp_position_cuda`;
- treat Megatron distributed world size 1 as the HF checkpoint warmup leader
  condition, even when Ray assigns the actor `LOCAL_RANK=1`;
- do not treat `waiting for data system to catch up` as a wedge by itself. It
  is only a stale `train_<n>` failure if the actor disappears or the partition
  stops draining;
- keep checkpointing enabled and verify the `torch_dist` save markers:
  `flatten_sharded_tensors=False`, `thread-local checkpoint results queue`,
  `ROCm streaming checkpoint write`, and `successfully saved checkpoint`.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| SGLang ROCm import and JIT/kernel bring-up | CUDA-centric assumptions in the local SGLang stack | Patch SGLang for ROCm and build the local kernel package for `gfx90a` before chasing higher-level bugs |
| Qwen weight conversion during sync | Relax expected older layernorm names | Accept current Megatron layernorm aliases in the converter |
| Packed-sequence attention startup | Plain `DotProductAttention` on ROCm does not support the packed path | Force `bshd` in the AMD launcher |
| Rollout init hung after SGLang was healthy | Node-local HTTP traffic was routed through the cluster proxy | Include resolved local addresses in `no_proxy` and `NO_PROXY` |
| First actor update OOMed in Adam state allocation | The single-GPU actor rank could not also hold the optimizer moments on device | Use the TP2 GPU optimizer path with two actor GPUs and sequence parallel; do not switch to CPU optimizer offload |
| CPU offload path crashed before training | Megatron compared a missing TE version as if it were a real version object | Treat this as historical context; current AMD Qwen3 launches should reject CPU optimizer offload |
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
| SGLang overlap scheduler future-token kernel failed on ROCm | SGLang loaded weights and started Uvicorn, then crashed in `resolve_future_token_ids_cuda` with `CUDA error: no ROCm-capable device is detected` | Disable the overlap scheduler on this MI210 launcher with `--sglang-disable-overlap-schedule` until the HIP JIT kernel path is validated |
| SGLang JIT KV-cache store kernel failed on ROCm | After overlap scheduling was disabled, the normal scheduler crashed in `kvcache.cuh:196` while storing KV cache | Keep Relax's HIP runtime patch active so SGLang bypasses the JIT `store_cache` kernel and uses tensor assignment fallback until the HIP kernel is validated |
| SGLang JIT clamp-position kernel failed on ROCm | During rollout decode, SGLang crashed in `clamp_position.cuh:46` and the router returned 503 because the worker became unhealthy | Keep Relax's HIP runtime patch active so SGLang uses `_clamp_position_native` instead of the JIT `clamp_position_cuda` helper on ROCm |
| Megatron actor waited forever on HF checkpoint warmup | Ray placed the single Megatron actor on local GPU 1 while SGLang owned GPU 0, so `LOCAL_RANK=1` waited for a nonexistent Megatron local rank 0 | Treat distributed world size 1 as the HF checkpoint warmup leader condition, while keeping the marker/lock guard for duplicate jobs |
| Megatron torch-dist checkpoint save killed the actor before the planner patch | Forced saves with `SAVE_INTERVAL=2 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=1` wrote only `common.pt`, then the actor exited with Ray EOF / `SYSTEM_ERROR` | Keep the Relax ROCm checkpoint hook active; it mirrors Megatron's current `flatten_sharded_tensors=False` planner setting and uses a HIP-safe streaming writer |
| Lazy ROCm torch-dist writer failed in Python | `ROCm lazy checkpoint prepare` was followed by `cannot unpack non-iterable WriteItem object` | Lazy buckets contain raw DCP `WriteItem` objects; pass the planner into the streaming write path and resolve each item immediately before writing |
| Resume test restarted at step 0 | Pointing only `SAVE_DIR` at an existing checkpoint still initialized the actor from step 0 | Set `LOAD_DIR` to pass Megatron `--load`; `SAVE_DIR` only controls where the next checkpoint is saved |
| Explicit `LOAD_DIR` failed on `common.pt` | PyTorch raised `_pickle.UnpicklingError` for `omegaconf.dictconfig.DictConfig` while loading Megatron common state | PyTorch 2.6 defaults `torch.load` to `weights_only=True`; patch Megatron's common loader on HIP to use `weights_only=False` for trusted local checkpoint metadata |
| Explicit `torch_dist` resume failed in `FP32Optimizer.sharded_state_dict` | Megatron wrapped HDO with a missing `init_state_fn` on the ROCm restore path | Install the narrow Relax HDO Adam-state initializer for single-rank HIP actor CPU offload |
| Explicit `torch_dist` resume died after `checkpoint version 3.0` | Generic optimizer state load targeted HDO public GPU params instead of inner CPU-offload params | Patch `FP32Optimizer.load_state_dict` on the ROCm HDO path to load state onto inner params and resync sub-optimizers |
| Extending a resumed smoke changed the scheduler horizon | `NUM_ROLLOUT=3` produced `class input value 48` while the checkpoint stored `32` | Keep strict mode by default, but use `SCHEDULER_RESUME_POLICY=override` for deliberate continuation with a new horizon |
| `sgl_kernel` was installed locally but missing in Ray workers | The runtime `PYTHONPATH` included SGLang's source checkout but not `sgl-kernel/build/lib.linux-x86_64-cpython-312` | Prepend the built SGLang kernel artifact before `sglang/python`; do not install CUDA-only packages in the ROCm environment |
| Fully_async foreground gate timed out before e2e completion | Actor, rollout, actor_fwd, SGLang, and checkpoint startup can exceed five minutes on MI210 | Treat exit `124` as a pass-to-tmux signal only after startup markers are healthy; e2e success requires actor_fwd logprobs, optimizer steps, checkpoint save, and Ray job success |

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
  sglang_disable_overlap_schedule: true
  actor_resource_gpus: 2
  rollout_resource_gpus: 2
  tensor_model_parallel_size: 2
  sequence_parallel: true
  save_interval: 20
  ckpt_format: torch_dist
  no_save_optim: false
  load_dir_for_resume: optional
  optimizer_cpu_offload: false
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
resume_health_checks:
  require_explicit_load_dir: true
  common_pt_weights_only_false: true
  hdo_adam_init_state_fn: true
  hdo_load_state_on_inner_params: true
  scheduler_resume_policy: strict_by_default
latest_validation:
  date: 2026-05-31
  current_topology: "4x MI210, actor=2 GPUs, rollout=2 GPUs, TP=2"
  current_save_dir: Qwen3-4B_mcore_4gpu-tp2-normpatch-overnight-20260530_233548
  current_resume_loaded_iteration: 99
  current_resume_starting_step: 100
  current_latest_checkpoint_seen: 139
  current_latest_completed_step_seen: 154
  current_completion_run: raysubmit_npbaWGLVyyJMxhD8
  historical_smoke_date: 2026-05-30
  smoke_log: validation_logs/live_torch_dist_lazyfix_tmux_20260530_112514.log
  smoke_tmux_session: tmux-13
  smoke_ray_job: raysubmit_FeKYagrKwzrfrcPU
  smoke_checkpoint_args: "--save --save-interval 1 --ckpt-format torch_dist"
  smoke_completed_actor_steps: 2
  smoke_last_completed_actor_step: 1
  saved_checkpoint: Qwen3-4B_mcore_2gpu-live-torch-dist-lazyfix-20260530_112514
  torch_dist_artifacts: ".metadata, __0_0.distcp, __0_1.distcp, common.pt, metadata.json, latest_checkpointed_iteration.txt"
  checkpoint_metadata_entries: 175
  checkpoint_storage_entries: 1123
  optimizer_state_present: true
  explicit_resume_log: validation_logs/live_torch_dist_explicit_load_hdoload_tmux_20260530_132101.log
  explicit_resume_tmux_session: tmux-15
  explicit_resume_ray_job: raysubmit_KiRTCAx7v8RCU9L6
  explicit_resume_wandb_run: aghbilif
  explicit_resume_loaded_iteration: 1
  explicit_resume_starting_step: 2
  explicit_resume_status: succeeded
  post_resume_train_log: validation_logs/live_torch_dist_resume_train_override_tmux_20260530_135915.log
  post_resume_train_tmux_session: tmux-16
  post_resume_train_ray_job: raysubmit_tD6jkVCbk6t7cZ21
  post_resume_train_wandb_run: 5fw7cmi1
  post_resume_scheduler_policy: override
  post_resume_starting_step: 2
  post_resume_completed_actor_step: 2
  post_resume_saved_checkpoint: Qwen3-4B_mcore_2gpu-resume-train-override-20260530_135915
  post_resume_saved_iteration: 2
  post_resume_status: succeeded
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `qwen3-0-6b-rocm-fully-async-e2e`, `qwen3-0-6b-rocm-sgl-kernel-e2e`, `rocm-megatron-tp2-checkpoint-resume`, `megatron-bridge-rocm-overrides`, `rocm-inductor-triton-cluster-dims`, `ray-rollout-import-isolation`
- Troubleshooting: `references/troubleshooting.md`
