# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Packed sequence with DotProductAttention on ROCm | Training dies after startup when Megatron reaches packed-sequence attention | AMD path is using plain `DotProductAttention`, which does not support packed sequence and expects TE-backed attention instead | Force the launcher to use `--qkv-format bshd` so the run avoids the THD packed-sequence path |
| `scaled_masked_softmax_cuda` import on ROCm | Training crashes inside Megatron fused softmax even after passing `--no-masked-softmax-fusion` | The launcher flag alone did not reach every Megatron-Bridge override path, and upstream fused softmax code still assumes the NVIDIA extension import exists | Propagate `masked_softmax_fusion` through the provider path and ensure fused softmax falls back cleanly when CUDA extensions are unavailable |
| TorchInductor ROCm `KernelMetadata.cluster_dims` failure | First actor training step crashes after rollout generation and reward execution | TorchInductor/Triton on this ROCm stack generates a kernel metadata object without `cluster_dims`, but the launcher code path expects it | Treat this as a compiler/runtime compatibility issue: reduce or disable the affected compiled path, or move to a PyTorch/Triton build where ROCm launcher metadata matches TorchInductor expectations |
| ROCm Megatron `jit_fuser` uses compiler decorators on HIP | The no-CPU-offload run either fails in `megatron/core/fusions/fused_cross_entropy.py` with `torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`, or fails during import when TorchScript compiles `L2Norm` and cannot resolve `self.eps` | `ROCm-Megatron-LM/megatron/core/jit.py` promotes `jit_fuser` from `torch.jit.script` to `torch.compile` on PyTorch >= 2.2; avoiding `torch.compile` still leaves TorchScript method limitations on HIP | Patch the local ROCm Megatron checkout so `jit_fuser` is an eager no-op when `torch.version.hip` is set, then rerun the foreground validation past import and the old step-0 boundary |
| ROCm CPU-offload safe path is undone by fp32 main-param wrapping | The actor reaches `optimizer.step()` but either wedges on a large CPU AdamW step with fp32 CPU copies, raises `RuntimeError: attempting to assign a gradient with dtype 'float' to a tensor with dtype 'c10::BFloat16'`, or hits a fp32-only clip-grad assertion | Megatron's mixed-precision optimizer wrapper can rebuild HDO over fp32 main params, then later routes bf16 live params through optimizer helper code that assumes CUDA float grads | On the single-rank ROCm actor CPU-offload path, force the optimizer config away from fp32/bf16 mixed wrappers, refresh HDO only when the wrapped optimizer is not `FP32Optimizer`, keep HDO CPU copies bf16, cast `main_grad` to the live param dtype in `FP32Optimizer.prepare_grads()`, and allow bf16 CUDA grads in `clip_grad_by_total_norm_fp32()` |
| Internal proxy intercepts local SGLang health checks | Rollout initialization hangs even though SGLang reports the server is ready | Internal HTTP requests to the node-local SGLang server go through the corporate proxy because the node hostname/IP is missing from `no_proxy` | Add `MASTER_ADDR`, the resolved local hostname/IP, and localhost entries to both `no_proxy` and `NO_PROXY` for the launcher runtime env and child processes |
| Adam state OOM on MI210 actor rank | First actor update dies on `torch.optim.adam._init_group` with HIP OOM while allocating `exp_avg`/`exp_avg_sq` | A single MI210 actor rank can hold the 4B bf16 model, but lazy Adam state allocation still exceeds device memory during the first optimizer step | Use Megatron CPU optimizer offload with `--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` |
| CPU optimizer offload crashes when TE is absent | Actor initialization fails before training with `TypeError: '>=' not supported between instances of 'NoneType' and 'Version'` | Megatron's `is_te_min_version()` assumes `get_te_version()` always returns a version object, but on TE-less ROCm systems it returns `None` | Patch the local Megatron checkout so `is_te_min_version()` returns `False` when Transformer Engine is not installed |
| SGLang transformers backend discovers local Megatron-LM | The actor dies during rollout startup with a generic Ray `ActorDiedError`, while the SGLang child logs Megatron-FSDP / optimizer warnings from the local `Megatron-LM` checkout | The rollout-side SGLang subprocess inherits `PYTHONPATH`, finds the local Megatron checkout, and switches into Megatron-specific code paths even though the launcher requested the plain `transformers` backend | When spawning SGLang with `model_impl=transformers`, temporarily strip `Megatron-LM` entries from the child `PYTHONPATH` so the SGLang server cannot discover the local Megatron checkout |
| ROCm launcher inherits stale non-ROCm Megatron path | A run from `Relax-rocm-megatron` still logs imports from `/vast/users/qirong.ho/erland/Python_project/Megatron-LM` instead of `ROCm-Megatron-LM` | The launcher set `MEGATRON_DIR` but built `PYTHONPATH` from the inherited shell value, so an older `Megatron-LM` entry could remain ahead of the ROCm fork in Ray's runtime env | Make `amd_qwen3_4b_2gpu_e2e.sh` construct `PYTHONPATH` explicitly as SGLang, `${MEGATRON_DIR}`, then `${ROOT_DIR}` instead of appending inherited `PYTHONPATH` |
| SGLang transformers import fails on optional Quark `aiter` path | Rollout startup fails inside `SGLangEngine.init()` with `Exception: Server process terminated unexpectedly`, and the scheduler traceback ends in `ValueError: Model architectures ['TransformersForCausalLM'] are not supported for now` | The generic SGLang transformers model class never registers because importing `sglang.srt.models.transformers` pulls in `ep_moe.layer`, which hard-imports Quark MXFP4 MoE schemes that require the missing optional `aiter` package | Patch the local SGLang checkout so the Quark MXFP4 MoE import is optional in `ep_moe.layer`, log that Quark support is unavailable, and allow the generic transformers backend to load without `aiter` |
| Actor dies during sync `set_rollout_manager` startup on MI210 | The run gets through actor init, rollout init, RolloutManager creation, and SGLang server launch, then dies when the controller calls `Actor.set_rollout_manager`, with the nested failure at `MegatronTrainRayActor.set_rollout_manager` | The sync training path was still doing fully-async rollout-manager setup: it made extra Ray round-trips to `set_train_parallel_config()` and `get_weight_sync_lock()` even though those fields are only used by the fully-async DCS weight-sync path | In `TrainRayActor.set_rollout_manager()`, return early for non-`fully_async` runs after storing the rollout-manager handle, and keep the extra rollout-manager wiring only for the fully-async path |
| Actor dies during `set_rollout_manager` after offloaded init sleep | The run survives actor init and rollout bring-up but the actor still dies on the first post-init rollout-manager RPC when CPU optimizer offload is enabled | The offloaded Megatron actor was going to sleep at the end of `_init()` even for the sync path, so the first `set_rollout_manager` call had to wake a partially torn-down process-group state during bootstrap | Track whether the actor is actually sleeping, only wake it when needed, and keep the sync offloaded actor resident until rollout-manager hookup is complete |
| Ray GCS times out during late rollout startup on single-node MI210 | The run clears actor initialization, reaches rollout creation, then the driver dies with `Failed to connect to GCS within 60 seconds` and Serve reports `Deadline Exceeded` while fetching resource usage | The single-node head was overprovisioned at 128 CPUs for a 2-GPU job and was generating excessive Ray control-plane traffic; internal actors were also still publishing task-event metadata the run did not need | Reduce the head to a modest CPU count, disable task events on internal Relax Ray actors/managers, and increase GCS reconnect timeouts so short control-plane stalls do not kill the whole job |
| Ray keepalive watchdog timeout during SGLang startup | The rollout replica hangs in `SGLangEngine.init()`, then Serve reports `ActorUnavailableError ... keepalive watchdog timeout rpc_code: 14`, and only later does the controller notice other actors died | The actual failure surface is a Ray control-plane stall during the long SGLang bring-up window: worker backlog reporting grows large enough that actor RPCs trip the keepalive watchdog before rollout initialization finishes | Disable periodic task-event reporting with `RAY_task_events_report_interval_ms=0`, widen Ray gRPC client keepalive time/timeout, and propagate those env vars into the job runtime so the driver, Serve replicas, and workers all use the same control-plane settings |
| SGLang overlap scheduler future-token kernel fails on ROCm | SGLang loads weights, allocates KV cache, starts Uvicorn, then the scheduler dies in `resolve_future_token_ids_cuda` with `CUDA error: no ROCm-capable device is detected` | The overlap scheduler uses the SGLang future-token JIT kernel path, which is not validated on this MI210/HIP launcher path | Add `--sglang-disable-overlap-schedule` to the AMD launcher and rerun the foreground validation |
| SGLang JIT KV-cache store kernel fails on ROCm | With overlap scheduling disabled, SGLang reaches the normal scheduler path and then dies in `kvcache.cuh:196` from `store_cache` with `CUDA error: no ROCm-capable device is detected` | SGLang's memory pool treats HIP as eligible for the optimized CUDA JIT KV-cache store path; the module can load, but the launch path is not usable on this MI210/HIP stack | Keep the Relax-side HIP runtime patch in `relax/backends/sglang/sglang_engine.py` enabled so `can_use_store_cache()` returns `False` on ROCm and SGLang uses its tensor assignment fallback |
| SGLang JIT clamp-position kernel fails on ROCm | After rollout starts decoding, SGLang scheduler dies in `clamp_position.cuh:46` with `CUDA error: no ROCm-capable device is detected`, and the router returns 503 `no_available_workers` | SGLang selects the JIT `clamp_position_cuda` helper on HIP even though the JIT launch path is not usable on this MI210 stack | Keep the Relax-side HIP runtime patch enabled so `sglang.srt.model_executor.forward_batch_info.clamp_position` is redirected to SGLang's `_clamp_position_native` torch fallback |
| Megatron actor waits on HF checkpoint page-cache warmup | The actor initializes, logs `[local_rank=1] waiting for local_rank=0 to warm HF checkpoint page cache`, and Serve keeps warning that the Actor replica is still initializing | Ray can assign the single Megatron actor to physical GPU/local rank 1 while SGLang owns GPU 0; the actor is distributed rank 0/world size 1, so no Megatron local rank 0 process exists to write the warmup marker | Treat a distributed world-size-1 Megatron process as the page-cache warmup leader even when `LOCAL_RANK != 0`; keep the marker/flock path for duplicate-job protection |
| `relax.distributed.ray.rollout` imports SGLang/Megatron too early | The rollout replica logs Megatron and SGLang warnings before any engine launch happens, and the `transformers` SGLang backend starts from an already-contaminated interpreter | Importing `sglang.srt.constants` from `rollout.py` triggers `sglang.__init__`, and `SGLangEngine` previously imported the checkpoint-service client at module scope, which pulled Megatron-backed DCS modules immediately | Mirror the small SGLang constants locally in `rollout.py` and make the checkpoint-service client import lazy in `sglang_engine.py` so importing the rollout stack does not import `sglang` or `megatron` |
| Module-import Megatron blocker kills `SGLangEngine` actor creation | `RolloutManager` dies while creating `SGLangEngine`, and Ray reports `ActorDiedError ... SGLangEngine.__init__()` with `Blocked import of megatron for SGLang transformers backend` | A module-import-time `MetaPathFinder` blocker runs before Ray has finished computing actor creation task inputs, so actor creation itself trips the blocker | At module import time, only prune local Megatron paths and editable import hooks. Install the stronger `megatron` import blocker later inside `SGLangEngine.__init__` and the spawned SGLang subprocess path |
| Rollout Serve replica unpickles Megatron enum objects before `__init__` | `ServeReplica:rollout:Rollout` dies during actor allocation with `ModuleNotFoundError: Blocked import of megatron.core.transformer.enums for SGLang transformers backend`, and the traceback points to `cloudpickle.loads(serialized_init_args)` inside Ray Serve replica construction | The rollout deployment payload was still carrying at least one Megatron enum object in `config` (for example `attention_backend=AttnBackend.auto`), so deserializing the rollout replica init args imported `megatron.core.transformer.enums` before `Rollout.__init__` could run | Keep the rollout-only module-import blocker, but sanitize the rollout service config before binding the Serve deployment by converting enum-valued args to plain serializable values (e.g. `.value`) so Ray can deserialize init args without importing Megatron |
| DCS package init pulls Megatron into control-plane imports | `DCSCoordinator`, `HealthStatus`, or `RolloutManager` emit Megatron/TE warnings before rollout engines are even created | Importing `relax.distributed.checkpoint_service.coordinator.service` first runs `relax.distributed.checkpoint_service.__init__`, which eagerly re-exported backend/client symbols and imported `backends.device_direct`, pulling in `from megatron.core import mpu` | Make `relax.distributed.checkpoint_service.__init__` and `relax.distributed.checkpoint_service.backends.__init__` lazy so coordinator-service imports stay control-plane only |
| `core.registry` imports Megatron through the advantages service | `HealthStatus`, `Rollout`, or `RolloutManager` still emit Megatron/TE warnings even after the DCS package leak is fixed, and plain `import relax.core.controller` or `import relax.core.registry` already pulls in Megatron | `relax.core.registry` eagerly imports `relax.components.advantages`, and `advantages.py` was importing `megatron.core.mpu` plus `relax.backends.megatron.loss` at module scope | Move those Megatron imports inside the PPO and OPD branches in `Advantages.compute_advantages_and_returns()` so importing the controller/registry stack stays lightweight |
| Rollout-side SGLang bootstrap starts before Megatron isolation is active | `Rollout` and `RolloutManager` still emit Megatron/TE warnings even after the controller and DCS imports are cleaned, especially when loading `relax.engine.rollout.sglang_rollout` or preparing SGLang helpers | The stronger Megatron isolation was only installed inside `SGLangEngine`, but `RolloutManager.__init__` loads the rollout function and rollout-side SGLang helpers earlier, and the Serve `Rollout` replica also starts with the unfiltered `Megatron-LM` checkout on `PYTHONPATH` | Install the same transformers-mode process isolation at the start of `Rollout.__init__` and `RolloutManager.__init__`, before loading rollout functions or creating rollout engines |
| Phase-1 timeout leaves a stale Ray training job behind | A `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh` validation appears to finish, but `python3 -m relax.entrypoints.train` and Ray workers remain alive and contaminate the next run with stale workers and mixed job state | The external shell timeout kills the launcher shell, not the full Ray job tree, so the validation cluster can keep running in the background unless it is explicitly stopped | After any timed-out foreground validation, explicitly run `ray stop --force` and kill leftover Relax train/launcher processes before starting the production tmux run |
| ROCm torch-dist checkpoint save dies after writing `common.pt` | Older MI210 runs reached `saving checkpoint at iteration N`, created only `iter_*/common.pt`, then the `MegatronTrainRayActor` died with Ray EOF / `SYSTEM_ERROR` and no Python traceback | The active Megatron torch-dist path needed the ROCm hook to patch the torch-strategy writer alias, avoid the pre-MCore 0.14 metadata path on PyTorch >= 2.6, disable PyTorch DCP's extra sharded-tensor flatten traversal, and stream GPU tensors to CPU one tensor at a time | Keep checkpointing enabled with the default `CKPT_FORMAT=torch_dist`. Verify logs contain `flatten_sharded_tensors=False`, `thread-local checkpoint results queue`, and `ROCm streaming checkpoint write`; the validated run saved `.metadata`, two `.distcp` shards, `common.pt`, `metadata.json`, and `latest_checkpointed_iteration.txt` |
| ROCm lazy torch-dist writer fails with `cannot unpack non-iterable WriteItem object` | The checkpoint save logs `ROCm lazy checkpoint prepare`, then fails in Python before writing bucket contents | Lazy DCP preparation stores raw `WriteItem` objects, but the streaming writer still treated bucket entries as already-resolved `(write_item, tensor)` pairs | Pass the planner and lazy marker through `get_save_function_and_args()`, avoid unpacking lazy entries in pre-write logging, and call `planner.resolve_data(write_item)` inside `_write_streaming_bucket()` one item at a time |
| Existing checkpoint directory does not resume when used only as `SAVE_DIR` | A launch pointed at an existing `SAVE_DIR` starts `Actor initialized with starting step 0` and begins rollout 0 again | `SAVE_DIR` only maps to Megatron `--save`; fresh-launch resume requires Megatron `--load`, and Relax recovery helpers do not infer `--load` from the save path | Set `LOAD_DIR=/path/to/checkpoint` when resuming. The AMD launcher appends `--load "${LOAD_DIR}"` only when `LOAD_DIR` is non-empty |
| PyTorch 2.6 rejects Megatron `common.pt` during `torch_dist` resume | Explicit `LOAD_DIR` reaches `_load_global_dist_base_checkpoint`, then fails with `_pickle.UnpicklingError` and `Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig` | PyTorch 2.6 changed `torch.load` to default `weights_only=True`; Megatron's `common.pt` contains trusted non-tensor metadata | Patch Megatron's `TorchCommonLoadStrategy.load_common` through the Relax ROCm checkpoint hook so HIP loads local trusted `common.pt` with `weights_only=False` |
| ROCm HDO resume fails with missing `init_state_fn` | Explicit `LOAD_DIR` reaches `FP32Optimizer.sharded_state_dict(is_loading=True)` and raises `TypeError: 'NoneType' object is not callable` | The ROCm CPU-offload path wraps `HybridDeviceOptimizer` in `FP32Optimizer`, but Megatron created that wrapper without the Adam-state initializer needed during checkpoint load | Keep `install_hybrid_device_optimizer_init_state_fn()` active for single-rank HIP actor CPU offload; it installs a fail-loud Adam initializer before Megatron loads optimizer state |
| ROCm HDO optimizer restore dies after `checkpoint version 3.0` | Resume passes common-state load and prints `checkpoint version 3.0`, then the Ray actor exits with `SYSTEM_ERROR` before reporting its restored step | PyTorch's generic optimizer load path maps checkpoint state onto HDO public param groups, which are live GPU model params, instead of HDO inner CPU-offload params | Keep the Relax ROCm HDO load patch enabled so `FP32Optimizer.load_state_dict` loads state onto inner params, syncs HDO sub-optimizer state, and restores public param groups afterward |
| Megatron scheduler rejects a resume with changed `NUM_ROLLOUT` | Explicit resume loads optimizer state, then fails while loading `opt_param_scheduler` with `class input value ... and checkpointvalue ... for total number of iterations do not match` | `NUM_ROLLOUT`, rollout batch size, or samples per prompt changed the current LR/WD schedule horizon relative to the checkpoint | Keep `SCHEDULER_RESUME_POLICY=strict` by default. For intentional continuation with a new horizon use `SCHEDULER_RESUME_POLICY=override`; to keep checkpoint scheduler values use `SCHEDULER_RESUME_POLICY=checkpoint` |
| Step-0 `dapo` reward stalls inside the Ray reward-worker pool | The run reaches `Actor training step 0/200` and rollout generation, but the Megatron actor only logs `start to get rollout_id: 0 data from transfer queue for train with mcore.` and then dies much later with `ActorDiedError`, while `RolloutManager` only logs `RewardExecutor: created 16 RewardWorker actors` before a long silent gap | `dapo` is a small local string/regex reward, but it was still being dispatched through the generic `RewardWorker` actor pool. On the MI210 step-0 path, that turned a cheap per-sample reward into extra Ray actor RPCs and left rollout blocked before it could transfer `train_0` to the actor | Keep `dapo` on a local in-process path using `asyncio.to_thread` and reserve the Ray reward-worker pool for the heavier/thread-unsafe reward types. Revalidate from a clean Ray cluster so the 5-minute timeout window no longer wedges at `RewardExecutor: created ... RewardWorker actors` |

## Megatron scheduler rejects a resume with changed `NUM_ROLLOUT`

**Added:** 2026-05-30
**Domain:** research

### Symptom

A resume run loads the model and optimizer state, prints `checkpoint version
3.0`, and then fails while loading `opt_param_scheduler`:

```text
OptimizerParamScheduler: class input value 48 and checkpointvalue 32 for total number of iterations do not match
```

In the validated failure, the checkpoint came from `NUM_ROLLOUT=2`, while the
continuation smoke used `NUM_ROLLOUT=3` to force one more post-resume training
step.

### Cause

Relax derives Megatron scheduler horizon from rollout configuration:

```text
train_iters = num_rollout * rollout_batch_size * n_samples_per_prompt / global_batch_size
lr_decay_steps = train_iters * global_batch_size
```

Changing `NUM_ROLLOUT`, `rollout_batch_size`, `n_samples_per_prompt`, or
`global_batch_size` changes the current scheduler values. Megatron rejects the
checkpoint unless the run explicitly chooses whether to keep the checkpoint
values or override them with the new run's values.

### Solution

The AMD launcher exposes an explicit scheduler resume policy:

```bash
SCHEDULER_RESUME_POLICY=strict      # default Megatron mismatch check
SCHEDULER_RESUME_POLICY=override    # append --override-opt-param-scheduler
SCHEDULER_RESUME_POLICY=checkpoint  # append --use-checkpoint-opt-param-scheduler
```

Use `override` only when the schedule change is intentional, for example when
extending a short validation checkpoint from `NUM_ROLLOUT=2` to
`NUM_ROLLOUT=3`. Use `checkpoint` when the resumed run should keep the old
schedule values.

### Prevention

Resume validation should first run with matching scheduler-driving args. If a
test deliberately extends the horizon, set `SCHEDULER_RESUME_POLICY=override`
and verify the launched command includes `--override-opt-param-scheduler`.

## ROCm HDO optimizer restore dies after `checkpoint version 3.0`

**Added:** 2026-05-30
**Domain:** research

### Symptom

After fixing `common.pt` loading and the missing HDO initializer, explicit
`LOAD_DIR` resume reaches Megatron optimizer state loading:

```text
Installed HybridDeviceOptimizer Adam-state initializer for ROCm checkpoint restore
Initialized 290 HybridDeviceOptimizer Adam states for ROCm checkpoint restore
loading distributed checkpoint from ... at iteration 1
checkpoint version 3.0
```

Then the `MegatronTrainRayActor` dies with Ray `SYSTEM_ERROR` before the
`Actor` service reports its restored starting step. There may be no useful
Python traceback because the worker exits during optimizer state placement.

### Cause

The HDO public optimizer param groups point at the live model parameters on the
GPU. HDO stores the offloaded Adam state on its inner CPU params. PyTorch's
generic `Optimizer.load_state_dict()` does not know about that mapping, so
calling it on the HDO object can load the restored `exp_avg` and `exp_avg_sq`
state onto the public GPU params. On the MI210 actor, that can kill the worker
immediately after Megatron reports `checkpoint version 3.0`.

### Solution

Keep the Relax-side HDO load patch in
`relax/backends/megatron/optimizer_utils.py`. For single-rank HIP actor CPU
offload, `install_hybrid_device_optimizer_init_state_fn()` now also patches the
`FP32Optimizer.load_state_dict` method so checkpoint state loads onto HDO inner
params, then synchronizes the HDO sub-optimizers and restores the public param
groups.

The validated success signature is:

```text
Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore
loading distributed checkpoint from ... at iteration 1
Initialized 290 HybridDeviceOptimizer Adam states for ROCm checkpoint restore
checkpoint version 3.0
Actor initialized with starting step 2
All training steps finished
Job 'raysubmit_KiRTCAx7v8RCU9L6' succeeded
```

### Prevention

Do not work around this by disabling optimizer checkpoint save/load. The
validated path uses `NO_SAVE_OPTIM=0` and keeps Megatron `torch_dist`
optimizer state enabled. If this boundary regresses, inspect HDO inner-param
state placement before changing checkpoint intervals or rollout logic.

## ROCm HDO resume fails with missing `init_state_fn`

**Added:** 2026-05-30
**Domain:** research

### Symptom

Explicit `LOAD_DIR` resume gets past the PyTorch 2.6 `common.pt` issue but
then fails during Megatron optimizer state-dict construction:

```text
TypeError: 'NoneType' object is not callable
...
FP32Optimizer.sharded_state_dict(...)
self.init_state_fn(self.optimizer, self.config)
```

### Cause

The ROCm safe CPU-offload path disables Megatron's mixed-precision optimizer
wrappers that are unsafe for this single-rank MI210 actor path, but Megatron
still wraps `HybridDeviceOptimizer` in `FP32Optimizer` for checkpoint load.
That wrapper can be created with `init_state_fn=None`. During load,
Megatron asks the optimizer for a sharded state dict with `is_loading=True`,
and `FP32Optimizer` calls the missing initializer.

### Solution

Install the Relax-side HDO initializer after optimizer construction:

```python
install_hybrid_device_optimizer_init_state_fn(optimizer, args, role)
```

The initializer is intentionally narrow: actor role only, single data-parallel
rank, HIP runtime, CPU optimizer offload enabled, and Adam optimizer only. For
non-Adam optimizers it fails loudly instead of inventing incompatible state.

### Prevention

Resume tests must grep for:

```text
Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore
Initialized 290 HybridDeviceOptimizer Adam states for ROCm checkpoint restore
```

If these markers are absent on the MI210 single-rank HDO path, expect resume to
fail before the actor reports the restored step.

## PyTorch 2.6 rejects Megatron `common.pt` during `torch_dist` resume

**Added:** 2026-05-30
**Domain:** research

### Symptom

After adding explicit `LOAD_DIR`, the resume path reaches Megatron
`torch_dist` checkpoint loading and fails before the actor can finish
initialization:

```text
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig
```

The traceback points through
`_load_global_dist_base_checkpoint()` -> `load_common_state_dict()` ->
`TorchCommonLoadStrategy.load_common()` -> `torch.load(common.pt)`.

### Cause

PyTorch 2.6 changed `torch.load` so `weights_only=True` is the default.
Megatron's `common.pt` is a trusted local checkpoint metadata file, not only a
plain tensor payload; it can include OmegaConf `DictConfig` objects and other
Megatron metadata that the safe weights-only unpickler rejects.

### Solution

Keep this in the Relax ROCm checkpoint hook rather than editing the local
Megatron checkout silently. On HIP,
`patch_rocm_checkpoint_writer()` patches Megatron's
`TorchCommonLoadStrategy.load_common()` so local trusted `common.pt` files load
with:

```python
torch.load(load_path, map_location="cpu", weights_only=False)
```

This patch is deliberately scoped to the Megatron checkpoint common-state load
path. It does not change arbitrary `torch.load` calls.

### Prevention

Any future PyTorch >= 2.6 `torch_dist` resume validation should check for the
log marker:

```text
HIP/ROCm detected: loading Megatron common.pt with weights_only=False
```

If that marker is absent and the checkpoint contains OmegaConf metadata, expect
resume to fail before the actor reports its restored starting step.

## Existing checkpoint directory does not resume when used only as `SAVE_DIR`

**Added:** 2026-05-30
**Domain:** research

### Symptom

After a successful live `torch_dist` checkpoint run, a follow-up launch pointed
`SAVE_DIR` at the validated checkpoint directory and expected the actor to
resume. The actor instead initialized from the base HuggingFace/ref load path
and started from step 0 again:

```text
Actor initialized with starting step 0
Starting rollout step 0
```

### Cause

`SAVE_DIR` only controls Megatron `--save`. It is not a resume signal. The
fresh launcher path does not infer `--load` from the save path, and
`recovery_load_path()` only applies to the restart/recovery flow rather than a
new shell launch.

### Solution

Use the explicit `LOAD_DIR` launcher variable when resuming from an existing
checkpoint:

```bash
LOAD_DIR=/path/to/checkpoint SAVE_DIR=/path/to/checkpoint \
  SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=2 \
  bash ./amd_qwen3_4b_2gpu_e2e.sh
```

The AMD launcher appends `--load "${LOAD_DIR}"` only when `LOAD_DIR` is
non-empty. Keep `SAVE_DIR` and `LOAD_DIR` separate so a run can save to a new
directory while loading from an older checkpoint when needed.

### Prevention

When checking resume behavior, grep the launched command for `--load` and
verify the actor starts at `latest_checkpointed_iteration + 1`. If the actor
starts at step 0, stop the run before it can overwrite the checkpoint directory.

## ROCm lazy torch-dist writer fails with `cannot unpack non-iterable WriteItem object`

**Added:** 2026-05-30
**Domain:** research

### Symptom

The ROCm checkpoint hook reaches lazy DCP planning and then fails in Python:

```text
ROCm lazy checkpoint prepare: write_items=1123, buckets=2
ROCm streaming checkpoint write failed: cannot unpack non-iterable WriteItem object
CheckpointException ranks:dict_keys([0])
```

### Cause

The lazy writer intentionally keeps bucket payloads as raw DCP `WriteItem`
objects so tensors are not resolved and staged up front. The first streaming
implementation still assumed `tensor_data` contained `(write_item, tensor)`
pairs when logging bucket sizes and when deciding how to write the bucket. That
local mismatch replaced the original native Ray-worker death with a fixable
Python exception.

### Solution

1. Keep lazy `prepare_write_data()` on HIP so DCP planning does not resolve all
   tensors before the write starts.
2. Store the planner and lazy marker on the writer and pass both through
   `get_save_function_and_args()`.
3. In `write_data_streaming()`, do not pre-unpack lazy `WriteItem` entries for
   tensor-byte logging; log `tensor_bytes_gib=unresolved`.
4. In `_write_streaming_bucket()`, require a planner when `lazy_write_items` is
   true and call `planner.resolve_data(write_item)` immediately before each
   item write.

### Prevention

Unit-test the writer at both levels: the top-level `write_data_streaming()`
should accept lazy `WriteItem` buckets without pre-unpacking, and
`_write_streaming_bucket()` should fail loudly if lazy items are used without a
planner.

## ROCm torch-dist checkpoint save previously died after writing `common.pt`

**Added:** 2026-05-29
**Domain:** research

### Symptom

The long MI210 W&B run reached repeated real training and then died at the
checkpoint boundary:

```text
saving checkpoint at iteration      99 to .../Qwen3-4B_mcore_2gpu in torch_dist format
Overwriting old incomplete / corrupted checkpoint...
```

The dead checkpoint directory contained only `iter_0000099/common.pt`; the
sharded torch-dist checkpoint files and metadata were missing. Ray reported the
`MegatronTrainRayActor` worker death as EOF / `SYSTEM_ERROR`, while the driver,
rollout service, and SGLang health checks stayed alive and rollout kept polling
`train_99`.

### Cause

Relax first patched
`megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync`.
That was not enough for the active torch-distributed save path because
Megatron's `strategies.torch` module imports `FileSystemWriterAsync` from
`filesystem_async` at module import time and uses the cached alias inside
`TorchDistSaveShardedStrategy.async_save()`.

After patching both aliases, the failure still reproduced with
`SAVE_INTERVAL=2 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=1`. Later debugging
narrowed the remaining death to Megatron's torch-dist planning path: the active
ROCm checkout did not use the newer upstream planner setting
`flatten_sharded_tensors=False`, so PyTorch DCP ran an unnecessary sharded-tensor
flatten traversal before write preparation. On HIP, the save path also needs
blocking tensor staging and a thread-local result queue so checkpoint writing
does not fork/spawn around pinned GPU-staging state.

### Solution

1. Keep checkpointing enabled in `amd_qwen3_4b_2gpu_e2e.sh`: pass `--save`,
   `--save-interval`, and `--ckpt-format`.
2. Use the launcher default `CKPT_FORMAT=torch_dist` on the MI210 path.
3. Keep `relax.utils.rocm_checkpoint_writer.patch_rocm_checkpoint_writer()` in
   place. It patches both Megatron writer aliases, forces the current
   torch-dist metadata path on PyTorch >= 2.6, disables
   `flatten_sharded_tensors` for Megatron DCP planners on HIP, and streams
   checkpoint tensors through blocking per-tensor CPU staging.
4. The `torch_dist` path was first validated on 2026-05-30 with a cached
   rollout smoke. It was then validated on a live non-cached two-rollout run:
   Ray job `raysubmit_FeKYagrKwzrfrcPU` in `tmux-13` completed actor steps 0
   and 1, saved iterations 0 and 1, included optimizer state in DCP metadata,
   and exited successfully.

### Prevention

For MI210 validation, keep the checkpoint log markers in the smoke criteria:
`HIP/ROCm detected: setting flatten_sharded_tensors=False`, `thread-local
checkpoint results queue`, `ROCm streaming checkpoint write`, and
`successfully saved checkpoint`. Also verify `.metadata`, `.distcp`,
`metadata.json`, `common.pt`, and `latest_checkpointed_iteration.txt` exist.

## Megatron actor waits on HF checkpoint page-cache warmup

**Added:** 2026-05-29
**Domain:** research

### Symptom

The post-SGLang ROCm run reaches Megatron actor initialization and then stops
making progress after:

```text
[local_rank=1] waiting for local_rank=0 to warm HF checkpoint page cache
```

Ray still shows the `MegatronTrainRayActor` alive, SGLang health checks keep
succeeding, and Serve repeatedly reports that the Actor replica is taking more
than 30 seconds to initialize.

### Cause

On the 2-GPU MI210 launcher, SGLang can own GPU 0 while the single Megatron
training actor owns GPU 1. That actor is distributed rank 0 with world size 1,
but its environment exposes `LOCAL_RANK=1`. The previous warmup logic treated
any nonzero `LOCAL_RANK` as a follower and waited for a Megatron local rank 0
process that did not exist in the actor process group.

### Solution

1. In `relax/backends/megatron/checkpoint.py`, treat
   `torch.distributed.get_world_size() == 1` as a warmup-leader signal even
   when `LOCAL_RANK != 0`.
2. Keep the `/dev/shm` marker and `flock` around the leader path so duplicate
   jobs or multiple single-rank actors still avoid redundant NFS reads.
3. Re-run the foreground validation and confirm the next log is page-cache
   warming or checkpoint loading rather than a repeated wait message.

### Prevention

Do not use physical GPU-local rank as the only ownership signal inside a Ray
actor. In single-process actor paths, prefer the distributed world-size/rank
contract for actor-local coordination.

## Packed sequence with DotProductAttention on ROCm

**Added:** 2026-04-15
**Domain:** research

### Symptom

The run starts normally, then fails once Megatron enters the training path:

```text
AssertionError: Packed sequence is not supported by DotProductAttention.Please use TEDotProductAttention instead.
```

### Cause

The AMD path reached plain Megatron `DotProductAttention` rather than a Transformer Engine attention implementation. The packed-sequence THD path is not supported there, so the run fails only after the system has already brought up Ray, Serve, and SGLang.

### Solution

1. Force the launcher away from the packed-sequence path with `--qkv-format bshd`.
2. Re-run the 2-GPU e2e launcher and confirm the failure boundary moves past rollout generation.
3. Keep this as the default on MI210 unless a TE-capable ROCm attention path is validated separately.

### Prevention

On ROCm bring-up, avoid assuming the default attention format used on NVIDIA is safe. Validate the attention layout explicitly in the launcher before starting long retry loops.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## Forcing fp32 CPU-offload params can exhaust host memory before step 0 completes

**Added:** 2026-04-20
**Domain:** research

### Symptom

The MI210 actor now survives dozens of sequential CPU sub-optimizers and then
dies partway through the first real optimizer update:

```text
HybridDeviceOptimizer step: cpu sub-optimizer 51 grad sync done
HybridDeviceOptimizer step: cpu sub-optimizer 51 step begin
```

There is no matching `step done` line, and Ray only reports EOF /
`SYSTEM_ERROR`. A follow-up validation that eagerly preinitialized all AdamW
state made the actor die even earlier during init, after logging many large
allocations such as:

```text
Preinitialized TorchCPUAdamW state for 1 params (380.00 MiB)
```

### Cause

The ROCm CPU-offload path had already disabled foreach, pinned buffers, and
overlap, but `torch.optim.AdamW` still initialized optimizer state lazily
inside `step()`. The follow-up eager-preinit experiment showed the deeper issue:
forcing the offloaded CPU parameters into fp32 on this MI210 safe path makes
both the copied CPU weights and the AdamW state too large. The actor can then
die either late in `cpu_optimizer.step()` or earlier during init if all state is
allocated up front.

### Solution

1. Keep the Relax ROCm CPU-offload wrapper (`TorchCPUAdamW`) on the simplest
   non-foreach path.
2. On the single-rank ROCm safe path (`pin_cpu_grads=False`,
   `pin_cpu_params=False`, `overlap_cpu_optimizer_d2h_h2d=False`), disable
   `param_update_in_fp32` inside the local Megatron `HybridDeviceOptimizer` so
   the offloaded CPU params stay in bf16.
3. Let CPU AdamW keep bf16 state tensors for those bf16 CPU params instead of
   forcing fp32 CPU copies and fp32 optimizer state.
4. Revalidate from a clean cluster and confirm the actor gets past the old
   step-0 optimizer boundary without reintroducing the earlier init-time crash.

### Prevention

When the actor dies deep inside the first CPU offload optimizer step with no
Python traceback, check whether the offloaded CPU params are being widened to
fp32. On this MI210 path, cutting persistent CPU optimizer memory mattered more
than moving allocation earlier.

### Related

- Code: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`, `relax/backends/megatron/optimizer_utils.py`
- Tests: `tests/utils/test_megatron_model.py`
- Experiment log: `references/experiment-log.md`

## ROCm CPU-offload optimizer step needs bf16-safe Megatron wrappers

**Added:** 2026-04-24
**Domain:** research

### Symptom

After moving the actor onto the ROCm Megatron checkout, the run reaches rollout,
actor weight sync, forward/backward, and the first optimizer boundary, but then
fails in one of three ways:

```text
HybridDeviceOptimizer step: cpu sub-optimizer 51 step begin
```

or:

```text
RuntimeError: attempting to assign a gradient with dtype 'float' to a tensor with dtype 'c10::BFloat16'
```

or:

```text
AssertionError
... clip_grads.py ... assert param.grad.type() == 'torch.cuda.FloatTensor'
```

### Cause

The single-rank MI210 path needs CPU optimizer offload to fit, but the default
Megatron mixed-precision optimizer stack keeps trying to widen or validate the
path as fp32:

1. `HybridDeviceOptimizer` can be built over fp32 main params instead of the
   bf16 model params, making the offloaded CPU AdamW state too large and slow.
2. `FP32Optimizer.prepare_grads()` assigns fp32 `main_grad` tensors directly to
   bf16 live params, which PyTorch rejects.
3. `clip_grad_by_total_norm_fp32()` asserts that every grad is
   `torch.cuda.FloatTensor`, which rejects the bf16 CUDA grads produced by the
   safe path.

### Solution

1. In the Relax actor path, only for single-rank ROCm CPU offload, set the
   optimizer config to avoid the fp32/bf16 mixed wrapper fields that rebuild HDO
   over fp32 main params.
2. Keep `refresh_hybrid_device_optimizer_param_groups()` from rebuilding an HDO
   instance wrapped by `FP32Optimizer`.
3. In the local ROCm Megatron checkout, keep HDO CPU copies in bf16 when the
   safe path is unpinned and non-overlapped.
4. In `FP32Optimizer.prepare_grads()`, cast `main_grad` to the target parameter
   dtype before assigning `param.grad`.
5. In `clip_grad_by_total_norm_fp32()`, accept CUDA bf16 grads in addition to
   CUDA fp32 grads.

The validation log `log/amd-qwen3-4b-2gpu-20260424_094405.log` confirms this
path completes multiple actor optimizer steps on the 2-GPU MI210 foreground
run.

### Prevention

When debugging ROCm CPU offload, inspect the actual CPU-copy dtype in HDO logs
or direct smokes. A launcher flag can look correct while Megatron's optimizer
wrappers have already rebuilt the offload path around fp32 main params.

### Related

- Code: `relax/backends/megatron/optimizer_utils.py`, `relax/backends/megatron/model.py`
- ROCm Megatron:
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`,
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/optimizer.py`,
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/clip_grads.py`
- Tests: `tests/utils/test_megatron_model.py`
- Experiment log: `references/experiment-log.md`

## Disabling CPU optimizer offload is a valid MI210 diagnostic branch

**Added:** 2026-04-21
**Domain:** research

### Symptom

After multiple ROCm safety patches, the long run can still die at:

```text
HybridDeviceOptimizer step: cpu sub-optimizer 51 step begin
```

with the actor disappearing underneath Ray and rollout wedging forever on
`train_0`.

### Cause

At that point the remaining narrow boundary is no longer generic rollout or Ray
startup. It is specifically the Megatron CPU-offload optimizer implementation
on the single-rank MI210 actor path.

### Solution

1. Remove `--optimizer-cpu-offload`, `--optimizer-offload-fraction`,
   `--overlap-cpu-optimizer-d2h-h2d`, and
   `--use-precision-aware-optimizer` from the AMD launcher.
2. Re-run the clean 5-minute foreground gate first.
3. Use the result to classify the next boundary:
   - if the actor now dies with HIP OOM, then CPU offload was required for fit
   - if the actor gets past `optimizer.step()`, then CPU offload was the real
     bottleneck

### Prevention

When a run consistently reaches real step-0 training and then dies inside
`HybridDeviceOptimizer`, stop broad debugging and branch the launcher into a
no-offload diagnostic run before changing more Ray or rollout code.

### Related

- Code: `amd_qwen3_4b_2gpu_e2e.sh`
- Experiment log: `references/experiment-log.md`

## SGLang init failure can leak orphaned scheduler processes and poison the next run

**Added:** 2026-04-20
**Domain:** research

### Symptom

A fresh 5-minute validation fails during rollout startup with:

```text
Exception: Server process terminated unexpectedly.
...
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 48.00 MiB.
```

At the same time, `/dev/kfd` is still held by old `sglang::scheduler` /
`sglang::detokenizer` processes from earlier failed runs.

### Cause

When `SGLangEngine.init()` launches the HTTP server and `_wait_server_healthy()`
raises before the server becomes ready, `launch_server_process()` used to
re-raise without first killing the spawned SGLang process tree. That left
orphaned scheduler / detokenizer descendants alive after the Ray actor failed.
Those stale workers kept GPU memory allocated, so the next rollout startup
OOMed during `Load weight begin` even though the new worker itself had only a
small amount of PyTorch memory allocated.

### Solution

1. In `launch_server_process()`, kill the full spawned process tree if
   `_wait_server_healthy()` fails.
2. In the AMD launcher, remove orphaned `sglang::scheduler` /
   `sglang::detokenizer` trees whose parent is an orphaned
   `multiprocessing.spawn` worker before starting Ray.
3. Revalidate from a clean cluster and confirm `/dev/kfd` no longer shows stale
   SGLang GPU holders before the next run starts.

### Prevention

Treat `Server process terminated unexpectedly` as both a child-traceback problem
and a process-lifecycle problem. If the next run OOMs unexpectedly early,
inspect `/dev/kfd` holders and orphaned SGLang processes before assuming the
model no longer fits.

### Related

- Code: `relax/backends/sglang/sglang_engine.py`, `amd_qwen3_4b_2gpu_e2e.sh`
- Tests: `tests/backends/sglang/test_sglang_engine.py`
- Experiment log: `references/experiment-log.md`

## W&B Secondary Shared Init Can Kill Actor Startup on MI210

### Symptom

The run reaches rollout readiness and then the actor replica fails during `MegatronTrainRayActor.init()` with a W&B communication error:

```text
wandb.errors.errors.CommError: Error uploading run: net/http: request canceled
```

Ray then restarts the actor replica and the 5-minute validation never reaches real training.

### Cause

The primary train process already creates the W&B run and uploads full config. Secondary shared-mode clients in rollout and actor processes were also sending `config=args.__dict__` during `wandb.init()`. On this MI210 path that secondary payload was large enough or slow enough to trigger W&B upload timeouts during actor initialization.

### Solution

1. Keep full config upload only in `init_wandb_primary()`.
2. In `init_wandb_secondary()`, attach to the existing run by ID only.
3. Preserve shared-mode settings, resume behavior, and optional router metrics forwarding.
4. Re-run the clean 5-minute validation and confirm the actor gets past init and into:
   - `Actor training step 0/200`
   - rollout generation
   - SGLang decode activity

### Prevention

In shared W&B mode, treat primary and secondary clients differently. The primary process owns run creation and config upload; secondary workers should only attach and emit metrics.

### Related

- Code: `relax/utils/metrics/adapters/wandb.py`
- Tests: `tests/utils/test_wandb_adapter.py`
- Experiment log: `references/experiment-log.md`

## W&B `train/step` stays at zero with MetricsService

### Symptom

The live training log shows actor steps advancing and metrics being reported:

```text
step 34: {..., 'train/step': 34}
Reported 66 metrics for step 34
```

But the W&B UI shows `train/step` as only `0` or leaves train charts on a
stale custom x-axis.

### Cause

`MetricsServiceAdapter.log()` used the configured `step_key` only as transport
metadata and removed it from the metric payload before sending the batch to the
MetricsService. That was correct for plain `"step"`, but not for W&B custom
axis metrics such as `train/step`, `rollout/step`, or `eval/step`.

The MetricsService defines:

```python
wandb.define_metric("train/step")
wandb.define_metric("train/*", step_metric="train/step")
```

If `train/step` is stripped from the payload, W&B receives train metrics at the
global SDK step but not the named custom step metric it uses for `train/*`.

### Solution

Preserve namespaced step metrics ending in `/step` in the MetricsService
payload, while continuing to remove the generic plain `"step"` helper key.

### Prevention

When adding metrics-service transport logic, keep W&B custom step metrics as
first-class metrics. The transport step and the named metric step can carry the
same integer, but the named metric must still reach W&B.

### Related

- Code: `relax/utils/metrics/metrics_service_adapter.py`
- Tests: `tests/utils/test_metrics_service.py`
- Experiment log: `references/experiment-log.md`

## Grouped CPU sub-optimizers can still crash inside `optimizer.step()` on MI210

## ROCm Megatron fork experiment should be isolated in a sibling Relax copy

**Added:** 2026-04-21
**Domain:** research

### Symptom

The baseline MI210 workspace has accumulated many Relax-side AMD patches while
still depending on the shared NVIDIA `Megatron-LM` checkout. Repointing that
baseline in place makes it hard to tell whether a regression came from the
Megatron fork or from local workspace drift.

### Cause

The launcher used an absolute shared Megatron path:

```text
/vast/users/qirong.ho/erland/Python_project/Megatron-LM
```

so trying a ROCm fork in-place would implicitly mutate the shared backend for
other runs too.

### Solution

1. Preserve the current `Relax/` directory as the baseline workspace.
2. Create a sibling copy for the fork experiment, for example
   `Relax-rocm-megatron/`.
3. Clone `ROCm/Megatron-LM` into a separate checkout, for example
   `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`.
4. In the sibling launcher, route `MEGATRON` and `PYTHONPATH` through a single
   `MEGATRON_DIR` variable that defaults to the ROCm checkout.

### Prevention

When comparing backend forks on this MI210 stack, keep the baseline workspace
and the forked-backend workspace separate. The first fork experiment should
change only the local Megatron checkout path, not the rest of the baseline.

### Related

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Experiment log: `references/experiment-log.md`

**Added:** 2026-04-20
**Domain:** research

### Symptom

The run now reaches real step-0 training and survives deep into the sequential
CPU-offload path, but the actor still dies during optimizer step:

```text
train_one_step rollout=0 step=0: starting optimizer.step
...
HybridDeviceOptimizer step: cpu sub-optimizer 12 step begin
```

There is no matching `step done` line for that sub-optimizer before Ray reports
`ActorDiedError` / EOF.

### Cause

The local Megatron `HybridDeviceOptimizer` was still grouping four parameters
into each CPU AdamW instance on the unpinned, non-overlap ROCm safety path.
That reduced per-optimizer overhead, but it still left each `cpu_optimizer.step()`
large enough to trigger the same native MI210 failure inside a grouped
sub-optimizer.

### Solution

1. Keep the ROCm fallback fully sequential.
2. Build one CPU optimizer per parameter on the unpinned, non-overlap path
   (`params_per_optimizer=1`).
3. Revalidate from a clean cluster and confirm the old grouped-sub-optimizer
   boundary no longer reproduces at the same index.

### Prevention

On this MI210 single-rank actor path, prefer smaller CPU optimizer units over
lower Python overhead. When logs already isolate the death to a specific grouped
CPU sub-optimizer, treat finer chunking as the next stability lever before
assuming the failure is elsewhere.

### Related

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Tests: `tests/utils/test_megatron_model.py`
- Experiment log: `references/experiment-log.md`

## Stale Relax train drivers survive `ray stop --force`

**Added:** 2026-04-20
**Domain:** research

### Symptom

Fresh AMD validation runs still emit confusing Ray noise such as:

```text
Mismatched WorkerID: ignoring RPC for previous worker
```

even after the previous run was supposedly cleaned up with `ray stop --force`.

### Cause

`ray stop --force` tears down the Ray cluster, but it does not reliably kill every older Relax launcher shell or `python3 -m relax.entrypoints.train` driver process that was started through `ray job submit`. Those stale drivers can survive outside Ray's process tree and contaminate later retries.

### Solution

1. Before starting Ray, kill stale Relax-only launcher processes that still reference the current e2e asset directory and command path.
2. Match only the Relax driver / `ray job submit` command lines so unrelated tmux sessions are not touched.
3. Then run `ray stop --force` and start the next validation from that clean state.

### Prevention

Do not treat `ray stop --force` alone as proof of a clean retry state on this stack. Check for surviving `relax.entrypoints.train` and `ray job submit` processes before trusting a new run boundary.

### Related

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Experiment log: `references/experiment-log.md`

## Sequential CPU optimizer stepping moves the ROCm crash boundary

**Added:** 2026-04-20
**Domain:** research

### Symptom

The run still dies at step 0 inside Megatron optimizer step, but only after getting much farther than before:

```text
train_one_step rollout=0 step=0: starting optimizer.step
...
HybridDeviceOptimizer step: cpu sub-optimizer 46 step begin
```

The worker then exits with Ray EOF / `SYSTEM_ERROR`.

### Cause

On the single-rank ROCm path with unpinned CPU grads and no D2H/H2D overlap, staging every offloaded CPU grad at once was too fragile. Splitting the step into per-parameter CPU optimizers removed the earlier first-copy death, proving the original boundary was not the first host grad copy itself.

### Solution

1. Build per-parameter CPU optimizers for the unpinned, non-overlap ROCm path.
2. Sync and step those CPU optimizers sequentially.
3. Use the resulting sub-optimizer index logs to narrow the next crash location.

### Prevention

When a ROCm optimizer death happens immediately after `starting optimizer.step`, first make the step path incremental and observable before assuming a specific tensor or copy primitive is broken.

### Related

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Skill: `megatron-hybrid-device-optimizer-rocm`
- Experiment log: `references/experiment-log.md`

## ROCm sequential CPU-offload path can die inside optimizer post-hook copy-back

**Added:** 2026-04-20
**Domain:** research

### Symptom

After sequential CPU stepping is enabled, the run still dies during `cpu_optimizer.step()` after dozens of successful CPU sub-optimizers, with the last visible line often being:

```text
HybridDeviceOptimizer step: cpu sub-optimizer <n> step begin
```

and no matching `step done`.

### Cause

For CPU-offloaded parameters, Megatron normally relies on an optimizer post-hook to copy updated CPU params back to GPU. On the single-rank ROCm sequential path, that means the dangerous boundary is still hidden inside `cpu_optimizer.step(...)`, even though the actual math may already be fine. The death can happen during the hook-managed param copy-back rather than inside AdamW math itself.

### Solution

1. On the unpinned, non-overlap ROCm path, skip registering CPU optimizer post-hooks for param copy-back.
2. After each `cpu_optimizer.step()`, explicitly copy that optimizer's params back to GPU and log the copy-back phase separately.
3. Re-run from a clean Ray cluster and look for whether the crash now moves past the old `cpu sub-optimizer 46` boundary.

### Prevention

On fragile ROCm optimizer paths, keep post-step data movement explicit and separately logged instead of burying it in optimizer hooks.

### Related

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Skill: `megatron-hybrid-device-optimizer-rocm`
- Experiment log: `references/experiment-log.md`

## 2026-04-20 - HybridDeviceOptimizer ignored unpinned-host async copy safety

### Symptom

The MI210 run now reaches real step-0 training and logs:

```text
train_one_step rollout=0 step=0: finished forward_backward
train_one_step rollout=0 step=0: starting optimizer.step
```

Then the `MegatronTrainRayActor` worker dies with:

```text
ActorDiedError
Worker exit type: SYSTEM_ERROR
Worker exit detail: connection error code 2. End of file
```

There is still no Python traceback in the dead worker log after the `starting optimizer.step` line.

### Cause

The local Megatron `HybridDeviceOptimizer` respected `pin_cpu_grads=False` and `pin_cpu_params=False` when allocating host buffers, but it still forced asynchronous host/device copies:

- GPU grad to CPU grad buffer used `copy_(..., non_blocking=True)`
- CPU param copy-back to GPU used `copy_(..., non_blocking=True)`

On this ROCm path those async copies no longer matched the actual host-buffer configuration once pinning was disabled, so the supposed ROCm-safe path still executed an unsafe transfer mode during `optimizer.step()`.

### Solution

Patch the local Megatron checkout so the async copy mode follows the pin flags:

1. In `HybridDeviceOptimizer._set_sub_optimizer_grads()`, use `non_blocking=self.pin_cpu_grads`
2. In the CPU param copy-back post-hook, use `non_blocking=self.pin_cpu_params`

That keeps the ROCm safety path internally consistent: if the run disables pinned host buffers, it also disables async host/device copies that depend on those buffers.

### Prevention

When adding ROCm safety knobs, verify downstream copy sites as well as allocation sites. It is not enough to disable `.pin_memory()` if later `copy_(..., non_blocking=True)` still assumes the memory is pinned.

### Related

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Relax safety helper: `relax/backends/megatron/optimizer_utils.py`

## Megatron actor post-load backup and ref checkpoint boundary

**Added:** 2026-04-19
**Domain:** research

### Symptom

The actor service dies during initialization with Ray `ActorDiedError` / EOF, and the last visible logs are HuggingFace checkpoint loads with no Python traceback.

### Cause

The post-load actor path had no stage-level visibility. After `initialize_model_and_optimizer()` returns, the actor immediately:

1. builds the weight backuper
2. snapshots actor weights to host memory
3. optionally loads the ref checkpoint and snapshots that state
4. creates the rollout weight updater

That made it impossible to tell whether the crash happened during the initial backup, ref backup, or updater setup.

### Solution

1. Add explicit logs around:
   - initial actor backup
   - ref / teacher / old-actor checkpoint backup
   - weight-updater creation
2. Add a `TensorBackuper` option to disable pinned host backup buffers when a caller needs a safer ROCm path.
3. Opt out of pinned host weight backups for the single-rank ROCm actor path entirely, not just `offload_train`.

### Current validation state

A clean 5-minute AMD foreground validation now survives through:

- Megatron actor initialization with `pinned_host_weight_backups=False`
- initial actor weight backup on the non-pinned host path

That removes the earlier fragile pinned-host configuration from the real actor path and keeps the 5-minute foreground gate healthy from a clean cluster.

### Related

- `relax/backends/megatron/actor.py`
- `relax/backends/megatron/optimizer_utils.py`
- `relax/utils/training/tensor_backper.py`
- `tests/utils/test_megatron_model.py`
- `tests/utils/test_tensor_backper.py`
- Experiment log: `references/experiment-log.md`

## HybridDeviceOptimizer ignored pin_cpu_params on ROCm

**Added:** 2026-04-20
**Domain:** research

### Symptom

The run gets past actor initialization, rollout generation, and the first `forward_backward`, then wedges at:

```text
train_one_step rollout=0 step=0: starting optimizer.step
```

even though the Relax-side actor path explicitly set:

```text
pin_cpu_params=False
pin_cpu_grads=False
```

### Cause

The local Megatron `HybridDeviceOptimizer` did not actually honor `pin_cpu_params=False`. In `_get_sub_optimizer_param_groups()`, it always materialized offloaded parameter copies with:

```python
param.detach().clone().cpu().pin_memory()
```

That meant the ROCm actor still used pinned host parameter buffers inside the optimizer path, even after the Relax-side safety knobs disabled them.

### Solution

Patch the local Megatron checkout so `HybridDeviceOptimizer` only calls `.pin_memory()` when `self.pin_cpu_params` is true.

### Prevention

When a local safety knob is meant to disable a memory behavior on ROCm, verify that the downstream implementation actually consumes that knob. Do not assume the config flag is wired through correctly just because argument parsing accepts it.

### Related

- Local Megatron file: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Relax safety helper: `relax/backends/megatron/optimizer_utils.py`
- Experiment log: `references/experiment-log.md`

## Single-rank ROCm actor dies inside CPU-offload optimizer.step

**Added:** 2026-04-19
**Domain:** research

### Symptom

The MI210 run gets through rollout generation, actor setup, and Megatron forward/backward, then dies immediately after:

```text
train_one_step rollout=0 step=0: starting optimizer.step
```

The actor-side Ray worker exits with EOF / `ActorDiedError`, and rollout spins forever waiting on the orphaned `train_0` partition.

### Cause

On this single-rank ROCm actor path, the optimizer still uses Megatron's CPU-offload `HybridDeviceOptimizer`. That path clones parameters to CPU, uses pinned host buffers, and defaults to Megatron's `CPUAdam` implementation. On MI210, that combination can die natively during `optimizer.step()` even after forward/backward succeeds.

### Solution

1. Keep CPU optimizer offload enabled so actor memory still fits on MI210.
2. For single-rank ROCm actor training only:
   - disable `pin_cpu_grads`
   - disable `pin_cpu_params`
   - force `use_torch_optimizer_for_cpu_offload`
3. Patch Megatron's CPU-offload builder at runtime so the CPU side uses a torch `AdamW` wrapper that strips Megatron-only kwargs such as `bias_correction` and `fused`.
4. Re-run the focused Megatron optimizer tests and the AMD foreground validation.

### Prevention

Treat single-rank ROCm CPU-offload as a separate safety mode from the default CUDA-oriented Megatron path. If the actor reaches `starting optimizer.step` and then disappears without a Python traceback, prefer a simpler torch CPU optimizer and non-pinned host buffers before changing the training loop itself.

### Related

- Optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Model setup: `relax/backends/megatron/model.py`
- Tests: `tests/utils/test_megatron_model.py`
- Experiment log: `references/experiment-log.md`

## Step-0 MI210 crash needs saved replay inputs before entering Megatron train

**Added:** 2026-04-18
**Domain:** research

### Symptom

The run survives long enough to:

- build actor and rollout services
- finish reward execution
- transfer `train_0`
- log rollout statistics on the actor side

and then the actor dies during step 0 with:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file
```

There is still no Python traceback from the worker before death.

### Cause

The remaining boundary is inside the native Megatron train path, but the original debug dump flow only saved train data after `train(...)` returned. When the worker crashed during step 0, the exact train batch was lost together with the process, which made train-only replay unnecessarily hard.

### Solution

1. Save the actor train batch before entering `train(...)`, immediately after rollout statistics are logged.
2. Add explicit boundary logs around:
   - actor-side data iterator setup
   - `compute_advantages_and_returns`
   - rollout-stat logging
   - `forward_backward_func(...)`
   - `optimizer.step()`
   - final loss reduction
3. Capture rollout samples with `--save-debug-rollout-data`.
4. Replay with `--load-debug-rollout-data` to run the actor in train-only mode without rebuilding rollout/SGLang.

The AMD launcher now accepts these debug flags through environment variables:

- `RELAX_SAVE_DEBUG_ROLLOUT_DATA`
- `RELAX_SAVE_DEBUG_TRAIN_DATA`
- `RELAX_LOAD_DEBUG_ROLLOUT_DATA`
- `RELAX_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE`
- `RELAX_DUMP_DETAILS`

### Prevention

For native-crash boundaries, always persist the replay inputs before entering the suspected kernel path. Post-step dumps are too late when the worker can disappear without raising Python exceptions.

### Related

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Actor: `relax/backends/megatron/actor.py`
- Train step: `relax/backends/megatron/model.py`
- Experiment log: `references/experiment-log.md`

## Single-rank MI210 actor crashes inside distributed `optimizer.step()`

**Added:** 2026-04-18
**Domain:** research

### Symptom

The run clears rollout generation, reward execution, train-batch transfer, and
Megatron forward/backward, then dies on the first optimizer step with the last
actor-side logs looking like:

```text
train_one_step rollout=0 step=0: finished forward_backward
train_one_step rollout=0 step=0: starting optimizer.step
ray.exceptions.ActorDiedError: MegatronTrainRayActor
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file
```

### Cause

The MI210 actor path currently runs with a single training rank
(`data_parallel_size == 1`), but Megatron setup was still forcing:

- `use_distributed_optimizer`
- `use_precision_aware_optimizer`
- `overlap_param_gather`
- `overlap_param_gather_with_optimizer_step`

That combination is useful only when optimizer state is actually sharded across
multiple data-parallel ranks. On the single-rank ROCm actor it only adds the
distributed-optimizer wrapper and its ROCm-specific crash surface around
`optimizer.step()`.

### Solution

Normalize the Megatron optimizer flags before model and optimizer construction:

1. detect the single-rank path from `data_parallel_size` or `world_size`
2. disable the distributed-optimizer-only flags listed above
3. leave CPU optimizer offload enabled, since that addresses the MI210 memory
   limit and does not require the distributed optimizer wrapper itself

This keeps the single-rank actor on the simpler mixed-precision / CPU-offload
optimizer path while preserving the multi-rank distributed optimizer path for
real DP sharding.

### Prevention

Do not force distributed optimizer features globally in ROCm bring-up scripts.
When DP size is 1, normalize back to the non-distributed optimizer path before
Megatron builds DDP wrappers and optimizer state.

### Related

- Train step: `relax/backends/megatron/model.py`
- Optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Experiment log: `references/experiment-log.md`

## HybridDeviceOptimizer CPU-offload mappings go stale after Megatron main-param wrapping

**Added:** 2026-04-18
**Domain:** research

### Symptom

After the single-rank distributed-optimizer flags are disabled, the saved
train-only replay gets farther but now fails with a real Python traceback:

```text
KeyError: ...
```

inside local Megatron's `HybridDeviceOptimizer` bookkeeping, during the first
step-0 optimizer update.

### Cause

Megatron's mixed-precision / main-param wrapping can replace the wrapped
optimizer `param_groups` after the CPU-offload optimizer has already built its
internal parameter mappings. On the MI210 actor path, that left
`HybridDeviceOptimizer` holding stale group-to-parameter bookkeeping, so the
first optimizer step crashed when it tried to synchronize the live param groups
back into its CPU/GPU sub-optimizers.

### Solution

After `get_megatron_optimizer(...)` returns, detect wrapped
`HybridDeviceOptimizer` instances and explicitly refresh them by re-running:

1. `_init_sub_optimizers()`
2. `_sync_hdo_param_groups_to_sub_optimizers()`

in the live wrapped optimizer tree.

This keeps the local CPU-offload optimizer's param-group mappings aligned with
Megatron's final wrapped/main-param view.

### Prevention

Whenever Megatron or a local wrapper mutates optimizer param groups after
optimizer construction, refresh any downstream bookkeeping layers that cache
those groups. Replay-mode failures with a deterministic `KeyError` are usually
easier to fix than the earlier native EOF crash and should be treated as
progress, not regression.

### Related

- Train setup: `relax/backends/megatron/model.py`
- Optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Local Megatron: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Experiment log: `references/experiment-log.md`

## Single-rank CPU-offload overlap wedges the MI210 replay path

**Added:** 2026-04-18
**Domain:** research

### Symptom

Even after the distributed-optimizer wrapper is removed and the CPU-offload
param mappings are refreshed, the saved train-only replay can still disappear
at step 0 unless the foreground validation kills it first. The last useful
actor-side boundary logs are:

```text
train_one_step rollout=0 step=0: finished forward_backward
train_one_step rollout=0 step=0: starting optimizer.step
```

with no inner Python traceback from the actor before the process is gone.

### Cause

The MI210 actor path currently runs with a single training rank. In that
configuration, `overlap_cpu_optimizer_d2h_h2d` adds asynchronous host/device
optimizer overlap complexity that provides no distributed benefit, while still
expanding the native ROCm crash surface inside the first optimizer step.

### Solution

Treat `overlap_cpu_optimizer_d2h_h2d` the same way as other single-rank
distributed-optimizer-only flags:

1. detect `data_parallel_size == 1` or `world_size == 1`
2. disable `overlap_cpu_optimizer_d2h_h2d` before model/optimizer construction
3. keep plain CPU optimizer offload enabled for memory savings

After this change, the saved replay survives the full 5-minute foreground
validation window and exits only because `timeout` terminates the Ray cluster.

### Prevention

Do not assume all CPU-offload overlap knobs are safe on a single-rank ROCm
actor just because they are useful on larger distributed runs. On MI210, first
prove the plain CPU-offload path is stable, then reintroduce overlap only with
a dedicated replay reproducer.

### Related

- Train setup: `relax/backends/megatron/model.py`
- Optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Replay capture: `log/debug_capture_20260418_153613`
- Experiment log: `references/experiment-log.md`

## Non-colocated serial startup creates the actor too early on MI210

**Added:** 2026-04-18
**Domain:** research

### Symptom

On the non-colocated sync path, the actor can still die before any real train
step even though SGLang eventually becomes healthy. The visible error appears
later when the controller tries to wire services together:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor
...
File ".../actor.py", line 78, in set_rollout_manager
```

But the worker-side timeline shows the actor had already died earlier, while
rollout was still bringing up SGLang.

### Cause

Serial service creation was still building the actor first and rollout second.
On MI210, rollout startup can spend minutes in SGLang initialization and weight
loading. That left the Megatron actor resident and idle during the entire
rollout bring-up window, and the actor could disappear before
`set_rollout_manager()` was ever called. The later RPC only exposed an
already-dead worker.

### Solution

On the non-colocated serial path:

1. create `rollout` before `actor`
2. let SGLang finish its heavy startup window first
3. create the actor only after rollout is ready
4. then call `set_rollout_manager()` and initial weight sync

After reordering service creation this way, the clean 5-minute foreground
validation moved past the old boundary: the actor survived rollout startup,
`set_rollout_manager()` succeeded, weight sync completed, and the run entered
real rollout generation before `timeout` ended the validation.

### Prevention

When a service has a long startup phase and another heavy service depends on it
only later, do not keep the dependent service resident and idle during that
window unless there is a strong reason. On MI210, the safe serial order is
rollout first, actor second.

### Related

- Controller: `relax/core/controller.py`
- Validation log: `log/amd-qwen3-4b-2gpu-20260418_181910.log`
- Experiment log: `references/experiment-log.md`

## Step-0 Megatron actor death after `train_0` is transferred on MI210

**Domain:** research

### Symptom

The run clears startup, rollout generation, and transfer-queue handoff, then dies during the first actor train step with a generic Ray worker death:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file
```

The important worker-side evidence is narrower than before:

- rollout logs `Batch ... transferred successfully for rollout_id: 0`
- Megatron logs `start to get rollout_id: 0 data from transfer queue for train with mcore`
- rollout stats are logged from `relax.backends.megatron.data`
- no Python traceback appears before the worker disappears

### Cause

At this boundary, the failure is no longer rollout bring-up or reward-worker plumbing. The actor has already consumed `train_0` and entered the real Megatron training path. On the MI210 ROCm path, the remaining candidate is the first backward / optimizer-kernel phase itself, especially when the launcher still uses an aggressive single-GPU training configuration.

Two launcher choices were particularly suspicious for this AMD path:

1. `--attention-backend auto` leaves Megatron free to select a more aggressive backend during training.
2. A fixed microbatch schedule on long responses keeps each train microbatch large even when sample lengths spike.

### Solution

Use a more conservative actor-side train configuration on AMD:

1. force `--attention-backend unfused`
2. disable `--bias-dropout-fusion`
3. avoid unnecessary single-GPU extras such as `--sequence-parallel`
4. keep `--micro-batch-size 1` explicit
5. reduce the actual train-step load with a smaller `--global-batch-size` / `--num-steps-per-rollout` pairing and, if needed, a shorter `--rollout-max-response-len`

Do not try `--use-dynamic-batch-size` on this path while `--qkv-format bshd` is required; Relax explicitly rejects that combination during argument validation.

This does not prove the final root cause is a specific kernel, but it moves the AMD launcher away from the most likely native-crash paths first.

### Prevention

When a ROCm run reaches `train_0` consumption but still dies before the first `train/...` metric, treat that as a train-kernel stability problem, not another control-plane issue. Tighten the actor launcher first, then rerun the standard 5-minute foreground validation from a clean Ray cluster before starting another tmux run.

### Related

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Experiment log: `references/experiment-log.md`

## `relax.distributed.ray.rollout` imports SGLang/Megatron too early

**Added:** 2026-04-17
**Domain:** research

### Symptom

The rollout-side startup path emits SGLang and Megatron warnings before any
`SGLangEngine` launch logs appear. In isolated reproduction, importing
`relax.distributed.ray.rollout` was enough to pull `sglang` and, through later
dependencies, Megatron-backed modules into the process.

### Cause

Two import surfaces were too eager:

1. `relax/distributed/ray/rollout.py` imported `sglang.srt.constants`, which
   forced Python through `sglang.__init__` and its large public API.
2. `relax/backends/sglang/sglang_engine.py` imported the checkpoint-service
   client at module scope, which imported DCS backends and then Megatron-backed
   utilities.

That meant the rollout stack could start in a contaminated interpreter before
the later Megatron import blocker had any chance to run.

### Solution

1. Replace the `sglang.srt.constants` import in `rollout.py` with local string
   constants matching SGLang's values.
2. Move checkpoint-service client import in `sglang_engine.py` behind a helper
   so it happens only when DCS registration is actually needed.
3. Keep a regression test that verifies importing
   `relax.distributed.ray.rollout` does not import `sglang`.

### Prevention

Treat rollout startup modules as control-plane code. Avoid importing large model
or engine packages at file scope when the values needed are simple constants or
can be loaded lazily.

### Related

- Files: `relax/distributed/ray/rollout.py`, `relax/backends/sglang/sglang_engine.py`
- Tests: `tests/distributed/ray/test_rollout.py`, `tests/backends/sglang/test_sglang_engine.py`

## Phase-1 timeout leaves a stale Ray training job behind

**Added:** 2026-04-17
**Domain:** research

### Symptom

After a foreground validation like:

```text
timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh
```

the shell exits with code `124`, but `python3 -m relax.entrypoints.train` and
its Ray workers can still be running in the background. The next run may then
show stale-worker symptoms such as mixed worker IDs or unexpected control-plane
noise.

### Cause

The shell timeout ends the outer launcher process, but that does not guarantee
the submitted Ray job tree is gone. On this MI210 flow, stale validation jobs
were able to survive long enough to pollute the next production run.

### Solution

1. After a timed-out foreground validation, explicitly stop the Ray cluster with
   `ray stop --force`.
2. Kill any leftover Relax launcher/train processes before starting the tmux
   run.
3. Start the production run only from that cleaned state.

### Prevention

Do not assume a successful shell timeout means the validation run self-cleaned.
Verify the process list or stop Ray explicitly before the next run.

### Related

- Experiment log: `references/experiment-log.md`

## Module-import Megatron blocker kills `SGLangEngine` actor creation

**Added:** 2026-04-17
**Domain:** research

### Symptom

Rollout startup gets far enough to create `RolloutManager`, but then fails while
creating the first `SGLangEngine` actor. Ray surfaces:

```text
ray.exceptions.ActorDiedError: ... ray::SGLangEngine.__init__()
...
System error: Blocked import of megatron for SGLang transformers backend
```

The nested path is typically:

```text
RolloutManager.__init__()
  -> start_rollout_servers()
  -> EngineGroup.start_engines()
  -> _allocate_rollout_engine_addr_and_ports_normal()
  -> engine._get_current_node_ip_and_free_port.remote()
```

### Cause

The import-time SGLang isolation was made too aggressive. Installing a
module-level `MetaPathFinder` blocker for `megatron` inside
`sglang_engine.py` runs before Ray has finished computing actor creation task
inputs. That means actor creation itself can fail before `SGLangEngine` reaches
its own `__init__` logic.

### Solution

1. Keep module-import-time isolation limited to pruning local Megatron paths,
   editable import hooks, and cached Megatron modules from the actor worker.
2. Delay the stronger `megatron` import blocker until `SGLangEngine.__init__`
   and the spawned SGLang server/scheduler subprocesses.
3. Revalidate from a clean Ray cluster and confirm rollout no longer dies at
   `SGLangEngine.__init__` with the blocked-import error.

### Prevention

Separate safe import-time pruning from hard import blocking. In Ray actor
workers, a blocker that can affect actor/task deserialization belongs later in
the actor lifecycle, not at module import time.

### Related

- Files: `relax/backends/sglang/sglang_engine.py`, `relax/distributed/ray/rollout.py`
- Experiment log: `references/experiment-log.md`

## DCS package init pulls Megatron into control-plane imports

**Added:** 2026-04-17
**Domain:** research

### Symptom

Control-plane services such as `DCSCoordinator`, `HealthStatus`, or
`RolloutManager` emit Megatron and Transformer Engine warnings during startup,
even though they should not be initializing training backends yet.

### Cause

`import relax.distributed.checkpoint_service.coordinator.service` first
executes `relax.distributed.checkpoint_service.__init__`. That package
`__init__` eagerly re-exported backend and client symbols, which immediately
imported `relax.distributed.checkpoint_service.backends.device_direct`.
`device_direct.py` imports `from megatron.core import mpu`, so a plain
coordinator-service import leaked Megatron into unrelated startup paths.

### Solution

1. Convert `relax.distributed.checkpoint_service.__init__` to lazy exports via
   `__getattr__` so importing the package does not eagerly import backends,
   client, or coordinator modules.
2. Convert `relax.distributed.checkpoint_service.backends.__init__` to lazy
   exports as well so `DeviceDirectBackend` is only imported when actually
   requested.
3. Keep a regression test that verifies importing
   `relax.distributed.checkpoint_service.coordinator.service` does not import
   `megatron` or `sglang`.

### Prevention

Treat package `__init__` files on control-plane paths as import boundaries, not
convenience dumping grounds. Re-export heavy backend symbols lazily when the
package also contains lightweight control-plane modules.

### Related

- Files: `relax/distributed/checkpoint_service/__init__.py`, `relax/distributed/checkpoint_service/backends/__init__.py`
- Tests: `tests/distributed/checkpoint_service/test_imports.py`
- Experiment log: `references/experiment-log.md`

## `core.registry` imports Megatron through the advantages service

**Added:** 2026-04-17
**Domain:** research

### Symptom

Even after fixing the checkpoint-service package leak, control-plane workers
such as `HealthStatus`, `Rollout`, or `RolloutManager` still emitted Megatron
warnings. A plain interpreter reproduction showed that importing either
`relax.core.registry` or `relax.core.controller` was already enough to load
`megatron`.

### Cause

`relax.core.registry` eagerly imports all component classes, including
`relax.components.advantages`. `advantages.py` imported `megatron.core.mpu`
and `relax.backends.megatron.loss.apply_opd_kl_to_advantages` at module scope,
so the entire controller/registry stack became Megatron-heavy even when only
control-plane services were being created.

### Solution

1. Keep `relax.components.advantages` importable without Megatron by moving
   `from megatron.core import mpu` into the PPO branch.
2. Move `apply_opd_kl_to_advantages` into the OPD-only branch where it is
   actually used.
3. Keep regression tests that verify importing `relax.components.advantages`
   and `relax.core.registry` does not import `megatron` or `sglang`.

### Prevention

Treat component modules referenced by global registries as import boundaries.
Heavy backend imports in those files should stay inside the exact runtime paths
that require them.

### Related

- Files: `relax/components/advantages.py`, `relax/core/registry.py`
- Tests: `tests/components/test_advantages_imports.py`, `tests/core/test_registry_imports.py`
- Experiment log: `references/experiment-log.md`

## Rollout-side SGLang bootstrap starts before Megatron isolation is active

**Added:** 2026-04-17
**Domain:** research

### Symptom

Even after cleaning the DCS package and controller/registry import surfaces,
the rollout-side processes still emitted Megatron warnings:

- `ServeReplica:rollout:Rollout`
- `RolloutManager`

This happened before the first `SGLangEngine` actor reported its own
`Installed Megatron import blocker` log line.

### Cause

The earlier isolation work only guaranteed blocking inside `SGLangEngine`.
However, `RolloutManager.__init__` loads the configured rollout function
(`relax.engine.rollout.sglang_rollout.generate_rollout`) and other rollout-side
SGLang helpers before any engine actor exists. The Serve `Rollout` replica also
starts in a process that still has the full `Megatron-LM` checkout on
`PYTHONPATH`.

### Solution

1. Install the same transformers-mode Megatron isolation at the start of
   `Rollout.__init__`.
2. Install it again at the start of `RolloutManager.__init__`.
3. Keep this limited to `sglang_model_impl == "transformers"` so native
   Megatron-backed paths are not broken.

### Prevention

When the rollout path uses the plain transformers backend, apply Megatron
isolation to every rollout-side process that may import SGLang modules, not
just the final engine actor.

### Related

- Files: `relax/components/rollout.py`, `relax/distributed/ray/rollout.py`
- Tests: `tests/distributed/ray/test_rollout.py`
- Experiment log: `references/experiment-log.md`

## Actor dies during sync `set_rollout_manager` startup on MI210

**Added:** 2026-04-16
**Domain:** research

### Symptom

The run gets through:

- actor initialization
- rollout deployment
- RolloutManager creation
- rollout-side SGLang startup

and then fails during the controller's actor hookup:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor died unexpectedly
```

The nested stack shows the failure surface at:

```text
Actor.set_rollout_manager
  -> RayTrainGroup.set_rollout_manager
  -> MegatronTrainRayActor.set_rollout_manager
```

with Ray reporting a worker `SYSTEM_ERROR` / EOF rather than a surfaced Python exception.

### Cause

For the sync training path, `TrainRayActor.set_rollout_manager()` was still doing two extra rollout-manager round-trips intended for the fully-async DCS path:

1. `rollout_manager.set_train_parallel_config(...)`
2. `rollout_manager.get_weight_sync_lock()`

Those values are only used later by `update_weights_fully_async()`. In the non-`fully_async` path they add cross-actor startup work during a fragile MI210 bring-up window without providing any value.

### Solution

1. Keep `self.rollout_manager = rollout_manager` for all paths.
2. Return immediately for non-`fully_async` runs.
3. Only execute the extra rollout-manager setup when `self.args.fully_async` is true.
4. Validate with a foreground launcher run long enough to clear the old boundary. The healthy signal is that the run survives past the previous `set_rollout_manager` death window and only stops because the validation timeout expires.

### Prevention

Do not let sync startup paths inherit fully-async wiring unless the downstream code actually consumes it. Cross-actor setup during bootstrap should be limited to what the current execution mode requires.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## Actor dies during `set_rollout_manager` after offloaded init sleep

**Added:** 2026-04-16
**Domain:** research

### Symptom

After enabling CPU optimizer offload, the run gets through:

- actor initialization
- rollout deployment
- RolloutManager creation

but then still dies at the first rollout-manager hookup call:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor died unexpectedly
```

Ray reports a worker `SYSTEM_ERROR` / EOF without a surfaced Python traceback.

### Cause

The earlier sync-path fix removed unnecessary fully-async rollout-manager setup, but the offloaded actor still went to sleep at the end of Megatron `_init()`. That meant the very first real post-init RPC had to wake the actor during bootstrap. On this MI210 path, waking immediately after init was brittle enough to kill the worker before a Python exception could be surfaced.

### Solution

1. Track whether the Megatron actor is actually sleeping.
2. In `TrainRayActor.set_rollout_manager()`, only call `wake_up()` when offload is enabled and the actor is already marked sleeping.
3. Keep the sync offloaded actor resident after `_init()` until rollout-manager hookup completes.
4. Validate with a foreground run long enough to clear the old boundary. The healthy signal is that the run survives past `set_rollout_manager`, actor service becomes ready, and rollout initialization continues.

### Prevention

Do not put the sync actor to sleep during bootstrap just because offload is enabled. Offload lifecycle transitions should happen only after the startup sequence that still depends on a live process-group state has completed.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## Ray GCS times out during late rollout startup on single-node MI210

**Added:** 2026-04-16
**Domain:** research

### Symptom

The run clears the old actor-death boundaries and reaches late rollout startup, then the control plane collapses:

```text
Failed to connect to GCS within 60 seconds. GCS may have been killed...
```

At the same time, Ray logs show symptoms like:

```text
RpcError: Deadline Exceeded
Failed to push error to driver
```

and the whole job terminates before a stable training loop starts.

### Cause

On this single-node MI210 setup, the Ray head was launched with far more CPU slots than the job needed. That encouraged large numbers of Python workers and avoidable control-plane chatter. Internal Relax actors and managers were also still emitting Ray task-event metadata that was not useful for this bring-up. Under rollout startup load, those control-plane queues backed up far enough that the driver lost contact with GCS.

### Solution

1. Launch the Ray head with a smaller CPU budget appropriate for the 2-GPU job, for example `--num-cpus 16`.
2. Disable Ray task events on internal Relax actors and managers that do not need them.
3. Increase GCS reconnect tolerance in the launcher environment so transient control-plane stalls do not kill the job immediately.
4. Re-run a foreground validation for about 9 minutes. The healthy signal is that the run stays alive through the former GCS timeout window and only stops because the shell `timeout` expires.

### Prevention

For single-node bring-up, size the Ray head for the actual workload instead of the full host CPU count, and keep internal control-plane metadata emission to the minimum needed for debugging. Excess scheduling and task-event traffic can become the limiting resource before GPU utilization does.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## Ray keepalive watchdog timeout during SGLang startup

**Added:** 2026-04-17
**Domain:** research

### Symptom

The run survives actor bring-up and begins rollout startup, then eventually fails with the first visible error in Serve:

```text
ray.exceptions.ActorUnavailableError: The actor ... is temporarily unavailable:
RpcError: RPC error: keepalive watchdog timeout rpc_code: 14
```

The top-level failure often looks misleading because a different actor is only reported dead later, for example during `set_rollout_manager`.

### Cause

The controller-side failure surface is delayed. The earlier logs show the real problem occurs while the rollout replica is still inside `SGLangEngine.init()`: Ray workers begin reporting `keepalive watchdog timeout`, and the first collapsed task is usually the rollout/SGLang startup actor. Ray state dumps from the same run show very large `NodeManagerService.grpc_server.ReportWorkerBacklog` volumes and long queueing/execution time, which indicates a control-plane stall rather than a Python exception in the Megatron actor.

### Solution

1. Set `RAY_task_events_report_interval_ms=0` before starting the Ray head to suppress periodic task-event reporting the run does not need.
2. Increase `RAY_grpc_client_keepalive_time_ms` and `RAY_grpc_client_keepalive_timeout_ms` so long rollout startup phases do not trip the watchdog so aggressively.
3. Forward those same Ray env vars into the submitted job/runtime env so the driver, Serve replicas, and child workers all inherit the same settings.
4. Re-run the standard 5-minute foreground validation. The healthy signal is that the run clears the old actor-death boundary and only exits because the shell `timeout` fires.

### Prevention

When debugging long single-node rollout initialization on Ray, do not assume the last actor reported dead is the root cause. Check worker and controller logs for earlier `keepalive watchdog timeout` messages and treat control-plane reporting load as a first-class failure mode.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## SGLang transformers import fails on optional Quark `aiter` path

**Added:** 2026-04-16
**Domain:** research

### Symptom

After switching the rollout launcher to `--sglang-model-impl transformers`, rollout startup still fails. The top-level Relax error is generic:

```text
Exception: Server process terminated unexpectedly.
```

The real scheduler traceback in the SGLang worker shows:

```text
ValueError: Model architectures ['TransformersForCausalLM'] are not supported for now.
```

At the same time, the model registry logs indicate that `sglang.srt.models.transformers` failed to import because `aiter` is missing.

### Cause

This is not actually a model-architecture mismatch. The generic SGLang transformers model class never gets registered. Importing `sglang.srt.models.transformers` pulls in `sglang.srt.layers.moe.ep_moe.layer`, which hard-imports `QuarkW4A4MXFp4MoE`. That Quark scheme module depends on the optional ROCm package `aiter`. On this MI210 machine `aiter` is not installed, so the import aborts before `TransformersForCausalLM` can be registered.

### Solution

1. Keep the guarded quantization backend registration in SGLang.
2. Patch the local SGLang checkout so `sglang.srt.layers.moe.ep_moe.layer` treats `QuarkW4A4MXFp4MoE` as optional:
   - catch the `ImportError`
   - log that Quark MXFP4 MoE support is unavailable
   - use an empty type tuple for the Quark-specific `isinstance` checks
3. Validate directly with a Python import using the local SGLang checkout on `PYTHONPATH`. The expected healthy signal is:

```text
OK .../sglang/srt/models/transformers.py
```

4. Re-run the AMD launcher and confirm the earlier `TransformersForCausalLM` registry failure no longer appears.

### Prevention

Optional quantization backends must not block import of the generic inference model path. Keep Quark and `aiter` dependencies isolated to the actual quantized execution branches rather than module import time.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## `scaled_masked_softmax_cuda` import on ROCm

**Added:** 2026-04-15
**Domain:** research

### Symptom

The run reaches training, then fails inside Megatron fused softmax:

```text
ModuleNotFoundError: No module named 'scaled_masked_softmax_cuda'
```

In one observed run, the parsed config still reported:

```text
masked_softmax_fusion ........................... False
```

### Cause

Disabling masked softmax fusion at the CLI was not sufficient because the Megatron-Bridge provider path did not propagate `masked_softmax_fusion` everywhere it needed to go. Upstream fused softmax code also assumed the CUDA extension import was always valid instead of treating a missing extension as a signal to fall back.

### Solution

1. Pass `--no-masked-softmax-fusion` in the AMD launcher.
2. Propagate `masked_softmax_fusion` through the Relax Megatron provider override path.
3. Make the local Megatron fused softmax path treat missing CUDA extensions as unavailable and fall back instead of crashing.

### Prevention

When a CUDA-specific fusion is disabled from the CLI, verify the effective parsed config and the provider override path, not just the launcher arguments. ROCm runs should prefer explicit fallback behavior over optimistic extension imports.

### Related

- Skill: `megatron-bridge-rocm-overrides`
- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## TorchInductor ROCm `KernelMetadata.cluster_dims` failure

**Added:** 2026-04-15
**Domain:** research

### Symptom

After rollout generation and reward execution succeed, the actor training path crashes during TorchInductor/Triton compilation:

```text
torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'
```

### Cause

This failure is later than the earlier startup issues and points to a compiler/runtime mismatch in the ROCm TorchInductor/Triton stack. TorchInductor's launcher generation expects `binary.metadata.cluster_dims`, but the ROCm-side kernel metadata object in this environment does not provide that field.

In the ROCm Megatron fork experiment, the concrete trigger was
`ROCm-Megatron-LM/megatron/core/jit.py`. On PyTorch >= 2.2 that file promoted
`jit_fuser` from `torch.jit.script` to `torch.compile`, including on HIP. The
first reproduced failing helper was
`megatron.core.fusions.fused_cross_entropy.calculate_logits_max`. When the HIP
path kept TorchScript instead, import then failed on
`megatron.core.transformer.torch_norm.L2Norm` because TorchScript could not
resolve `self.eps`.

### Solution

1. Treat the error as a PyTorch/Triton ROCm compatibility issue rather than a Relax orchestration failure.
2. Capture the exact PyTorch, Triton, and ROCm versions before changing anything else.
3. Bisect by disabling or narrowing the compiled Megatron path that triggers TorchInductor launcher generation, then validate with a short foreground run before restoring tmux retries.
4. For the local ROCm Megatron checkout, make `jit_fuser` an eager no-op when `torch.version.hip` is set instead of promoting it to `torch.compile` or TorchScript.
5. If available, move to a PyTorch/Triton build where ROCm kernel metadata and TorchInductor launcher expectations match.

### Prevention

Once the run reaches rollout generation, stop labeling later failures as "startup" issues. Preserve the exact compiler/runtime versions with the log so future ROCm debugging starts from the right layer.

### Related

- Skill: `rocm-inductor-triton-cluster-dims`
- Skill: `rocm-relax-bringup`
- Local Megatron: `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/jit.py`
- Experiment log: `references/experiment-log.md`

## Internal proxy intercepts local SGLang health checks

**Added:** 2026-04-15
**Domain:** research

### Symptom

SGLang reaches:

```text
The server is fired up and ready to roll!
```

but the rollout service hangs during initialization and direct health probes return a Squid access error instead of the local server response.

### Cause

The machine-wide proxy environment only bypassed loopback hostnames. Relax and SGLang communicate over the node IP (for example `172.27.112.25`), so those requests were sent to the proxy unless the node hostname/IP was added to `no_proxy`.

### Solution

1. Extend both `no_proxy` and `NO_PROXY` with `127.0.0.1`, `localhost`, `::1`, `MASTER_ADDR`, the resolved `MASTER_ADDR`, the current hostname, and the resolved hostname IP.
2. Apply that both in the Ray job runtime env and in Relax's child-process environment propagation.
3. Re-run a short foreground validation and confirm local SGLang health requests return `200` directly.

### Prevention

Whenever Ray jobs launch local HTTP services on managed clusters, treat proxy bypass as part of runtime bring-up rather than as an afterthought.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## Adam state OOM on MI210 actor rank

**Added:** 2026-04-15
**Domain:** research

### Symptom

The run gets through rollout generation and reward execution, then the first optimizer step fails with:

```text
torch.OutOfMemoryError: HIP out of memory ... in torch.optim.adam._init_group
```

### Cause

The actor rank held the bf16 model weights successfully, but the lazy creation of Adam moment tensors (`exp_avg` and `exp_avg_sq`) pushed the 4B model beyond single-MI210 device memory.

### Solution

1. Enable Megatron CPU optimizer offload in the AMD launcher.
2. Use the full supported flag bundle:

```text
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

3. Re-run a foreground validation and confirm the actor initializes successfully instead of dying on the first optimizer-state allocation.

### Prevention

On 2x MI210 runs where actor and rollout each get a single GPU, treat optimizer-state placement as a first-class memory decision, not just activation sizing.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## CPU optimizer offload crashes when TE is absent

**Added:** 2026-04-15
**Domain:** research

### Symptom

After enabling CPU optimizer offload, actor initialization fails with:

```text
TypeError: '>=' not supported between instances of 'NoneType' and 'Version'
```

originating from `megatron.core.utils.is_te_min_version()`.

### Cause

The local Megatron checkout assumes `get_te_version()` always returns a valid version object. On ROCm systems without Transformer Engine installed, it returns `None`, and the version comparison crashes before the offload path can proceed.

### Solution

1. Patch the local Megatron checkout so `is_te_min_version()` returns `False` when `get_te_version()` returns `None`.
2. Validate directly in Python that `is_te_min_version("2.1.0")` returns `False` on the TE-less environment.
3. Re-run the AMD e2e launcher and confirm actor initialization progresses past optimizer construction.

### Prevention

Capability probes for optional dependencies should degrade to `False`, not raise type errors. Keep TE absence on ROCm as an expected configuration in local bring-up patches.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`

## SGLang transformers backend discovers local Megatron-LM

**Added:** 2026-04-16
**Domain:** research

### Symptom

The run clears actor initialization, then dies during rollout bring-up with a generic Ray error:

```text
ray.exceptions.ActorDiedError: MegatronTrainRayActor died unexpectedly
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file
```

At the same time, the rollout-side SGLang child logs warnings sourced from the local `Megatron-LM` checkout, such as:

```text
Using Megatron-FSDP without Transformer Engine.
```

or local Megatron optimizer / FSDP import warnings even though `--sglang-model-impl transformers` was set.

### Cause

The Relax runtime propagates `PYTHONPATH` to Ray workers so the actor can import the local Megatron checkout. That is correct for the training side, but it also means the spawned SGLang server subprocess can discover `Megatron-LM`. On this MI210 ROCm path, once SGLang's Transformers backend sees the local checkout, it can enter Megatron-FSDP-related code during rollout startup, correlating with the actor worker dying before the later `set_rollout_manager` call.

### Solution

1. Keep the actor-side `PYTHONPATH` propagation for local Megatron imports.
2. When spawning SGLang with `model_impl=transformers`, strip `Megatron-LM` from both `PYTHONPATH` and `sys.path` before the SGLang child imports `http_server`.
3. Keep the strip active for the child server lifetime, not just around `p.start()`, because the server can import more modules later during engine initialization.
4. Also filter `PYTHONPATH` in the rollout actor runtime env so the rollout-side SGLang process does not begin life with the local Megatron checkout on its search path.
5. Re-run the foreground validation. The healthy signal is that the SGLang path no longer logs `Using Megatron-FSDP without Transformer Engine.` or `Detected Megatron Core, using Megatron-FSDP with Megatron.` during rollout startup, and the run survives past the earlier actor-death window.

### Prevention

When a repository exposes both training and inference stacks in one Ray job, do not assume the same `PYTHONPATH` is safe for every child process. Keep rollout-side inference imports narrower than actor-side training imports, and do not restore a broadened import path before the inference server has finished importing its backend.

### Related

- Skill: `rocm-relax-bringup`
- Experiment log: `references/experiment-log.md`
