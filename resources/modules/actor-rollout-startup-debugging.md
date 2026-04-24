# Topic: Actor / Rollout Startup Debugging

> Generated via `/deep-dive` on 2026-04-17. Covers the training entrypoint, controller/service orchestration, actor/rollout startup, and the worker-log workflow needed to debug late startup failures.

* **Scope:** `relax.entrypoints.train`, `relax.core.controller`, `relax.core.service`, `relax.components.actor`, `relax.distributed.ray.{placement_group,actor_group,train_actor,rollout}`, and `relax.backends.megatron.actor`
* **Relevance:** This is the shortest path for learning how Relax starts actor and rollout services, and it is the critical path for debugging the current MI210 `MegatronTrainRayActor` death during rollout-manager hookup.

## Overview

Relax startup is easier to reason about if you split it into two layers. The first layer is orchestration: the training entrypoint initializes Ray, constructs a `Controller`, deploys services, and wires actor and rollout together. The second layer is backend execution: the actor service allocates a Ray train group, each worker initializes distributed state and the Megatron model, and the rollout side starts the SGLang engine group. The code intentionally hides some of this behind Ray Serve handles, so many failures surface one layer above their true origin.

For debugging, the important habit is to identify the first failing worker and then walk back up the call graph. In this codebase, controller logs often show where a request failed, but not why the worker process died. The worker-specific Ray logs under `/tmp/ray/session_latest/logs/` are usually the real source of truth. The current MI210 failure is a good example: the visible exception is an `ActorDiedError` while calling `set_rollout_manager`, but the Megatron worker had already exited earlier with a Ray `SYSTEM_ERROR` / EOF.

## Implementation Details

### Entrypoint and cluster bring-up

The process starts in `main()` inside `relax/entrypoints/train.py:53-98`. That function loads `configs/env.yaml`, applies runtime-env post-processing, initializes tracking, and calls `ray.init(runtime_env=runtime_env)` if Ray is not already initialized (`relax/entrypoints/train.py:56-76`). It then constructs the `Controller` and hands off control to `ctrl.training_loop()` (`relax/entrypoints/train.py:76-89`).

This division matters because failures before controller construction are usually environment or cluster failures, while failures after controller construction are generally service startup or training-loop failures.

### Controller owns service orchestration

`Controller.__init__()` builds the data system, checkpoint coordinator, optional metrics/autoscaler services, and then calls `register_all_serve()` to create the role-specific services (`relax/core/controller.py:35-80`). Each service is wrapped by `Service`, which deploys the actual Ray Serve deployment and keeps a handle for later RPCs (`relax/core/controller.py:168-188`, `relax/core/service.py:22-70`).

The controller is the right place to read when you want to know:

1. Which roles exist for the chosen algorithm.
2. How placement groups are shared.
3. When actor and rollout are connected.

The current failure boundary is in the later controller startup path where it wires actor to rollout. The relevant call is `self.serve_dict[ROLES.actor].set_rollout_manager(rollout_manager)` in `relax/core/controller.py:385`.

### Service is a thin Ray Serve wrapper

`Service` binds the deployment class, deploys it with `serve.run`, and exposes convenience methods that forward to the Serve handle (`relax/core/service.py:72-89`, `relax/core/service.py:146-186`). This wrapper is intentionally thin, which means a failure here usually reflects a deployment/worker issue elsewhere.

For the current bug, `Service.set_rollout_manager()` simply awaits `self.handle.set_rollout_manager.remote(rollout_manager)` (`relax/core/service.py:157-159`). If this fails with `ActorDiedError`, the service wrapper is only reporting that the underlying deployment or one of its child Ray actors died.

### Actor deployment bridges Serve to the train group

The Serve deployment class is `Actor` in `relax/components/actor.py:22-84`. During construction it allocates the train group with `allocate_train_group(...)` and immediately calls `self.actor_model.async_init(...)` via `ray.get(...)` (`relax/components/actor.py:55-73`).

The important consequence is that when the actor deployment reports “Actor initialized”, that only means the underlying train actors completed their initialization path. It does not mean rollout has started or that weight-sync wiring is finished.

Later, `Actor.set_rollout_manager()` stores the rollout manager, forwards the call to the train group, and then triggers `update_weights()` in non-`fully_async` mode (`relax/components/actor.py:75-84`). This is why the current failure is visible at `set_rollout_manager`: this is the first time the actor deployment actively touches the rollout side after the long rollout startup window.

### Placement group and train group construction

`allocate_train_group()` returns `RayTrainGroup` with `num_gpus_per_actor=0.4` (`relax/distributed/ray/placement_group.py:72-79`). `RayTrainGroup` creates one Ray actor per world rank and injects runtime env vars, visible-device control, and offload-related `LD_PRELOAD` settings when `offload_train` is enabled (`relax/distributed/ray/actor_group.py:31-106`).

`RayTrainGroup.set_rollout_manager()` simply fans the call out to all worker actors and blocks on `ray.get(...)` (`relax/distributed/ray/actor_group.py:165-166`). If one worker died earlier, this is where the controller sees the failure.

### Train actor base class

`TrainRayActor` sets `MASTER_ADDR`, `MASTER_PORT`, rank/world-size env vars, chooses the local GPU, and initializes the distributed process groups (`relax/distributed/ray/train_actor.py:30-69`). Its `set_rollout_manager()` implementation stores the rollout manager and, in `fully_async` mode only, also fetches train parallel config and the weight-sync lock from rollout (`relax/distributed/ray/train_actor.py:124-138`).

For the current sync-path MI210 run, that means the base-class `set_rollout_manager()` is not doing much beyond wake-up bookkeeping if the actor had already been offloaded (`relax/distributed/ray/train_actor.py:124-131`). This is one reason the current `set_rollout_manager` exception is more likely a symptom than a root cause.

### Megatron actor initialization and offload behavior

The heavy initialization work happens in `MegatronTrainRayActor._init()` (`relax/backends/megatron/actor.py:79-253`). The method:

1. patches checkpoint write behavior and base distributed setup (`relax/backends/megatron/actor.py:87-99`);
2. initializes Megatron and tracking (`relax/backends/megatron/actor.py:97-103`);
3. loads config/tokenizer serially across local ranks (`relax/backends/megatron/actor.py:105-115`);
4. builds the model and optimizer (`relax/backends/megatron/actor.py:140-145`);
5. creates `TensorBackuper` and weight updater objects in sync mode (`relax/backends/megatron/actor.py:154-190`);
6. clears memory and decides whether to offload (`relax/backends/megatron/actor.py:231-241`).

The current AMD path intentionally leaves the actor resident after init when `offload_train` is enabled but `fully_async` is false (`relax/backends/megatron/actor.py:234-241`). That earlier change moved the failure boundary forward by avoiding a sleep/wake transition before rollout-manager hookup.

The explicit offload hooks are `sleep()` and `wake_up()` (`relax/backends/megatron/actor.py:255-280`). `sleep()` destroys process groups and pauses the memory saver; `wake_up()` resumes it and reloads process groups. If the actor dies before either of these methods runs again, the root cause is elsewhere.

### Rollout startup is long and asynchronous

`RolloutManager` is created by `create_rollout_manager()` and forced onto the head node with `NodeAffinitySchedulingStrategy` (`relax/distributed/ray/placement_group.py:82-121`). This creation path blocks until `rollout_manager.get_num_rollout_per_epoch.remote()` returns, so rollout startup is part of controller startup rather than a background detail (`relax/distributed/ray/placement_group.py:101-121`).

The rollout implementation in `relax/distributed/ray/rollout.py` is large, but the important high-level fact is that it starts SGLang engines, health-checks them, and only then reports itself ready. In the current MI210 run, this startup window took about 10 minutes, which created a long period where the actor was already initialized but rollout was not yet ready.

## Code Flow

1. `parse_args()` and `main()` prepare runtime env and initialize Ray (`relax/entrypoints/train.py:53-67`).
2. `Controller` initializes the data system and registers services (`relax/core/controller.py:35-80`, `relax/core/controller.py:230-260`).
3. `Service` deploys the `Actor` and `Rollout` Serve deployments (`relax/core/service.py:22-89`).
4. `Actor.__init__()` allocates a `RayTrainGroup` and calls `async_init()` on each underlying train actor (`relax/components/actor.py:55-73`).
5. `RayTrainGroup` creates Ray actors with the prepared runtime env and placement-group schedule (`relax/distributed/ray/actor_group.py:47-106`).
6. `MegatronTrainRayActor._init()` builds Megatron state, loads checkpoint weights, and prepares weight update/offload structures (`relax/backends/megatron/actor.py:79-253`).
7. `create_rollout_manager()` starts the rollout side and waits for it to finish initialization (`relax/distributed/ray/placement_group.py:82-121`).
8. After rollout is ready, the controller calls `Actor.set_rollout_manager()` (`relax/core/controller.py:385`).
9. `Actor.set_rollout_manager()` forwards into `RayTrainGroup.set_rollout_manager()` and then `MegatronTrainRayActor.set_rollout_manager()` (`relax/components/actor.py:75-84`, `relax/distributed/ray/actor_group.py:165-166`, `relax/distributed/ray/train_actor.py:124-138`).
10. If the underlying worker already died during the rollout startup window, Ray surfaces `ActorDiedError` at this point.

## Configuration & Usage

### Practical reading order

If you want to learn this path efficiently, read files in this order:

1. `relax/entrypoints/train.py`
2. `relax/core/controller.py`
3. `relax/core/service.py`
4. `relax/components/actor.py`
5. `relax/distributed/ray/actor_group.py`
6. `relax/distributed/ray/train_actor.py`
7. `relax/backends/megatron/actor.py`
8. `relax/distributed/ray/placement_group.py`
9. `relax/distributed/ray/rollout.py`

That sequence keeps orchestration ahead of backend detail.

### Practical debugging loop

Use the same loop every time:

1. Run a bounded foreground validation first.
2. Find the first visible failure timestamp in the run log.
3. Extract the actor id and pid from the visible exception.
4. Search `/tmp/ray/session_latest/logs` by pid and worker id.
5. Compare controller timestamps with worker-side logs.

Useful commands:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl
timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh
```

```bash
ls -1t log/amd-qwen3-4b-2gpu-*.log | head -n 1
```

```bash
rg -n "3364142|d93e63769a3183ce779591fc02000000" /tmp/ray/session_latest/logs -g '*'
```

```bash
ls /tmp/ray/session_latest/logs/worker-*3364142*
ls /tmp/ray/session_latest/logs/python-core-worker-*3364142*
```

### Current MI210 failure boundary

For the latest known failing run:

- visible call path: `Controller.run_all_services -> Actor.set_rollout_manager -> RayTrainGroup.set_rollout_manager -> MegatronTrainRayActor.set_rollout_manager`
- failing actor id: `d93e63769a3183ce779591fc02000000`
- worker pid: `3364142`
- worker id: `e43612c668cd566ce6193cb1d4cfb47ba58110c54e6a4796b09f7f11`

The critical detail is that GCS reported the worker exit at `2026-04-17 04:32:20 UTC`, before the controller attempted `set_rollout_manager` at `2026-04-17 04:32:35 UTC`. That means `set_rollout_manager` is only the first place the system notices the death, not the place that caused it.

## Relevant Code

| Component | File | Lines |
|-----------|------|-------|
| training entrypoint | `relax/entrypoints/train.py` | 53-98 |
| controller setup | `relax/core/controller.py` | 35-80, 168-188, 230-260, 385 |
| service wrapper | `relax/core/service.py` | 22-89, 146-159 |
| actor deployment | `relax/components/actor.py` | 22-84 |
| train-group allocation | `relax/distributed/ray/placement_group.py` | 72-121 |
| ray train group | `relax/distributed/ray/actor_group.py` | 31-106, 165-166 |
| base train actor | `relax/distributed/ray/train_actor.py` | 30-69, 124-138 |
| megatron actor init/offload | `relax/backends/megatron/actor.py` | 79-280 |
| rollout manager implementation | `relax/distributed/ray/rollout.py` | 57-260 |

## Key Takeaways

- The most useful mental split is orchestration first, backend second.
- `ActorDiedError` at `set_rollout_manager` does not imply that method caused the crash; always verify the worker death timestamp in Ray logs.
- `Service` and `Actor` are mostly forwarding layers. Real root causes usually live in the worker logs or rollout startup path.
- In sync mode, `TrainRayActor.set_rollout_manager()` is lightweight, so a dead worker there usually died earlier.
- The current MI210 failure happens during the long rollout startup window after Megatron actor init, not during initial checkpoint load and not during the first train step.
