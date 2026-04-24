---
name: ray-stale-live-state-triage
description: >
  Distinguish a genuinely running Relax job from a stale Ray live state after the training actor has already died.
  Use when: Ray or tmux still looks alive, but rollout is polling forever on `train_<n>` and progress has stopped.
metadata:
  short-description: "Triaging stale-live Ray states on Relax"
  tags:
    - ray
    - serve
    - relax
    - debugging
    - rocm
  domain: research
  created: 2026-04-20
  author: Codex
---

# Ray Stale Live State Triage

## General Description

This skill captures the failure mode where Relax looks alive from the outside,
but the useful training worker is already gone. It is meant for the MI210 path
where Ray, the driver, rollout, and SGLang can remain alive after the actor has
died, leaving the system stuck on a `train_<n>` partition and making Ray job status
misleading.

## When to Apply

Use this knowledge when:
- Ray still reports a job as `RUNNING`, or tmux still shows a live shell.
- `Rollout` keeps printing `Current partitions: ['train_<n>']`.
- `SGLangEngine` is only serving health or tiny keepalive-style traffic.

Do NOT use when:
- the actor is still alive and producing fresh training-side logs
- the failure is still in early rollout or actor initialization

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Misleading signal | Ray job still `RUNNING` | Useful workers can already be gone |
| Reliable wedge signature | `train_<n>` polled forever | Rollout produced data but the training consumer disappeared |
| Reliable worker check | `MegatronTrainRayActor` absent from `ps` | Stronger signal than dashboard/job state |
| Latest W&B e2e stale-live case | `train_99` polled after actor PID 2216956 died | Ray still reported `RUNNING` and SGLang kept serving health checks |

## Recommended Practice

Treat this as a worker-liveness triage problem first, not a dashboard problem.
On this stack, the state that matters is whether the actor is still alive and
whether the current `train_<n>` partition is draining.
For later runs, the partition may be `train_99` or any other `train_<n>`.

### Step 1: Check the pane, but do not trust it alone

Use:

```bash
env -u TMUX tmux capture-pane -t tmux-1 -p -S -120
```

If the pane only shows repeated:

```text
Rollout 99: waiting for data system to catch up. Current partitions: ['train_99']
```

the run is already suspicious.

### Step 2: Check the actual worker processes

Use:

```bash
ps -eo pid,etime,cmd | rg 'relax\.entrypoints\.train|MegatronTrainRayActor|SGLangEngine|RolloutManager|gcs_server|raylet'
```

Interpretation:

- if `MegatronTrainRayActor` is gone
- but `python3 -m relax.entrypoints.train`, `RolloutManager`, `SGLangEngine`,
  `gcs_server`, and `raylet` are still alive

then the run is wedged/orphaned, not healthy.

### Step 3: Confirm that the data system is stalled, not progressing

Look for repeated transfer-queue polling without consumption progress:

```text
Current partitions: ['train_<n>']
```

If that repeats for minutes or hours, treat it as a dead consumer, not a slow
step.

### Step 4: Use the latest actor-side log timestamp as the truth source

The useful question is not “is the job live?” but “when did the last real
training-side progress happen?”

Check the latest run log and compare:

- last actor-side training line
- current time
- whether only health traffic is still changing

If only health traffic is fresh, the run is functionally dead.

### Step 5: Stop and inspect worker logs before retrying

Once the actor is gone and `train_<n>` is wedged:

1. stop the run cleanly
2. inspect the dead worker-side Ray logs for that exact attempt
3. only then patch the next boundary

Do not start another blind retry from a stale-live state.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Trusted Ray `RUNNING` as proof of health | Ray job metadata outlived the useful training actor | Always check actor process presence directly |
| Treated repeated `train_<n>` polling as slow progress | The consumer was gone, so the partition would never drain | A training partition waiting forever is a dead-consumer signature |
| Debugged rollout or SGLang while the actor was already gone | The visible healthy services were only the survivors | Find the missing worker first, then inspect its logs |
| W&B e2e looked alive after step 99 | Ray stayed `RUNNING`, SGLang health checks continued, and rollout kept polling `train_99` after `MegatronTrainRayActor` died | Treat fresh health traffic as survivor noise unless actor-side progress is fresh |

## Configuration

```yaml
healthy_run_requires:
  live_megatron_actor: true
  train_partition_draining: true
misleading_signals:
  - ray_job_running
  - live_tmux_shell
  - sglang_health_traffic_only
stop_and_triage_when:
  - megatron_actor_missing
  - rollout_waiting_on_train_partition
known_stale_live_examples:
  - log: log/amd-qwen3-4b-2gpu-20260424_134200.log
    partition: train_99
    ray_status: RUNNING
    missing_worker: MegatronTrainRayActor
    last_good_actor_boundary: "train_actor rollout=99: megatron train completed"
    next_visible_boundary: "saving checkpoint at iteration 99, then worker SYSTEM_ERROR"
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`, `megatron-hybrid-device-optimizer-rocm`
- Troubleshooting: `references/troubleshooting.md`
