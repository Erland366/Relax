---
name: ray-stale-live-state-triage
description: >
  Distinguish a genuinely running Relax job from a stale Ray live state after a training actor or one distributed rank has failed.
  Use when Ray or tmux still looks alive, but rollout is polling forever on `train_<n>`, only health traffic is fresh, or actor progress has stopped.
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
but the useful training worker is already gone or one distributed actor rank
has failed. On the MI210 path, Ray, the driver, rollout, SGLang, and even a
surviving actor rank can remain alive after a fatal actor error. The uncleared
`train_<n>` partition then hits the staleness gate, making Ray job status and
process presence misleading.

## When to Apply

Use this knowledge when:
- Ray still reports a job as `RUNNING`, or tmux still shows a live shell.
- `Rollout` keeps printing `Current partitions: ['train_<n>']`.
- `SGLangEngine` is only serving health or tiny keepalive-style traffic.
- `Actor training failed at step <n>` appears without a later matching
  `Actor training completed step <n>`.
- One DP actor rank is gone or failed while another rank remains alive.

Do NOT use when:
- every actor rank is alive and producing fresh completed training steps
- the failure is still in early rollout or actor initialization

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Misleading signal | Ray job still `RUNNING` | Useful workers can already be gone |
| Reliable wedge signature | `train_<n>` polled forever | Rollout produced data but the training consumer disappeared |
| Reliable worker check | All expected actor ranks are alive and completing steps | A surviving DP rank alone does not prove health |
| Latest W&B e2e stale-live case | `train_99` polled after actor PID 2216956 died | Ray still reported `RUNNING` and SGLang kept serving health checks |
| DP2 visual-XOR stale-live case | `train_0` and `train_1` remained after rank 0 failed | `MAX_STALENESS=1` permanently blocked the next rollout |
| First causal DP2 error | RCCL reduce-scatter OOM at actor step 0 | The rollout wait warning was downstream, not causal |

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

### Step 2: Check every actor rank and the latest completed step

Use:

```bash
ps -eo pid,etime,cmd | rg 'relax\.entrypoints\.train|MegatronTrainRayActor|SGLangEngine|RolloutManager|gcs_server|raylet'
```

Interpretation:

- if every `MegatronTrainRayActor` is gone, or only a subset of the expected
  DP ranks remains
- or the actor service logged a fatal training exception without a later
  completed step
- while `python3 -m relax.entrypoints.train`, `RolloutManager`,
  `SGLangEngine`, `gcs_server`, and `raylet` remain alive

then the run is wedged/orphaned, not healthy.

Do not use the presence of one actor PID as the health gate for DP or TP jobs.
Compare the expected actor world size with the live ranks and require fresh
`Actor training completed step` progress.

### Step 3: Reconstruct the partition/backpressure chain

Look for repeated transfer-queue polling without consumption progress:

```text
Current partitions: ['train_<n>']
```

If that repeats for minutes or hours, treat it as a dead consumer, not a slow
step.

For bounded staleness, identify the exact chain:

```text
actor or one actor rank fails
    -> train_<n> is never consumed and cleared
    -> later train partitions accumulate
    -> MAX_STALENESS blocks the next rollout
    -> rollout waits forever
```

The repeated rollout warning is evidence of the blocked consequence. It is
not the original error.

### Step 4: Find the first actor-side exception

The useful question is not “is the job live?” but “when did the last real
training-side progress happen?”

Check the latest run log and compare:

- first actor-side traceback or `Actor training failed` line
- last actor-side training line
- current time
- whether only health traffic is still changing

Inspect the exact failed worker's Ray log when the top-level traceback wraps
the error in `RayTaskError`, `ActorDiedError`, or `DistBackendError`. If only
health traffic is fresh, the run is functionally dead.

### Step 5: Check global GPU ownership before blaming model size

On ROCm, compare GPU-wide memory with the failing rank's PyTorch allocator:

```bash
fuser -v /dev/kfd
rocm-smi --showpids --showmemuse --showuse
```

A large gap between total device usage and
`torch.cuda.memory_allocated()` means memory is held outside that rank's
PyTorch allocator. Inspect other processes, stale workers, and allocation
placement before reducing DP or model size.

In the visual-XOR DP2 failure, GPU-wide usage was 62.44 GiB while the failing
rank reported only 2.77 GiB allocated. RCCL then failed to allocate 6 MiB.
A later clean run completed the same DP2 topology for 250 training cycles and 500
optimizer updates, proving the topology itself fit.

### Step 6: Stop and inspect worker logs before retrying

Once the actor is gone and `train_<n>` is wedged:

1. stop only the affected Ray job cleanly
2. inspect the dead worker-side Ray logs for that exact attempt
3. verify unexpected `/dev/kfd` owners are gone
4. only then patch or retry the next boundary

Do not start another blind retry from a stale-live state.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Trusted Ray `RUNNING` as proof of health | Ray job metadata outlived the useful training actor | Always check actor process presence directly |
| Trusted one surviving DP actor PID as proof of health | A collective peer had already failed, so the remaining rank could not complete training | Compare expected and live ranks, then require a newly completed actor step |
| Treated repeated `train_<n>` polling as slow progress | The consumer was gone, so the partition would never drain | A training partition waiting forever is a dead-consumer signature |
| Debugged rollout or SGLang while the actor was already gone | The visible healthy services were only the survivors | Find the missing worker first, then inspect its logs |
| W&B e2e looked alive after step 99 | Ray stayed `RUNNING`, SGLang health checks continued, and rollout kept polling `train_99` after `MegatronTrainRayActor` died | Treat fresh health traffic as survivor noise unless actor-side progress is fresh |
| Blamed DP2 for a step-0 RCCL OOM | GPU-wide occupancy was much larger than the failing rank's PyTorch allocation | Check global GPU ownership and validate from a clean allocation before changing topology |

## Configuration

```yaml
healthy_run_requires:
  all_expected_megatron_actor_ranks_live: true
  fresh_completed_actor_step: true
  train_partition_draining: true
misleading_signals:
  - ray_job_running
  - live_tmux_shell
  - sglang_health_traffic_only
  - one_surviving_distributed_actor_rank
stop_and_triage_when:
  - megatron_actor_rank_missing_or_failed
  - actor_failure_without_later_completed_step
  - rollout_waiting_on_train_partition
known_stale_live_examples:
  - log: log/amd-qwen3-4b-2gpu-20260424_134200.log
    partition: train_99
    ray_status: RUNNING
    missing_worker: MegatronTrainRayActor
    last_good_actor_boundary: "train_actor rollout=99: megatron train completed"
    next_visible_boundary: "saving checkpoint at iteration 99, then worker SYSTEM_ERROR"
  - log: log/visual-xor-refinement-20260722_105748.log
    partitions:
      - train_0
      - train_1
    max_staleness: 1
    ray_status: RUNNING
    first_causal_error: "DP rank 0 RCCL reduce-scatter OOM at actor step 0"
    downstream_symptom: "rollout waiting for data system to catch up"
    later_clean_control: "same DP2 topology completed 250 cycles and 500 optimizer updates"
```

## References

- Related reports:
  `training_reports/2026-07-26-relax-synthetic-rl-battlefield.md`,
  `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`, `megatron-hybrid-device-optimizer-rocm`
- Troubleshooting: `references/troubleshooting.md`
