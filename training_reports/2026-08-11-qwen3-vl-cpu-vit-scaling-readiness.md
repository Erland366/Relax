# Qwen3-VL CPU-ViT Scaling Readiness

- **Date:** 2026-08-11
- **Status:** implementation and local tests passed; E0-E4 hardware matrix not
  run because the current allocation has only two SMT threads of one physical
  core
- **Scope:** replica-aware live telemetry, demand-profile launch controls,
  allocation preflight, offline topology/resource measurement, and the
  conditional live-batching gate
- **Out of scope:** asynchronous weight-update optimization, a feature
  registry/database, sticky routing, and alternative CPU runtimes

## Outcome

The repository is ready to run E0-E3 on a new allocation with four MI210 GPUs,
16 scheduler CPUs, and at least eight distinct physical cores. The live CPU
encoder now returns the immutable Qwen3-VL final-plus-three-DeepStack feature
bundle together with replica-local work and resource telemetry. The rollout
manager aggregates the newest snapshot observed from every replica instead of
querying one load-balanced replica.

The offline runner now enforces an eight-CPU ViT budget, keeps only one process
layout alive, reuses that layout across explicit batch trials, and produces
the capacity, scaling-efficiency, process-CPU, peak-RSS, and batching-gate
evidence needed for E2.

No CPU layout or positive batch-wait timeout has been selected. The current
allocation exposes affinity CPUs `8,72`; both map to package zero, physical
core 16. The new capacity preflight therefore rejects it before Ray startup.
Implementing live dynamic batching without an E2 artifact would skip the
plan's 20% eligibility gate, so positive
`VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS` values fail loudly for now.

## Implemented measurement contract

Each successful `VisionEncoder.encode` response contains:

- the original `Qwen3VLFrozenVisionFeatures` object;
- serving replica and requested feature identities;
- cache-hit status;
- backend-forward wall time and actual backend batch size; and
- cumulative requests, cache work, backend work, encoded images, emitted
  feature bytes, backend wall time, process CPU time, monotonic snapshot time,
  and Linux peak RSS.

The rollout aggregation reports:

- expected, observed, active, idle, and not-yet-observed replicas;
- cumulative and interval request distribution per replica;
- cumulative and interval hits, misses, evictions, backend forwards, encoded
  images, emitted bytes, and backend seconds;
- cumulative and interval unique feature counts and duplicate-encode ratios;
- cumulative and interval actual backend batch-size mean, nearest-rank p95,
  and maximum;
- per-replica process CPU utilization and peak RSS; and
- existing client service-round-trip and packed-transport timing
  distributions, including mean and p95.

Out-of-order responses cannot replace a newer cumulative snapshot from the
same replica. Interval feature and backend-miss identity sets reset only after
collection, so repeated feature IDs remain visible in each rollout interval.
Legacy raw-feature responses now fail rather than silently bypassing
telemetry.

## Offline E2 contract

The intended command is:

```bash
python -m examples.visual_xor.benchmark_cpu_vision_scaling \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_train.parquet" \
  --output benchmark_results/cpu_vision/e2_scaling.json \
  --peak-unique-images-per-second "$E1_PEAK_DEMAND" \
  --replica-counts 1,2,4 \
  --thread-counts 1,2,4,8 \
  --batch-sizes 1,2,4,8 \
  --max-total-cpus 8 \
  --num-images 64 \
  --repeats 5
```

Layouts with more than eight reserved ViT CPUs are excluded. The capacity
recommendation is the smallest reservation that sustains at least `1.25`
times the measured E1 peak demand, with ties broken by higher throughput,
fewer replicas, fewer threads, and smaller explicit batch size.

For every replica/thread layout, explicit batching is eligible only when the
best batch size above one is at least 20% faster than batch size one. The
candidate batch size is the smallest one within 5% of the best batched
throughput. This artifact is a backend capability result, not proof that live
cross-request batching already exists.

## Live matrix to run

Every E1 profile produces 64 responses per rollout. Set `NUM_ROLLOUT=12`,
discard cycles 0-1, and analyze ten steady cycles.

| Profile | Rollout batch | Samples per prompt | Unique images | Interpretation |
|---|---:|---:|---:|---|
| D0 | 8 | 8 | 8 | Existing grouped baseline |
| D1 | 16 | 4 | 16 | RL-valid |
| D2 | 32 | 2 | 32 | Hardest RL-valid demand |
| D3 | 64 | 1 | 64 | Performance-only; no GRPO learning claim |

The CPU launcher accepts all four fanouts. Set
`CPU_VISION_CAPACITY_MODE=1`, `VISION_ENCODER_NUM_REPLICAS=1`,
`VISION_ENCODER_NUM_CPUS=1`, `VISION_ENCODER_OMIT_GPU_WEIGHTS=1`,
`VISION_ENCODER_CACHE_MAX_BYTES=1`,
`VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0`, and `ENABLE_EVAL=0` for E1.
Checkpoint saving remains disabled by the launcher.

After E2, compare D2 under `1x1`, `1x4`, `2x2`, and `4x1`, plus any distinct
E2 winner. If D2 does not raise `1x1` actor wait to at least 5%, compare only
`1x1` and the E2 winner under D3 as the maximum-demand saturation probe.

## Conditional E4 boundary

Live cross-request batching remains unimplemented. It may be added only after
E2 records a qualifying explicit-batch gain. At that point, use Red-Green TDD
for the asynchronous per-replica miss queue, heterogeneous grid splitting,
ordered feature identity, mixed hit/miss behavior, overflow, and shared error
propagation before running 1 ms and 5 ms pilots.

If the E2 gate fails, record the negative result and stop the batching branch.
If it passes, adoption still requires unchanged parity and either at least 10%
lower steady rollout wall at the same CPU reservation or a 25% smaller passing
CPU reservation while actor wait remains below 5%. Service p95 must not regress
by more than 10%.

## Local verification

The combined focused suite passed 108 tests. It covers response telemetry,
stale/out-of-order replica snapshots, interval aggregation, cache and duplicate
identities, resource evidence, matrix filtering, pool lifecycle, capacity
preflight, demand-profile overrides, and the staged batch-wait contract.
Hardware parity and live E0-E4 runs were not attempted because the allocation
fails the new preflight.

The repository-local `.venv` documented by the global contract is absent.
Tests requiring Ray/SGLang imports used the existing
`relaxrl_rocm_after_fix` environment with `ROCR_VISIBLE_DEVICES` unset. Ruff is
not installed in the available project environments, so validation used
focused pytest, Python compilation, shell syntax checks, `git diff --check`,
and a changed-line check against the repository's 119-column limit.

## Next boundary

Acquire the required allocation, rerun E0 parity, execute E1, then use its
measured peak consumer demand as the E2 input. Do not select a live layout or
implement batching before those artifacts exist. After the winning layout is
confirmed live, restore the 1 GiB cache and measure aggregate backend encodes
per unique feature ID. Only a ratio above `1.25` justifies feature-sticky
replica routing as the next experiment; it does not justify jumping directly
to a database or feature registry.
