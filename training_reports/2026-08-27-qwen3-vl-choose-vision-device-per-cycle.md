# Qwen3-VL plan-based vision-device choice

- **Date:** 27 August 2026
- **Status:** Plan-based choice is implemented for a controlled workload; online Geo3K choice is intentionally
  disabled.

## Implemented behavior

- A versioned `VisionDevicePlan` stores non-negative GPU and CPU time models, declared cycle workloads, and the
  selected device for each cycle.
- `fixed` mode uses one configured device for the complete run.
- `automatic` mode reads the complete plan before training and uses its immutable CPU/GPU choice for each declared
  rollout cycle.
- A 5% default minimum gap prevents switching when the predicted difference is small; the previous device is kept
  inside that band.
- Choices are ordered by rollout ID and carried in coroutine-local state, so overlapping evaluation and training do
  not share mutable decision state.
- Automatic mode fails at startup when GPU encoder weights are skipped, full-dataset preload is enabled, partial
  rollout is enabled, vision weights are trainable, or the plan is invalid or incomplete.
- Structured `VISION_DEVICE_CHOICE` records and W&B-compatible metrics expose the selected device and predicted
  times.
- The timing-model builder uses exact small-dimensional non-negative least squares, records model rank, and reports
  any collinear predictor it drops.
- The visual-XOR workload builder alternates low- and high-resolution cycles.
- The matched comparison runs five repeats with counterbalanced execution order and reports match rate, extra time,
  and training-cycle speedup using direct setting names.
- Geo3K conversion, deterministic evaluation configuration, and an eight-GPU fixed-device launcher are present.
- Paper-performance launchers disable checkpoint writes by default.

## Validation completed

Focused runtime, analyzer, service, workload-builder, Geo3K-helper, and launcher tests passed. The workload builder
also ran against the local 64-row refinement dataset:

| Workload | Resolution | Visual tokens per image | Feature bytes per cycle |
|---|---:|---:|---:|
| Low resolution | 112 × 112 | 64 | 16,777,984 |
| High resolution | 448 × 448 | 196 | 51,380,992 |

Each workload contains 32 action-A and 32 action-B rows. These values validate the reduced checkpoint and are not
universal Qwen3-VL constants.

## What the current method is—and is not

The method reads an exact cycle map before rollout data is fetched. It is appropriate only for the controlled,
unshuffled, unfiltered, zero-staleness workload whose cycle-to-data mapping is known in advance. It is not an online
policy and does not observe arbitrary Geo3K batches at runtime.

The public terms are therefore:

- **timing measurements** for the CPU/GPU observations used to build the plan;
- **vision-device plan** for the versioned JSON input;
- **automatic device choice** for consuming that plan;
- **minimum gap** for the relative predicted difference required to switch;
- **match rate** for agreement with the faster matched device outside the minimum gap; and
- **extra time** for measured time above that faster device.

No statistical-confidence claim is attached to the minimum gap. The matched faster device is an empirical
comparison, not a theoretical upper bound.

## Hardware and data boundary

The allocation available during implementation exposed four authorized MI210 GPUs and 16 CPUs. The eight physical
GPUs visible to host ROCm tools belonged to more than one Slurm allocation; devices outside the active allocation
were not used. No Qwen3-VL-4B or Geo3K download, eight-GPU run, Ray launch, or training command was attempted.

The paper PDFs were also not imported because the node's HTTPS proxy rejected MLSys and arXiv. The approved
`add-resource` workflow failed before writing entries; imports must be retried from a network-enabled shell.

## Next accepted run order

1. Collect rank-aware CPU and GPU timing measurements over the controlled workload.
2. Build the versioned plan and run a two-cycle foreground device-switch smoke test.
3. Run the five-repeat controlled comparison and require every analyzer gate to pass.
4. Acquire an authorized eight-MI210 allocation and the public Qwen3-VL-4B and Geo3K assets.
5. Repeat the feature-and-logit parity ladder on 4B.
6. Run matched fixed-device Geo3K system profiles.
7. Implement runtime Geo3K batch-workload observation before enabling automatic Geo3K experiments.

The Geo3K launcher's automatic case currently fails with an actionable error. It must not silently run a
precomputed schedule under an online-choice description.
