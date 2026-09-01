# Naming the CPU/GPU vision work

This document defines the public vocabulary for the frozen Qwen3-VL vision-encoder work. Code, command-line
arguments, environment variables, metrics, analysis artifacts, launchers, reports, and figures must use the same
terms.

## Principles

1. Name the observable mechanism or setting. Do not require readers to decode experiment IDs.
2. Use `gpu` and `cpu` for devices. Do not use `native` as a synonym for GPU.
3. Use `fixed` when one device is selected for the run and `automatic` when a versioned plan selects a device per
   cycle. The current method reads a complete plan before training; it is not online adaptation.
4. Name caches by owner: `CPU cache` and `SGLang cache`. Do not number them as tiers.
5. Use `preload all vision features` for the optional operation that fills both caches from the finite dataset before
   cycle 0. It is not an oracle.
6. Name a repeated run `repeat_N`. A rollout cycle is an observation within a run, not an independent repeat.
7. Name a configuration by its values. For example, use `prompts32_samples2` and `replicas2_threads2`, not `D2`
   and `2x2`.
8. Reserve `mock` for an actual test double or deliberately fabricated model implementation. Small models, reduced
   layers, short workloads, and smoke runs are not mocks merely because they are small.

## Canonical terms

| Term | Exact meaning |
| --- | --- |
| GPU vision | The model's vision encoder executes on the GPU. |
| CPU vision | The frozen vision encoder executes in the CPU Ray Serve service. |
| skip GPU vision encoder | SGLang and Megatron do not materialize the GPU vision-encoder weights. |
| keep GPU vision encoder | GPU vision-encoder weights remain available, including for automatic device choice. |
| fixed device | The entire run uses the configured vision device. |
| automatic device choice | A versioned `VisionDevicePlan` selects CPU or GPU for each rollout cycle. |
| vision-device plan | The precomputed JSON artifact containing measured time models, cycle workloads, and choices. |
| vision time model | A `VisionTimeModel` containing fitted non-negative timing coefficients for one device. |
| minimum gap | The minimum predicted relative time difference required to change devices. Within this band, the
previous device is retained. |
| CPU cache | The byte-bounded LRU cache owned by the CPU `VisionEncoder` service. |
| SGLang cache | The byte-bounded process-local LRU cache owned by the SGLang processor. |
| full-dataset preload | Filling both caches with every unique finite-dataset feature before rollout cycle 0. |
| training-cycle time | `perf/step_time`, equal to actor wait time plus training time. |
| rollout time | `perf/rollout_time`. |
| faster device | The faster of matched fixed-GPU and fixed-CPU observations for the same cycle. |
| match rate | Among cycles over the minimum gap, the fraction where automatic choice selected the faster device. |
| extra time | Automatic choice's relative additional time compared with the faster device for the matched cycle. |
| speedup | $1 - T_{candidate}/T_{baseline}$. Positive values mean the candidate is faster. |

## Replaced vocabulary

| Do not use | Use instead | Why |
| --- | --- | --- |
| native | GPU | `native` does not identify a device. |
| adaptive placement | automatic device choice | The current choice is computed from a plan before training, not
adapted online. |
| route | device choice | No network request is being routed by this decision. |
| calibration | timing measurements or vision-device plan | The artifact contains both fitted times and complete
cycle choices. |
| hysteresis | keep previous device within the minimum gap | The direct rule is easier to verify. |
| confident decision | measured gap exceeds the minimum gap | The threshold is not statistical confidence. |
| oracle route / d-oracle | faster device | This is an empirical matched baseline, not an oracle. |
| regret | extra time | The metric is relative time overhead, not decision-theoretic regret. |
| Tier 1 / Tier 2 | CPU cache / SGLang cache | Ownership matters more than an invented hierarchy. |
| cache prewarm / cache oracle | full-dataset preload | The operation fills known caches; it does not provide future
knowledge. |
| actor-cycle time | training-cycle time | The value is the complete `perf/step_time` boundary. |
| rollout wall | rollout time | The metric already has a direct runtime name. |
| M0–M3 | explicit device and cache settings | Experiment IDs hide the controlled variables. |
| D0–D3 | `promptsN_samplesN` | The name states the actual rollout fanout. |
| E0–E4 | parity, demand, scaling, worker-layout, or cache-reuse study | Stage numbers do not describe the test. |
| CPU resident / CPU omitted | CPU with GPU encoder kept / CPU with GPU encoder skipped | The object that is kept or
skipped must be named. |
| perception plane / representation plane | frozen vision-feature path | The direct mechanism is more precise. |

Established names from dependencies and the research domain remain unchanged when they are accurate. Examples
include Ray placement groups, PyTorch native operators, SGLang's `torch_native` attention backend, DeepStack,
`image_grid_thw`, LRU, p95, Student-t intervals, and Python test mocks.

## Versioned public interfaces

The naming migration intentionally breaks the earlier experimental interfaces. No aliases are provided.

- `--vision-encoder-device={gpu,cpu}`
- `--skip-gpu-vision-encoder`
- `--vision-device-mode={fixed,automatic}`
- `--vision-device-plan=PATH`
- `--vision-device-minimum-gap=FLOAT`
- `--preload-vision-features`
- `--vision-encoder-max-images-per-request=INT`
- `POST /relax/vision-features/cache`
- `VISION_DEVICE_CHOICE` and `VISION_FEATURE_PRELOAD` log records

The comparison schema names its three runs `gpu`, `cpu`, and `automatic`. Its direct aggregate fields are
`total_cycles`, `cycles_over_minimum_gap`, `device_matches`, `match_rate`, and `mean_extra_time`. The plan schema uses
`time_models`, `timing_model_fit`, `predicted_cpu_seconds_after_overlap`, `predicted_gap`, and `minimum_gap`.

The renamed plan, preload, and analysis artifacts use `schema_version: 2`. Readers must reject older schemas rather
than silently translating them during an experiment.
