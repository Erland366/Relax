# Qwen3-VL comparison of vision-device and cache settings

- **Date:** 2026-08-25
- **Status:** Slurm job `154079` completed all 16 profiles and passed the analysis gates.
- **Question:** Does moving frozen vision computation to CPU, enabling the CPU cache, or enabling the SGLang cache
  improve complete training-cycle time under one matched RL workload?

## Primary metric

The end-to-end boundary is:

$$
T_{\mathrm{training\ cycle}} = T_{\mathrm{train\ wait}} + T_{\mathrm{train\ scope}}
$$

Relax reports this value as `perf/step_time`. The train scope already contains both Megatron optimizer intervals,
health checks, the fully asynchronous weight update, and barriers. Adding those timers again would double-count
work. Because `NUM_STEPS_PER_ROLLOUT=2`, the corresponding rates are:

$$
\mathrm{training\ cycles/hour} = \frac{3600}{\overline{T}_{\mathrm{training\ cycle}}}
$$

$$
\mathrm{optimizer\ intervals/hour} =
\frac{2 \times 3600}{\overline{T}_{\mathrm{training\ cycle}}}
$$

## Compared settings

| Setting | Vision compute | CPU cache | SGLang cache |
|---|---|---:|---:|
| GPU vision | Frozen GPU encoder | Not used | Disabled |
| CPU vision without caches | One CPU replica, one thread | Effectively disabled with a one-byte capacity | Disabled |
| CPU vision with CPU cache | One CPU replica, one thread | 1 GiB | Disabled |
| CPU vision with CPU and SGLang caches | One CPU replica, one thread | 1 GiB | 1 GiB |

Every setting used `prompts32_samples2`: 32 prompts, two samples per prompt, 64 accepted responses, global batch
32, two optimizer intervals, fully asynchronous execution, maximum staleness four, dynamic batching disabled,
sequential data, no evaluation, and no checkpoint writes. Each run contained 22 cycles; cycles 0 and 1 were warm-up,
and cycles 2 through 21 were analyzed.

Four matched repeats balanced execution position:

| Repeat | Train seed | Rollout seed | Execution order |
|---|---:|---:|---|
| `repeat_1` | 1234 | 42 | GPU; CPU without caches; CPU with CPU cache; CPU with both caches |
| `repeat_2` | 1235 | 43 | CPU without caches; CPU with CPU cache; CPU with both caches; GPU |
| `repeat_3` | 1236 | 44 | CPU with CPU cache; CPU with both caches; GPU; CPU without caches |
| `repeat_4` | 1237 | 45 | CPU with both caches; GPU; CPU without caches; CPU with CPU cache |

Each complete run is an experimental unit. The 20 steady cycles estimate that run's mean; they are not treated as
20 independent experiments. The analyzer computes one paired speedup per repeat:

$$
S_{\mathrm{CPU\ vision}} =
1 - \frac{T_{\mathrm{CPU\ without\ caches}}}{T_{\mathrm{GPU}}}
$$

$$
S_{\mathrm{CPU\ cache}} =
1 - \frac{T_{\mathrm{CPU\ cache}}}{T_{\mathrm{CPU\ without\ caches}}}
$$

$$
S_{\mathrm{SGLang\ cache}} =
1 - \frac{T_{\mathrm{CPU\ and\ SGLang\ caches}}}{T_{\mathrm{CPU\ cache}}}
$$

$$
S_{\mathrm{all\ caching}} =
1 - \frac{T_{\mathrm{CPU\ and\ SGLang\ caches}}}{T_{\mathrm{CPU\ without\ caches}}}
$$

The report gives all four repeat-level effects, their mean and standard deviation, and the paired Student-$t$ 95%
confidence interval with three degrees of freedom.

## Result

| Setting | Mean training-cycle time |
|---|---:|
| GPU vision | `11.931008 s` |
| CPU vision without caches | `11.056925 s` |
| CPU vision with CPU cache | `11.108226 s` |
| CPU vision with CPU and SGLang caches | `11.212820 s` |

Moving frozen vision computation from GPU to CPU improved training-cycle time by `7.328%`; the paired 95%
confidence interval was `3.711%` to `10.944%`. Under this workload:

- enabling the CPU cache was `0.544%` slower than CPU vision without caches;
- enabling the SGLang cache in addition to the CPU cache was another `1.009%` slower; and
- enabling both caches was `1.504%` slower than CPU vision without caches.

The 95% confidence interval for the speedup from enabling both caches was `-10.170%` to `7.161%`. The experiment therefore
demonstrates neither a caching speedup nor a definitive caching slowdown at the complete training-cycle boundary.

## Mechanism evidence and claim boundary

Completed cache-reuse job `142817` established that the SGLang cache changes the intended mechanism:

| Metric | CPU cache only | CPU and SGLang caches | Relative reduction |
|---|---:|---:|---:|
| Mean rollout time | `1.344446 s` | `1.261462 s` | `6.172%` |
| Mean per-cycle p95 service RTT | `0.153542 s` | `0.108549 s` | `29.303%` |
| Request-body bytes | — | — | `99.824%` |
| Serialization time | — | — | `77.939%` |

The rollout-time reduction did not pass the prespecified `10%` adoption gate. These are mechanism measurements,
not evidence of a training-cycle speedup. The matched throughput benefit comes from moving frozen vision computation
off the GPU, not from caching it under `prompts32_samples2`.

The vision tower is too small for a VRAM-only claim. The paper contribution should center on the immutable frozen
vision-feature path, verified reuse by both model consumers, and the measured boundary between subsystem savings and
the asynchronous training critical path.

## Acceptance criteria and artifact

Every accepted profile contained cycles 0 through 21, exactly 64 responses and two optimizer intervals per cycle,
valid-action rate `1.0`, complete CPU service telemetry when applicable, no SGLang republish, final synchronization,
no evaluation, no checkpoint write, and clean Ray Serve shutdown. Grouped-GPU drift remains an independent known
limitation; the existing `0.01` gates are unchanged, so GPU vision is a performance comparison rather than the
semantic reference for that grouped path.

The authoritative artifact is:

```text
benchmark_results/cpu_vision/slurm_154079_vision_settings/vision_settings.json
```

The run used a three-hour, four-MI210, 16-CPU exclusive-node allocation. The analysis derives paper numbers only
from the versioned artifact; request bytes, serialization, and RTT support mechanism claims, while
`perf/step_time` and optimizer intervals per hour support training-throughput claims.
