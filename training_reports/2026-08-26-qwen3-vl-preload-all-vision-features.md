# Qwen3-VL full-dataset vision-feature preload

- **Status:** Complete. Slurm job `160198` ran eight accepted profiles on four MI210 GPUs and 16 physical CPU
  cores.
- **Question:** Did lazy cache filling hide a steady-state training-cycle benefit from complete feature reuse?
- **Answer:** No. Preloading eliminated repeated encoding and almost all feature transport, but did not produce a
  statistically supported training-cycle speedup.

## Motivation

Matched job `154079` measured `11.056925 s` per training cycle for CPU vision without caches and `11.212820 s` with
both caches enabled lazily. Enabling both caches was `1.504%` slower, with a 95% confidence interval spanning
`-10.170%` to `7.161%`. This follow-up removes cold-fill cycles by loading every feature from the finite 64-image
dataset into both caches before rollout cycle 0.

The two caches are named by their owners:

- **CPU cache:** the byte-bounded LRU cache in the CPU `VisionEncoder` service.
- **SGLang cache:** the byte-bounded process-local LRU cache in the SGLang processor.

The preload performs no generation, reward calculation, optimizer work, or training-sample consumption. Dataset
cursor, order, shuffle state, and fingerprint must remain unchanged.

## Implemented interface and barrier

The opt-in interface is disabled by default:

```text
--preload-vision-features
PRELOAD_VISION_FEATURES=0|1
POST /relax/vision-features/cache
```

Preloading requires CPU vision, positive capacities for both caches, a finite eager global dataset, streaming
disabled, and exactly one SGLang rollout engine. Before any service loop or actor training starts, the controller:

1. snapshots the complete dataset without advancing it;
2. runs the production image processor and CPU encoder;
3. validates feature identity, revision, schema, grid, dtype, and immutable streams;
4. deduplicates by the complete immutable identity;
5. stores every unique feature through the admission-only SGLang endpoint;
6. publishes an identity only after SGLang confirms storage;
7. proves both caches retain all 64 identities with zero eviction;
8. records a versioned preload artifact;
9. resets interval telemetry while preserving cumulative preload counters; and
10. releases the barrier and starts training.

Any capacity, identity, dataset-state, or admission error aborts before training. The admission endpoint performs no
tokenization, prefill, decode, generation, or KV-cache creation.

The implementation is in:

- `relax/engine/rollout/data_source.py` for non-consuming dataset snapshots;
- `relax/backends/sglang/precomputed_vision.py` and `sglang_engine.py` for admission-only storage;
- `relax/distributed/ray/rollout.py` and `relax/core/controller.py` for the pre-cycle-0 barrier;
- `scripts/debug/qwen3_vl_preload_all_vision_features.sh` for the matched runner; and
- `examples/visual_xor/measure_full_dataset_preload.py` for strict analysis.

## Matched experiment

The two settings were:

| Setting | CPU cache | SGLang cache | Before cycle 0 |
|---|---:|---:|---|
| No caches | One-byte capacity | Disabled | No preload |
| Full-dataset preload | 1 GiB | 1 GiB | Store all 64 features |

Both settings used four MI210 GPUs, 16 physical CPU cores, one one-thread CPU vision replica, rollout batch 32, two
samples per prompt, exactly 64 responses, global batch 32, two optimizer intervals, 22 rollout cycles, maximum
staleness four, fully asynchronous execution, sequential data without shuffle, dynamic batching disabled, no
evaluation, and no checkpoint writes.

Four matched repeats counterbalanced execution order:

| Repeat | Train seed | Rollout seed | Execution order |
|---|---:|---:|---|
| `repeat_1` | 1234 | 42 | No caches; full-dataset preload |
| `repeat_2` | 1235 | 43 | Full-dataset preload; no caches |
| `repeat_3` | 1236 | 44 | No caches; full-dataset preload |
| `repeat_4` | 1237 | 45 | Full-dataset preload; no caches |

Each complete run is an experimental unit. Cycles within a run are repeated observations, not independent
experiments. For each repeat, the raw cycle-0-inclusive training-cycle speedup is:

$$
S_{\mathrm{preload}} = 1 -
\frac{\overline{T}_{\mathrm{full\ dataset\ preload},0:21}}
{\overline{T}_{\mathrm{no\ cache},0:21}}
$$

Across $K$ cycles, the preload-inclusive speedup is:

$$
S_{\mathrm{including\ preload}}(K) =
1 -
\frac{T_{\mathrm{preload}} + \sum_{i=0}^{K-1}T_{\mathrm{full\ dataset\ preload},i}}
{\sum_{i=0}^{K-1}T_{\mathrm{no\ cache},i}}
$$

When the preloaded run is faster, the cycles needed to recover preload time are:

$$
K_{\mathrm{break\ even}} =
\left\lceil
\frac{T_{\mathrm{preload}}}
{\overline{T}_{\mathrm{no\ cache}} - \overline{T}_{\mathrm{full\ dataset\ preload}}}
\right\rceil
$$

The analyzer reports every repeat's speedup, the paired Student-$t$ 95% confidence interval across the four
repeats, the 22-cycle preload-inclusive speedup, and the cycles needed to recover preload time for each repeat.

## Acceptance criteria

An accepted profile must contain:

- cycles 0 through 21, each with exactly 64 accepted responses and valid-action rate `1.0`;
- exactly two optimizer completions per cycle and successful final weight synchronization;
- all 64 preloaded features, complete identity/grid/byte metadata, and zero cache eviction;
- 32 CPU-cache hits and 32 SGLang-cache identity hits in every preloaded cycle, with no miss, encode, republish, or
  inline publication;
- 32 requests, misses, backend forwards, and encoded images in every no-cache cycle;
- proof that preloading performed zero generation and training work and preserved dataset state;
- no evaluation, no checkpoint write, complete controller shutdown, and owned-runtime cleanup.

Incomplete or contaminated runs are rejected rather than partially pooled.

## Execution

The five-minute foreground gate ran in job `160119` and reached real rollout, optimizer, no-cache accounting, and
owned-runtime cleanup before the expected timeout. Two full attempts were rejected:

- the first encountered an intermittent RCCL bind collision on port `11963` during the first lazy weight broadcast;
- the second lost the allocation when a fallible monitoring command exited the allocation-control shell.

Neither attempt entered the accepted analysis. Slurm job `160198` then completed the full matrix on
`auh7-1b-gpu-315`. All eight profiles exited zero, producing 176 rollout cycles, 11,264 valid responses, and 352
optimizer completions.

The accepted artifact directory is:

```text
benchmark_results/cpu_vision/full_dataset_preload_retry2_20260827_081348
```

The checked analysis artifact is:

```text
benchmark_results/cpu_vision/full_dataset_preload_retry2_20260827_081348/full_dataset_preload_analysis.json
```

## Result

Mean one-time preload time was `8.660052 s`. The raw cycle-0-inclusive speedups for `repeat_1` through `repeat_4`
were `0.557%`, `1.287%`, `3.005%`, and `-3.552%`. Their paired mean was `0.324%`, with a 95% confidence interval
from `-4.100%` to `4.748%`. Complete cache residency therefore did not demonstrate a training-cycle speedup.

Charging the one-time preload over 22 cycles produced repeat-level effects of `-2.623%`, `-1.894%`, `-0.231%`, and
`-6.807%`. The paired mean was `-2.889%`, with a 95% confidence interval from `-7.340%` to `1.562%`.

The first three repeats needed 126, 55, and 24 cycles to recover preload time. `repeat_4` never recovered it
because the preloaded run was slower. The mean across the three recovering repeats, `68.33` cycles, is conditional and
descriptive; mixed speedup signs and the confidence interval prohibit an unconditional recovery claim.

The cache mechanism itself worked:

| Metric | No caches | Full-dataset preload | Relative reduction |
|---|---:|---:|---:|
| Mean rollout time | `2.799012 s` | `1.069061 s` | `61.806%` |
| Mean service RTT | `1.237178 s` | `0.028158 s` | `97.724%` |
| Request-body bytes | — | — | `99.824%` |
| Serialization time | — | — | `71.158%` |

Mean weight-update time changed by only `+0.110%`. Training and weight synchronization remained on the critical
path, so the large encoding and transport reductions did not yield a supported end-to-end speedup.

Per-device idle VRAM was byte-identical before and after the matrix. The owned runner stopped all eight Ray runtimes
after their profiles.

## Decision

Keep full-dataset preload as an opt-in correctness and upper-bound experiment. Do not present it as a
training-throughput optimization for `prompts32_samples2`. Any further cache optimization must first demonstrate a
workload where rollout or feature transport paces the actor; otherwise, investigate training and weight-update time.
