# Qwen3-VL reuse of frozen vision features

- **Date:** 2026-08-18
- **Status:** both cache implementations and focused local regressions pass;
  GPU/CPU parity and cache reuse completed; cache reuse establishes mechanism-level savings but does not pass
  the rollout-time adoption gate or establish a training-cycle speedup
- **Scope:** immutable feature identity, CPU cache compute reuse, SGLang host
  transport reuse, cache-miss recovery, feature-sticky routing, CPU vision demand
  analysis, and actor-deduplication decision gates
- **Out of scope:** persistent databases, cross-job storage, actor batch-schema
  changes, asynchronous weight-update optimization, and unconditionally
  implementing live CPU-ViT dynamic batching

## Outcome

The useful research contribution is not that the Qwen3-VL visual tower is
large. For this checkpoint it is only 27,084,160 parameters, and the measured
four-device VRAM reduction from CPU vision with the GPU encoder skipped is meaningful
but not sufficient by itself for a systems paper.

The stronger result is a frozen vision features boundary for fully
asynchronous multimodal RL. When the complete visual tower, merger,
projection, processor contract, and output schema are frozen, the mapping
from processed image to final-plus-DeepStack features is immutable across
policy updates. Relax can therefore compute that representation independently
of the language-policy version, reuse it across rollout branches and cycles,
and feed the same semantic bundle to SGLang and Megatron without introducing
policy staleness.

The implementation now has two bounded, explicit host-memory caches:

1. The CPU encoder's CPU cache LRU avoids repeated visual forwards.
2. The SGLang processor's consumer LRU avoids repeated packed BF16/base64
   serialization, HTTP transport, and reconstruction after one inline
   publication.

Both caches use the complete identity:

$$
(\text{feature schema},\ \text{vision revision},\ \text{feature ID})
$$

The feature ID itself is content addressed over processed pixels, grid,
dtype, visual configuration, schema, and the visual parameter revision.
Neither cache is persistent. A cache miss is an explicit protocol event, not a
silent semantic fallback.

## Representation and recovery protocol

CPU cache remains the per-Ray-Serve-replica `ByteBoundedLRUCache`. Entries larger
than the capacity bypass admission without evicting existing entries. Every
encoder response carries the immutable feature bundle, feature identity,
replica identity, hit state, backend batch evidence, and a cumulative
replica-local telemetry snapshot.

SGLang cache is disabled by default:

```bash
export SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=0
```

A positive value enables a process-local SGLang host LRU:

```bash
export SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=1073741824
```

The first grouped request serializes and publishes the complete packed BF16
feature. Later requests send only identity and grid. All publication, lookup,
and republish requests use `X-SMG-Routing-Key: <feature_id>`. Multiple engines
are rejected unless the router uses `consistent_hashing`, and enabled caches
require the Transformers SGLang implementation and exactly one tokenizer
worker.

An unknown identity raises a `ValueError` containing the stable marker
`RELAX_SGLANG_VISION_FEATURE_CACHE_MISS:` and compact identity JSON. The exact
marked-400 recovery path is covered locally: Relax translates only a valid
marked 400 into `SGLangVisionFeatureCacheMiss`, then republishes inline once.
Unrelated 400 responses, malformed markers, mismatched identities, mismatched
grids, and a second failure propagate normally. A live forced-miss recovery
claim remains separate from the warm-cache vision cache reuse ablation.

Client telemetry distinguishes:

- inline publications;
- ID-only requests;
- ID-only hits;
- ID-only misses; and
- inline republishes.

The grouped hit path does not call the packed BF16/base64 serializer. The
actor contract remains tensor-only; feature IDs and schema strings are not
inserted into Megatron's concatenate-and-broadcast dictionary.

## GPU/CPU correctness evidence

Slurm job `141944` repeated the established representation gates before the
demand sweep.

| Comparison | Token/text result | Maximum log-probability drift | Maximum margin drift | Maximum normalized probability drift | Gate |
|---|---:|---:|---:|---:|---:|
| Scalar GPU vs. scalar CPU | 8/8 match | `1.4305115e-6` | `1.0430813e-7` | `1.7429432e-8` | Pass |
| Grouped CPU `n=8` vs. scalar CPU | 64/64 match | `9.5367432e-7` | `1.0430813e-7` | `2.0303581e-8` | Pass |
| Grouped GPU `n=8` vs. scalar GPU | 64/64 match | `0.0393803` | `0.0625000` | `0.0145715` | Fail, preserved |

The Hugging Face feature and full-vocabulary parity artifact also passed. The
grouped-GPU failure remains an independent SGLang branch-score artifact;
its `0.01` gates were not weakened and it is not evidence against CPU
features.

## CPU vision demand result

The checked analyzer pairs rollout cycle $i$ with training-cycle $i$, discards
cycles 0-1, and computes:

$$
\text{consumer demand}_i =
\frac{\text{unique feature IDs consumed in cycle }i}
{\text{training-cycle time}_i}
$$

It does not substitute isolated encoder throughput for consumer demand.

| Profile | Unique images per rollout | Peak demand (images/s) | Mean actor wait | Maximum actor wait | Mean rollout time | Exact steady request accounting |
|---|---:|---:|---:|---:|---:|---:|
| prompts8_samples8, `8 x 8` | 8 | `0.810161` | `0.626992%` | `0.908897%` | `1.146834 s` | Yes |
| prompts16_samples4, `16 x 4` | 16 | `1.614181` | `0.534455%` | `0.690668%` | `1.587334 s` | Yes |
| prompts32_samples2, `32 x 2` | 32 | `3.240455` | `0.552462%` | `0.655000%` | `2.792857 s` | Yes |
| prompts64_samples1, `64 x 1` | 64 | `5.858836` | `0.495635%` | `0.614838%` | `5.635120 s` | No: 29 extra refill/filter encodes in cycle 11 |

prompts32_samples2 is the hardest RL-valid profile and did not reach the 5% actor-wait
saturation gate. prompts64_samples1 is a performance-only probe because $n=1$ has no
meaningful GRPO group advantage. It produced 64 accepted responses per cycle,
but the final interval performed 93 encodes for 64 unique identities. The
artifact records those 29 extra candidates instead of presenting prompts64_samples1 as an
exact-accounting RL run.

The global demand input for CPU scaling is:

$$
5.8588356399\ \text{images/s}
$$

The required 25% headroom target is:

$$
1.25 \times 5.8588356399 = 7.3235445499\ \text{images/s}
$$

The source artifact is
`benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand.json`.

## CPU scaling execution

The first CPU scaling submission, job `142800`, was cancelled after preflight showed
that node exclusivity left the batch shell with all 128 logical CPUs despite
`--cpus-per-task=16`. No result from that attempt is accepted.

The corrected wrapper executes both preflight and the benchmark in explicit
16-core Slurm steps. Job `142801` exposed 16 physical cores (32 SMT siblings
in the core-bound affinity mask), while the matrix hard-limited total ViT
worker threads to eight. It completed all 36 valid combinations of:

```text
replicas: 1, 2, 4
threads:  1, 2, 4, 8
batches:  1, 2, 4, 8
```

with 64 images and five repeats, then exited successfully in 7m58s.

The smallest passing live layout is one replica by one thread. Its explicit
batch-size-one throughput is `17.433295` images/s, a `2.975` capacity ratio.
The same layout reaches `18.820158` images/s at explicit batch eight, but the
gain is only `7.955%`, below the 20% dynamic-batching gate for the selected
layout.

Larger layouts do show offline batching eligibility: `1x4`, `1x8`, `2x4`, and
`4x2` gain `34.59%`, `61.21%`, `38.21%`, and `25.30%`, respectively. Those
layouts reserve four to eight CPUs even though one batch-size-one CPU already
provides almost three times peak demand. They therefore do not justify adding
live coalescing to the smallest passing configuration. The dynamic-batching
branch stops as a negative result at the winner; its positive oversized-layout
capability result remains recorded in the CPU scaling artifact.

## CPU worker-layout result and cache-reuse execution

CPU worker layouts compares the three missing equal-ViT-CPU prompts32_samples2 layouts, `1x4`, `2x2`, and
`4x1`. The completed CPU vision demand prompts32_samples2 `1x1` run remains the control, and the completed CPU vision demand
prompts64_samples1 `1x1` run remains the maximum-demand probe because CPU scaling selected `1x1`.

The first CPU worker layouts launch, job `142802`, was cancelled without accepting a result.
With Ray advertising 16 CPUs, the one-replica/four-thread VisionEncoder was
alive, but the Rollout Serve replica remained `PENDING_CREATION`. Live `ray
status` showed one pending CPU demand while GPU placement groups and control
actors consumed the schedulable Ray CPU budget. This falsified the assumption
that the existing Relax control plane plus a four-CPU ViT layout fits in a
16-Ray-CPU advertisement, even though the exclusive Slurm allocation exposed
64 physical cores.

The corrected CPU worker layouts job `142804` requested and advertised 20 CPUs and completed
with exit code zero in 17m54s. The ViT reservation remained exactly four CPUs
for every compared layout, so the topology comparison did not gain CPU-ViT
capacity from the correction. All expected replicas were observed in every
steady cycle, each cycle accounted for exactly 32 requests and encoded images,
and valid-action rate remained 1.0.

| Layout | Mean encoded images / summed backend wall (images/s) | Mean service RTT | Mean cycle-p95 RTT | Mean rollout time | Mean actor wait | Aggregate replica RSS | Final request fractions |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU vision demand `1x1` context | `15.559` | `1.197 s` | `2.161 s` | `2.793 s` | `0.552%` | `2.26 GiB` | `100%` |
| `1x4` | `30.854` | `0.536 s` | `0.876 s` | `1.613 s` | `0.482%` | `2.27 GiB` | `100%` |
| `2x2` | `24.261` | `0.234 s` | `0.387 s` | `1.405 s` | `0.568%` | `4.41 GiB` | `53.6% / 46.4%` |
| `4x1` | `15.719` | `0.206 s` | `0.369 s` | `1.454 s` | `0.616%` | `8.58 GiB` | `25.3% / 25.3% / 25.0% / 24.5%` |

The backend-rate column divides images by the sum of per-request backend wall
times. It is an efficiency measure, not deployment wall throughput when
replicas execute concurrently. At an equal four-CPU reservation, `2x2`
provides the lowest rollout time. `4x1` does not improve rollout time over
`2x2` and nearly doubles RSS; `1x4` minimizes RSS but has higher service
latency. These equal-four-CPU results do not displace the system recommendation
of `1x1`: CPU scaling already proves `2.975` times peak-demand capacity at batch one,
and every live layout keeps actor wait below 1%. Four reserved ViT CPUs improve
rollout latency without removing an actor bottleneck that does not exist.

vision cache reuse job `142805` started after successful CPU worker layouts completion. CPU cache completed, but
the first repeated ID-only SGLang cache requests exposed a JSON wire bug: the compact
payload retained `image_grid_thw` as a tensor. The HTTP client retried the
non-serializable request, so the job was cancelled and no SGLang cache result from it
is accepted. A red regression now requires an ID payload built from raw
features to pass `json.dumps`; the builder converts tensor grids to CPU integer
lists.

Job `142809` then completed the CPU cache profile, but its SGLang cache profile stopped
after cycle 1. SGLang's early media loader treated
`precomputed_embedding_id` as a GPU image and rejected it before the
processor-level cache resolver. The outer Ray launcher returned zero despite
the failed Ray job, so the original Slurm wrapper incorrectly wrote a
completion message. Neither its partial CPU cache result nor its two SGLang cache
cycles are accepted as the matched ablation.

The corrective red tests require the early SGLang loader to pass only the
ID-only format through while leaving all other inputs on the original path.
The Slurm wrapper now rejects any profile missing vision, rollout, or training
cycles 0-11 and generates the checked vision cache reuse artifact itself. A four-hour pending
submission, job `142816`, was cancelled before allocation and replaced by the
one-hour backfill job `142817`; only the reservation ceiling changed. Job
`142817` completed both profiles within one allocation on 2026-08-18 with exit
code zero. Dynamic batching remained disabled throughout.

`examples.visual_xor.measure_vision_cache_reuse` accepted all ten paired steady
cycles. CPU cache and SGLang identity hit rates were both 1.0, no republish was
observed, and valid-action rate remained 1.0. SGLang cache reduced request-body bytes
by `99.824%`, serialization wall by `77.939%`, and mean cycle-p95 service RTT by
`29.303%`. Mean rollout time changed from `1.344446 s` to `1.261462 s`, a
`6.172%` reduction, so the prespecified `10%` rollout-time adoption gate did
not pass.

Those are rollout-path mechanism metrics, not a training-throughput result.
In the same single run, the post-hoc mean `perf/step_time` changed from
`10.602945 s` to `11.073159 s`, or `4.434%` slower with SGLang cache. The profiles
were not repeated or order balanced, so that difference is a covariate rather
than a cache effect estimate. It motivated the four-repeat matched vision-setting comparison
documented on 2026-08-25.

## What the cache can and cannot improve

The CPU cache is already ahead of the actor. CPU vision demand's one-thread backend averaged
approximately 14.5-15.8 images/s while the maximum measured consumer demand
was 5.86 images/s. The paper must therefore not claim that adding more CPU
ViT capacity solves current actor starvation.

SGLang cache attacks a different boundary. Before caching, one prompts64_samples1 cycle sends 64
requests of approximately 700 KiB each, or about 44.8 MB of JSON request body.
After the 64-image working set is published, ID-only requests can remove
nearly all repeated feature-body traffic. Job `142817` confirmed that byte
reduction, but not an end-to-end speedup: feature H2D and prefill remain, and
CPU vision demand actor wait was already below 1%.

The live cache ablation must compare the same layout and sample order with:

```text
CPU cache:    disabled vs. 1 GiB
SGLang cache: disabled vs. 1 GiB
```

Primary outcomes are request bytes, serialization wall, ID hit/miss/republish
counts, service RTT mean/p95, rollout time, actor wait, SGLang RSS, and exact
accepted/refill accounting. Adoption still requires unchanged representation
and logit parity, no more than 10% p95 RTT regression, and either at least 10%
lower steady rollout time or a smaller passing CPU reservation.

## Actor deduplication boundary

Megatron currently repeats tensor feature dictionaries per sample. Changing
that is not a cache toggle: it requires a unique-feature table plus ordered
references throughout TransferQueue slicing, sampling, concatenation, and
device staging. The implementation therefore adds only a pure measurement
gate. Actor deduplication becomes eligible when either:

$$
\frac{\text{transported feature bytes}}{\text{unique feature bytes}} > 2
$$

or:

$$
\frac{\text{feature H2D time}}{\text{actor compute time}} > 0.10
$$

Exact thresholds do not pass. No actor cache or new batch schema has been
introduced.

## Paper framing and ablations

The defensible paper question is:

> When perception is frozen during asynchronous multimodal RL, can a
> schema-safe frozen vision-feature path decouple visual computation and transport
> from policy updates without changing learning semantics, and where does
> reuse stop helping?

The central ablation should separate four mechanisms:

| Setting | GPU vision compute | CPU cache reuse | SGLang cache reuse | Purpose |
|---|---:|---:|---:|---|
| GPU frozen vision | Yes | No | No | Policy and performance counterfactual |
| CPU inline, caches off | No | No | No | Worst-case decoupled baseline |
| CPU vision with CPU cache | No | Yes | No | Isolate avoided vision-encoder forwards |
| CPU vision with CPU and SGLang caches | No | Yes | Yes | Isolate avoided repeated feature transport |

A later actor-table variant belongs in the paper only if its independent gate
passes. Dynamic batching also remains conditional on CPU scaling's 20% same-layout
gain.

The paper should emphasize:

- one immutable representation consumed by inference and training;
- explicit schema/revision identity and fail-fast invalidation;
- two independent cache controls for compute and transport;
- live overlap and demand-before-supply methodology;
- negative results, including grouped-GPU drift, low actor wait, and any
  failed batching/cache adoption gate; and
- cache locality and recovery under realistic replica routing.

It should not claim:

- a large-model VRAM breakthrough from a 27M-parameter visual tower;
- that CPU vision demand showed actor starvation;
- that byte reduction automatically implies training throughput;
- that prompts64_samples1 is a valid GRPO learning configuration;
- that a process-local host LRU is a persistent feature database; or
- that actor deduplication or dynamic batching has already been adopted.

## Verification

Red-Green TDD covered schema propagation, oversized CPU cache bypass, disabled
compatibility, inline publication, ID-only hit/miss behavior, mismatched
revision/schema/grid rejection, actual SGLang processor integration, marked
HTTP 400 translation, scalar and grouped one-republish recovery, sticky
routing, launcher/runtime propagation, unsafe topology rejection, telemetry,
and actor gate thresholds.

The combined focused suite passed:

```text
193 passed, 17 warnings
```

After the two live vision cache reuse integration findings, the new JSON-wire,
early-media-loader, and Slurm completion-gate slice passed 56 tests. A broader
CPU-vision/SGLang/launcher regression slice then passed 191 tests with the
same 17 dependency deprecation warnings.

Python compilation, shell syntax checks, and `git diff --check` also passed.
Ruff is not installed in the project environment, and no dependency was
installed. The matched live SGLang cache performance artifact remains the next
blocking evidence; local unit tests are not presented as a live speed result.
