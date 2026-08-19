# Qwen3-VL Policy-Invariant Representation Cache

- **Date:** 2026-08-18
- **Status:** two-tier cache implementation and focused local regressions pass;
  E0-E3 completed; corrected E4 cache job `142817` is pending in Slurm;
  no live Tier 2 performance result is claimed yet
- **Scope:** immutable feature identity, producer compute reuse, SGLang host
  transport reuse, cache-miss recovery, feature-sticky routing, E1 demand
  analysis, and actor-deduplication decision gates
- **Out of scope:** persistent databases, cross-job storage, actor batch-schema
  replacement, asynchronous weight-update optimization, and unconditionally
  implementing live CPU-ViT dynamic batching

## Outcome

The useful research contribution is not that the Qwen3-VL visual tower is
large. For this checkpoint it is only 27,084,160 parameters, and the measured
four-device VRAM reduction from the complete CPU-omission path is meaningful
but not sufficient by itself for a systems paper.

The stronger result is a policy-invariant representation boundary for fully
asynchronous multimodal RL. When the complete visual tower, merger,
projection, processor contract, and output schema are frozen, the mapping
from processed image to final-plus-DeepStack features is immutable across
policy updates. Relax can therefore compute that representation independently
of the language-policy version, reuse it across rollout branches and cycles,
and feed the same semantic bundle to SGLang and Megatron without introducing
policy staleness.

The implementation now has two bounded, explicit host-memory tiers:

1. The CPU encoder's producer LRU avoids repeated visual forwards.
2. The SGLang processor's consumer LRU avoids repeated packed BF16/base64
   serialization, HTTP transport, and reconstruction after one inline
   publication.

Both tiers use the complete identity:

$$
(\text{feature schema},\ \text{vision revision},\ \text{feature ID})
$$

The feature ID itself is content addressed over processed pixels, grid,
dtype, visual configuration, schema, and the visual parameter revision.
Neither tier is persistent. A cache miss is an explicit protocol event, not a
silent semantic fallback.

## Representation and recovery protocol

Tier 1 remains the per-Ray-Serve-replica `ByteBoundedLRUCache`. Entries larger
than the capacity bypass admission without evicting existing entries. Every
encoder response carries the immutable feature bundle, feature identity,
replica identity, hit state, backend batch evidence, and a cumulative
replica-local telemetry snapshot.

Tier 2 is disabled by default:

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
claim remains separate from the warm-cache E4 ablation.

Client telemetry distinguishes:

- inline publications;
- ID-only requests;
- ID-only hits;
- ID-only misses; and
- inline republishes.

The grouped hit path does not call the packed BF16/base64 serializer. The
actor contract remains tensor-only; feature IDs and schema strings are not
inserted into Megatron's concatenate-and-broadcast dictionary.

## E0 correctness evidence

Slurm job `141944` repeated the established representation gates before the
demand sweep.

| Comparison | Token/text result | Maximum log-probability drift | Maximum margin drift | Maximum normalized probability drift | Gate |
|---|---:|---:|---:|---:|---:|
| Scalar native vs. scalar CPU | 8/8 match | `1.4305115e-6` | `1.0430813e-7` | `1.7429432e-8` | Pass |
| Grouped CPU `n=8` vs. scalar CPU | 64/64 match | `9.5367432e-7` | `1.0430813e-7` | `2.0303581e-8` | Pass |
| Grouped native `n=8` vs. scalar native | 64/64 match | `0.0393803` | `0.0625000` | `0.0145715` | Fail, preserved |

The Hugging Face feature and full-vocabulary parity artifact also passed. The
grouped-native failure remains an independent SGLang branch-score artifact;
its `0.01` gates were not weakened and it is not evidence against CPU
features.

## E1 live demand result

The checked analyzer pairs rollout cycle $i$ with actor cycle $i$, discards
cycles 0-1, and computes:

$$
\text{consumer demand}_i =
\frac{\text{unique feature IDs consumed in cycle }i}
{\text{actor step wall time}_i}
$$

It does not substitute isolated encoder throughput for consumer demand.

| Profile | Unique images per rollout | Peak demand (images/s) | Mean actor wait | Maximum actor wait | Mean rollout wall | Exact steady request accounting |
|---|---:|---:|---:|---:|---:|---:|
| D0, `8 x 8` | 8 | `0.810161` | `0.626992%` | `0.908897%` | `1.146834 s` | Yes |
| D1, `16 x 4` | 16 | `1.614181` | `0.534455%` | `0.690668%` | `1.587334 s` | Yes |
| D2, `32 x 2` | 32 | `3.240455` | `0.552462%` | `0.655000%` | `2.792857 s` | Yes |
| D3, `64 x 1` | 64 | `5.858836` | `0.495635%` | `0.614838%` | `5.635120 s` | No: 29 extra refill/filter encodes in cycle 11 |

D2 is the hardest RL-valid profile and did not reach the 5% actor-wait
saturation gate. D3 is a performance-only probe because $n=1$ has no
meaningful GRPO group advantage. It produced 64 accepted responses per cycle,
but the final interval performed 93 encodes for 64 unique identities. The
artifact records those 29 extra candidates instead of presenting D3 as an
exact-accounting RL run.

The global demand input for E2 is:

$$
5.8588356399\ \text{images/s}
$$

The required 25% headroom target is:

$$
1.25 \times 5.8588356399 = 7.3235445499\ \text{images/s}
$$

The source artifact is
`benchmark_results/cpu_vision/slurm_141944_e0_e1/e1_demand.json`.

## E2 execution

The first E2 submission, job `142800`, was cancelled after preflight showed
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
capability result remains recorded in the E2 artifact.

## E3 live topology result and E4 execution state

E3 compares the three missing equal-ViT-CPU D2 layouts, `1x4`, `2x2`, and
`4x1`. The completed E1 D2 `1x1` run remains the control, and the completed E1
D3 `1x1` run remains the maximum-demand probe because E2 selected `1x1`.

The first E3 launch, job `142802`, was cancelled without accepting a result.
With Ray advertising 16 CPUs, the one-replica/four-thread VisionEncoder was
alive, but the Rollout Serve replica remained `PENDING_CREATION`. Live `ray
status` showed one pending CPU demand while GPU placement groups and control
actors consumed the schedulable Ray CPU budget. This falsified the assumption
that the existing Relax control plane plus a four-CPU ViT layout fits in a
16-Ray-CPU advertisement, even though the exclusive Slurm allocation exposed
64 physical cores.

The corrected E3 job `142804` requested and advertised 20 CPUs and completed
with exit code zero in 17m54s. The ViT reservation remained exactly four CPUs
for every compared layout, so the topology comparison did not gain CPU-ViT
capacity from the correction. All expected replicas were observed in every
steady cycle, each cycle accounted for exactly 32 requests and encoded images,
and valid-action rate remained 1.0.

| Layout | Mean encoded images / summed backend wall (images/s) | Mean service RTT | Mean cycle-p95 RTT | Mean rollout wall | Mean actor wait | Aggregate replica RSS | Final request fractions |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 `1x1` context | `15.559` | `1.197 s` | `2.161 s` | `2.793 s` | `0.552%` | `2.26 GiB` | `100%` |
| `1x4` | `30.854` | `0.536 s` | `0.876 s` | `1.613 s` | `0.482%` | `2.27 GiB` | `100%` |
| `2x2` | `24.261` | `0.234 s` | `0.387 s` | `1.405 s` | `0.568%` | `4.41 GiB` | `53.6% / 46.4%` |
| `4x1` | `15.719` | `0.206 s` | `0.369 s` | `1.454 s` | `0.616%` | `8.58 GiB` | `25.3% / 25.3% / 25.0% / 24.5%` |

The backend-rate column divides images by the sum of per-request backend wall
times. It is an efficiency measure, not deployment wall throughput when
replicas execute concurrently. At an equal four-CPU reservation, `2x2`
provides the lowest rollout wall. `4x1` does not improve rollout wall over
`2x2` and nearly doubles RSS; `1x4` minimizes RSS but has higher service
latency. These equal-four-CPU results do not displace the system recommendation
of `1x1`: E2 already proves `2.975` times peak-demand capacity at batch one,
and every live layout keeps actor wait below 1%. Four reserved ViT CPUs improve
rollout latency without removing an actor bottleneck that does not exist.

E4 job `142805` started after successful E3 completion. Tier 1 completed, but
the first repeated ID-only Tier 2 requests exposed a JSON wire bug: the compact
payload retained `image_grid_thw` as a tensor. The HTTP client retried the
non-serializable request, so the job was cancelled and no Tier 2 result from it
is accepted. A red regression now requires an ID payload built from raw
features to pass `json.dumps`; the builder converts tensor grids to CPU integer
lists.

Job `142809` then completed the Tier 1 profile, but its Tier 2 profile stopped
after cycle 1. SGLang's early media loader treated
`precomputed_embedding_id` as a native image and rejected it before the
processor-level cache resolver. The outer Ray launcher returned zero despite
the failed Ray job, so the original Slurm wrapper incorrectly wrote a
completion message. Neither its partial Tier 1 result nor its two Tier 2
cycles are accepted as the matched ablation.

The corrective red tests require the early SGLang loader to pass only the
ID-only format through while leaving all other inputs on the original path.
The Slurm wrapper now rejects any profile missing vision, rollout, or actor
cycles 0-11 and generates the checked E4 artifact itself. A four-hour pending
submission, job `142816`, was cancelled before allocation and replaced by the
one-hour backfill job `142817`; only the reservation ceiling changed. Job
`142817` reruns both profiles within one allocation, using the completed E1 D2
run as cache-off context and the measured `1x1` winner with 1 GiB at each
enabled tier. Dynamic batching remains disabled throughout.

`examples.visual_xor.analyze_cpu_vision_cache` requires all ten paired steady
rollout/actor cycles and reports producer hit rate, SGLang ID hit rate, request
bytes, serialization wall, service RTT mean/p95, rollout wall, actor wait,
republishes, backend encodes, and valid-action integrity. Until job `142817`
completes and that artifact passes, the cache remains an implemented mechanism
with a byte-reduction hypothesis, not a live speedup claim.

## What the cache can and cannot improve

The producer is already ahead of the actor. E1's one-thread backend averaged
approximately 14.5-15.8 images/s while the maximum measured consumer demand
was 5.86 images/s. The paper must therefore not claim that adding more CPU
ViT capacity solves current actor starvation.

Tier 2 attacks a different boundary. Before caching, one D3 cycle sends 64
requests of approximately 700 KiB each, or about 44.8 MB of JSON request body.
After the 64-image working set is published, ID-only requests can remove
nearly all repeated feature-body traffic. This is a strong byte-reduction
hypothesis, but not yet an end-to-end speedup result: feature H2D and prefill
remain, and E1 actor wait was already below 1%.

The live cache ablation must compare the same layout and sample order with:

```text
Tier 1 producer cache: 0 vs. 1 GiB
Tier 2 SGLang cache:   0 vs. 1 GiB
```

Primary outcomes are request bytes, serialization wall, ID hit/miss/republish
counts, service RTT mean/p95, rollout wall, actor wait, SGLang RSS, and exact
accepted/refill accounting. Adoption still requires unchanged representation
and logit parity, no more than 10% p95 RTT regression, and either at least 10%
lower steady rollout wall or a smaller passing CPU reservation.

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
> schema-safe representation plane decouple visual computation and transport
> from policy updates without changing learning semantics, and where does
> reuse stop helping?

The central ablation should separate four mechanisms:

| Variant | GPU visual compute | CPU producer reuse | SGLang transport reuse | Purpose |
|---|---:|---:|---:|---|
| Native frozen vision | Yes | No | No | Policy and performance counterfactual |
| CPU inline, caches off | No | No | No | Worst-case decoupled baseline |
| CPU Tier 1 only | No | Yes | No | Isolate avoided visual forwards |
| CPU Tier 1 + Tier 2 | No | Yes | Yes | Isolate avoided repeated representation transport |

A later actor-table variant belongs in the paper only if its independent gate
passes. Dynamic batching also remains conditional on E2's 20% same-layout
gain.

The paper should emphasize:

- one immutable representation consumed by inference and training;
- explicit schema/revision identity and fail-fast invalidation;
- two independent cache placements for compute and transport;
- live overlap and demand-before-supply methodology;
- negative results, including grouped-native drift, low actor wait, and any
  failed batching/cache adoption gate; and
- cache locality and recovery under realistic replica routing.

It should not claim:

- a large-model VRAM breakthrough from a 27M-parameter visual tower;
- that E1 showed actor starvation;
- that byte reduction automatically implies training throughput;
- that D3 is a valid GRPO learning configuration;
- that a process-local host LRU is a persistent feature database; or
- that actor deduplication or dynamic batching has already been adopted.

## Verification

Red-Green TDD covered schema propagation, oversized Tier 1 bypass, disabled
compatibility, inline publication, ID-only hit/miss behavior, mismatched
revision/schema/grid rejection, actual SGLang processor integration, marked
HTTP 400 translation, scalar and grouped one-republish recovery, sticky
routing, launcher/runtime propagation, unsafe topology rejection, telemetry,
and actor gate thresholds.

The combined focused suite passed:

```text
193 passed, 17 warnings
```

After the two live E4 integration findings, the new JSON-wire,
early-media-loader, and Slurm completion-gate slice passed 56 tests. A broader
CPU-vision/SGLang/launcher regression slice then passed 191 tests with the
same 17 dependency deprecation warnings.

Python compilation, shell syntax checks, and `git diff --check` also passed.
Ruff is not installed in the project environment, and no dependency was
installed. The matched live Tier 2 performance artifact remains the next
blocking evidence; local unit tests are not presented as a live speed result.
