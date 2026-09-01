# Frozen CPU Vision Encoder

Relax has an experimental, opt-in CPU service for the frozen visual module of
the local Qwen3-VL 0.37B visual-XOR checkpoint. It exists to test whether
static visual work can overlap rollout and actor work without introducing
policy staleness.

This is not a generic VLM backend. The first implementation is deliberately
narrow:

- Qwen3-VL images only; audio and video fail explicitly.
- PyTorch CPU inference only; llama.cpp and Prima.cpp are not used.
- Both the vision tower and every visual projection/merger must be frozen.
- Context parallelism and pipeline parallelism must both be one.
- The normal path remains unchanged unless
  `--vision-encoder-device cpu` is set.

## Data flow

```text
raw image
  -> Relax Hugging Face processor
  -> CPU vision_encoder Ray Serve service
       -> final projected visual embeddings
       -> three ordered DeepStack embedding streams
       -> byte-bounded, schema/revision-aware CPU cache
  -> one SGLang precomputed_embedding request per eligible prompt group
       -> byte-bounded SGLang cache in host memory
       -> later requests carry only immutable feature identity
       -> GPU SGLang n-way decode branches
  -> Megatron precomputed visual forward boundary
```

The canonical per-sample actor bundle is:

```text
image_grid_thw                  int64 [M, 3]
vision_embeds                   bf16  [N, H_text]
deepstack_visual_embeds_0       bf16  [N, H_text]
deepstack_visual_embeds_1       bf16  [N, H_text]
deepstack_visual_embeds_2       bf16  [N, H_text]
```

For the current checkpoint, `H_text=1024`, the visual grid for one 256×256
image produces `N=64`, and one complete bundle occupies approximately 512
KiB. The service reads only `model.visual.*` tensors from safetensors; it does
not materialize the language model or LM head.

The actor validates that the number of image placeholders equals both the
grid-derived merged-token count and the number of embedding rows. It then
lets the existing Qwen3-VL forward perform text-embedding replacement, MRoPE,
THD packing, and DeepStack injection. Raw pixel tensors are removed from the
actor payload.

SGLang receives the same final and DeepStack streams concatenated in its
Qwen3-VL precomputed-embedding layout. Relax installs a gated compatibility
patch in SGLang processes. During prefill, the adapter splits the packed
streams, scatters the final embeddings into processor-expanded image-token
positions, and passes the ordered DeepStack streams, visual mask, and MRoPE
positions directly to the Qwen3-VL language model. GPU-image and decode
requests retain SGLang's normal path. A batch mixing GPU and precomputed
items fails explicitly.

## Resource and cache configuration

`RESOURCE_JSON` keeps its existing `[num_services, num_gpus]` meaning. A CPU
encoder therefore uses:

```bash
export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,1],"actor_fwd":[1,1],"advantages":[1,0],"vision_encoder":[1,0]}'
```

CPU reservation and cache sizing are separate:

```bash
export VISION_ENCODER_DEVICE=cpu
export VISION_ENCODER_NUM_CPUS=8
export VISION_ENCODER_NUM_REPLICAS=1
export VISION_ENCODER_CACHE_MAX_BYTES=1073741824
export SGLANG_VISION_FEATURE_CACHE_MAX_BYTES=1073741824
export VISION_ENCODER_MAX_IMAGES_PER_REQUEST=8
export SKIP_GPU_VISION_ENCODER=0
```

The cache key includes the processed pixels, grid, output dtype, visual
configuration, feature-schema version, and a hash of every visual tensor.
Reusing a feature from a different checkpoint revision or representation
schema is therefore not allowed. Oversized entries bypass CPU cache admission
without evicting smaller resident entries.

`SGLANG_VISION_FEATURE_CACHE_MAX_BYTES` defaults to zero. A positive value
enables the separate process-local SGLang cache in the SGLang multimodal
processor. The first request publishes the packed BF16 tensor inline; later
requests send `feature_id`, `vision_revision`, `feature_schema_version`, and
the grid only. An exact marked miss produces one inline republish. Other HTTP
400 errors, identity mismatches, grid mismatches, and a second failure are not
retried through a hidden scalar fallback.

SGLang cache is job-lifetime host RAM, not a persistent registry or database. It
saves repeated BF16/base64 serialization on grouped cache hits, HTTP request
bytes, and SGLang base64 reconstruction. It does not remove
tokenizer-to-scheduler tensor transfer, feature H2D, multimodal prefill, or
actor transport. Enabled operation requires the PyTorch CPU-vision backend,
the SGLang Transformers implementation, and one tokenizer worker. Multiple
engines require `consistent_hashing`; publication, ID lookup, and republish
all use the content-addressed `feature_id` routing key.

`VISION_ENCODER_NUM_CPUS` is reserved **per replica**. For example, two
replicas with eight threads reserve sixteen Ray CPU slots, in addition to the
CPU slots needed by the other services. The replicas are request-level data
parallel workers, not DDP ranks. Each replica owns an independent LRU cache;
Ray Serve does not currently provide image-key-sticky routing for this
deployment, so repeated images can be encoded once per replica. Measure this
duplication before increasing the replica count.

Run the dedicated fully asynchronous visual-XOR recipe with:

```bash
bash scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh
```

The correctness recipe defaults to one reserved CPU core per encoder replica
so the service remains schedulable beside Ray and Serve control actors on
CPU-constrained GPU allocations. Override `VISION_ENCODER_NUM_CPUS` only for a
measured scaling run on a CPU-rich allocation. No model checkpoints are
written by that recipe. W&B and the ordinary run log remain enabled. Set
`ENABLE_EVAL=0` for a training-only profiling run; the default `1` preserves
the established evaluation configs and intervals.

## Correctness and parity gates

Run the fixed-image Hugging Face parity gate before skipping the GPU
vision encoder:

```bash
python -m examples.visual_xor.validate_cpu_vision_parity \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/parity.json \
  --device cuda:0 \
  --num-images 8
```

The diagnostic compares the CPU encoder against the GPU visual tower
for the final projected stream and all three DeepStack streams. It then
compares the GPU raw-image Hugging Face forward with a language-only
forward consuming the CPU features. The artifact fails unless:

- every feature stream has cosine similarity at least `0.999` and satisfies
  `rtol=0.01`, `atol=0.01` for at least `99.99%` of values, with maximum
  absolute error at most `0.05`;
- mean full-vocabulary KL is at most `1e-3`;
- every sample KL is at most `1e-2`;
- every sample has the same top token; and
- the absolute probability delta for both action tokens is at most `0.01`.

This local diagnostic has full-vocabulary logits. The live Relax path
currently exposes the generated-token log probabilities from SGLang and
Megatron, not both full vocabulary tensors. For live cross-backend checking,
use `--dump-details` together with
`train_rollout_logprob_abs_diff`, the corresponding probability difference,
and the TIS mismatch metrics. Do not describe those sampled-token diagnostics
as an exact full-vocabulary KL.

The dedicated live SGLang gate compares GPU and precomputed requests inside
one server that keeps its GPU vision weights. It requests the exact next-token log-probabilities
for A and B, flushes the cache between paths, writes every raw result, and
fails if the generated token, text, score, margin, or normalized action
probability crosses its gate:

```bash
HIP_VISIBLE_DEVICES=0 \
python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/sglang_live_parity.json \
  --num-images 8 \
  --host 127.0.0.1 \
  --port 31000 \
  --base-gpu-id 0
```

Run it with the same SGLang Python and ROCm `sgl_kernel` paths as the Relax
launcher. It installs no packages and terminates its server before returning.

Every CPU feature bundle carries an immutable content-addressed `feature_id`
and `vision_revision`. They are retained in the SGLang payload and parity
artifacts. Megatron's batched multimodal dictionary remains tensor-only;
putting strings into that dictionary would break its concatenate-and-broadcast
path.

## Skip the GPU vision encoder

After the parity artifact passes, skip the GPU encoder explicitly:

```bash
export SKIP_GPU_VISION_ENCODER=1
```

The default remains `0`. When the GPU encoder is skipped:

- Megatron-Bridge receives `add_encoder=False` before provider finalization,
  so it never constructs the GPU visual tower. A parameterless guard is
  installed afterward and raw-image forwards fail with an actionable error.
- SGLang's Transformers Qwen3-VL model replaces its meta-device visual module
  with a parameterless guard before recursive GPU materialization.
  `model.visual.*` checkpoint tensors are intentionally skipped.
- Precomputed final and DeepStack streams continue through the same language
  model paths as CPU vision with the GPU encoder kept.

For the local 0.37B visual-XOR checkpoint, the visual subtree contains
27,084,160 parameters: 51.66 MiB in BF16 (103.32 MiB in FP32) per otherwise
complete GPU model instance. This is the theoretical parameter-storage
reduction, not a measured allocator delta. Measure per-device peak and steady
VRAM because framework buffers, allocator rounding, and the number of actor,
actor-forward, and rollout copies also matter.

An skipped-vision Megatron checkpoint is configuration-specific. Do not resume
it with the GPU encoder kept or expect a complete standalone Hugging Face VLM
export from it; the skipped visual weights are still sourced from the original
frozen checkpoint used by the CPU service.

## CPU scaling benchmark

Use the local multiprocessing benchmark to choose threads, explicit image
batch size, and replica count against a measured rollout demand:

```bash
python -m examples.visual_xor.measure_cpu_vision_scaling \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_train.parquet" \
  --output benchmark_results/cpu_vision/scaling.json \
  --peak-unique-images-per-second 20 \
  --replica-counts 1,2,4 \
  --thread-counts 1,2,4,8 \
  --batch-sizes 1,2,4,8 \
  --max-total-cpus 8 \
  --num-images 64 \
  --repeats 5
```

The benchmark creates one frozen backend per spawned process, warms each
worker before timing, keeps only one replica/thread process layout alive, and
reuses that layout across its consecutive batch-size trials. It records outer
wall time, summed backend-forward wall time, summed backend process CPU time,
per-process configured thread count and peak RSS, emitted feature bytes,
completed images, and throughput. `--max-total-cpus` defaults to eight and
removes every layout where `replicas * threads_per_replica` exceeds that
budget.

A configuration passes only if it sustains at least `1.25` times the supplied
peak unique-image demand. The recommendation chooses the smallest passing CPU
reservation, then breaks ties by higher throughput, fewer replicas, fewer
threads, and smaller explicit batch size. The artifact also reports scaling
efficiency relative to the `1 replica x 1 thread x batch size 1` baseline.
The demand value must come from the rollout workload; inventing a small value
only proves the harness, not capacity.

The batch-size dimension is currently a backend capability/what-if sweep.
Visual-XOR rollout sends one image per service request and the service does not
yet coalesce independent Ray Serve requests into a larger visual batch.
`VISION_ENCODER_MAX_IMAGES_PER_REQUEST` caps images already present in one request; it
does not create dynamic batching. Use the batch-size-one result for the
current end-to-end recipe. Treat larger-batch results as motivation for a
separate request-batching change, not as throughput the live path already
achieves. The offline artifact marks a layout batching-eligible only when its
best explicit batch size improves throughput by at least 20% over batch size
one, and selects the smallest batch size within 5% of that best throughput.

`--vision-encoder-batch-wait-timeout-ms` and
`VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS` currently accept only zero. Negative or
non-finite values fail validation, and positive values fail with an explicit
`NotImplementedError`; they are not a silent no-op. Implement the asynchronous
per-replica miss queue only after an CPU scaling artifact passes the 20% gate.

Every `VisionEncoder.encode` response carries the immutable feature bundle,
replica and feature identities, cache-hit state, backend-forward wall time,
actual backend batch size, and the serving replica's cumulative counter
snapshot. The rollout manager retains the newest snapshot observed from every
replica; it does not call load-balanced `get_metrics()` and mislabel one
arbitrary replica as deployment-wide state. It writes the aggregate after
every eval and rollout and forwards cumulative (`*_total`) and
since-last-snapshot (`*_interval`) values to W&B:

```text
vision_encoder/cache/entries
vision_encoder/cache/resident_bytes
vision_encoder/cache/hits_total, hits_interval
vision_encoder/cache/misses_total, misses_interval
vision_encoder/cache/evictions_total, evictions_interval
vision_encoder/cache/hit_rate_total, hit_rate_interval
vision_encoder/requests_total, requests_interval
vision_encoder/backend/encode_requests_total, encode_requests_interval
vision_encoder/backend/encoded_images_total, encoded_images_interval
vision_encoder/backend/emitted_feature_bytes_total, emitted_feature_bytes_interval
vision_encoder/backend/encode_seconds_total, encode_seconds_interval
vision_encoder/backend/images_per_second_interval
vision_encoder/backend/seconds_per_request_interval
vision_encoder/replicas/{expected,observed,active,idle,unobserved}
vision_encoder/replica/<id>/requests_total, requests_interval
vision_encoder/replica/<id>/request_fraction_total, request_fraction_interval
vision_encoder/replica/<id>/process_cpu_seconds_total
vision_encoder/replica/<id>/process_cpu_utilization_percent_interval
vision_encoder/replica/<id>/rss_bytes
vision_encoder/features/unique_total, unique_interval
vision_encoder/backend/duplicate_encode_ratio, duplicate_encode_ratio_interval
vision_encoder/backend/batch_size_{mean,p95,max}
vision_encoder/backend/batch_size_{mean,p95,max}_interval

perf_detail/rollout/vision_service_round_trip_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/precomputed_prepare_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/precomputed_to_list_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_request_build_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_response_wait_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_response_read_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_response_decode_time/{count,total,mean,p50,p95,max}
perf_detail/rollout/precomputed_feature_bytes/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_request_body_bytes/{count,total,mean,p50,p95,max}
perf_detail/rollout/http_response_body_bytes/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_generation_requests/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_parallel_samples/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_vision_cache_inline_publish_requests/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_vision_cache_id_only_requests/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_vision_cache_id_only_hits/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_vision_cache_id_only_misses/{count,total,mean,p50,p95,max}
perf_detail/rollout/sglang_vision_cache_republish/{count,total,mean,p50,p95,max}
```

Cache hits increment total requests but do not increment backend work or
backend time. `rss_bytes` is Linux process peak RSS from `ru_maxrss`, converted
from KiB to bytes. CPU utilization is derived per replica from changes in
process CPU seconds divided by changes in the replica's monotonic snapshot
time. Interval feature sets are cleared after collection, so a recurring
feature ID is counted again in the next rollout interval. Duplicate-encode
ratio uses backend-miss feature identities rather than cache-hit identities.

The client-side CPU-service round trip includes Ray/Serve scheduling and
feature-result transport, while the service's backend encode counter measures
only actual visual execution on cache misses. Shared feature preparation and
packed BF16 serialization are counted once per prompt group. HTTP request
construction measures the actual JSON body construction performed by `httpx`
and reports its exact byte length without serializing the payload a second
time. `http_response_wait` still combines server queueing, reconstruction/H2D,
prefill, and decode; those require separate server-side instrumentation if this
combined boundary dominates.

The historical metric name `precomputed_to_list_time` remains for W&B series
continuity. It now measures contiguous-BF16/base64 packing, not conversion to
nested Python lists. On a grouped SGLang cache ID hit, that serialization is skipped
entirely. The actor still receives the established tensor-only dictionary;
actor-side feature-table deduplication remains measurement-gated and is not
silently introduced by the rollout cache.

## CPU-ViT capacity experiment

Set `CPU_VISION_CAPACITY_MODE=1` for CPU vision demand and CPU worker layouts live runs. Before invoking the
base launcher, the import-light preflight requires at least 16 CPUs in the
process affinity mask, at least eight distinct `(physical_package_id,
core_id)` pairs, and no more than eight reserved ViT CPUs across all replicas.
This prevents an SMT-only allocation from being reported as a CPU scaling
result.

On the `faculty` Slurm cluster, submit the GPU-CPU parity-CPU vision demand sequence with:

```bash
sbatch scripts/slurm/qwen3_vl_measure_cpu_vision_demand.sbatch
```

The batch job requests node-exclusive access, four MI210 GPUs, 16 non-SMT CPU
cores, and 128 GiB for at most 12 hours. Node exclusivity is required because
the single-node launcher uses a fixed Ray dashboard port and performs broad
local Ray cleanup. It starts with the local full-vocabulary parity gate, runs
the live scalar/grouped parity probe, and then runs prompts8_samples8-prompts64_samples1 sequentially in the
same allocation. The grouped-GPU artifact remains non-blocking when the
scalar CPU and grouped CPU gates pass. As an additional guard, the wrapper
refuses to invoke the local Ray launcher if another active job owned by the
same user shares its node.
Artifacts and the prompts8_samples8-prompts64_samples1 log manifest are written under
`benchmark_results/cpu_vision/slurm_<job_id>_e0_e1/`; Slurm output is written
to `log/slurm-cpu-vision-demand-<job_id>.log`.

The four demand profiles keep exactly 64 generated responses per rollout:

| Profile | `ROLLOUT_BATCH_SIZE` | `N_SAMPLES_PER_PROMPT` | Claim |
|---|---:|---:|---|
| prompts8_samples8 | 8 | 8 | Existing grouped GRPO baseline |
| prompts16_samples4 | 16 | 4 | RL-valid GRPO |
| prompts32_samples2 | 32 | 2 | Highest unique-image RL-valid demand |
| prompts64_samples1 | 64 | 1 | System-performance probe only; no GRPO learning claim |

For each CPU vision demand run, also set:

```bash
CPU_VISION_CAPACITY_MODE=1
NUM_ROLLOUT=12
VISION_ENCODER_NUM_REPLICAS=1
VISION_ENCODER_NUM_CPUS=1
SKIP_GPU_VISION_ENCODER=1
VISION_ENCODER_CACHE_MAX_BYTES=1
VISION_ENCODER_BATCH_WAIT_TIMEOUT_MS=0
ENABLE_EVAL=0
```

Keep `GLOBAL_BATCH_SIZE=32`, `NUM_STEPS_PER_ROLLOUT=2`, `MAX_STALENESS=4`,
checkpoint saving disabled, and dataset shuffling disabled. Discard rollout
cycles 0-1. For each later cycle, pair
`vision_encoder/features/unique_interval` with the corresponding complete
training-cycle wall time. CPU vision demand peak consumer demand is the maximum paired ratio,
not the encoder's isolated throughput.

Run CPU worker layouts on prompts32_samples2 with `1x1`, `1x4`, `2x2`, and `4x1`, plus the distinct CPU scaling winner.
If prompts32_samples2 does not make `1x1` actor data wait reach 5%, use prompts64_samples1 only as a saturation
probe. Keep weight-update wall visible as a timing covariate; it is not a
CPU-ViT optimization objective.

The current checkout was prepared under affinity CPUs `8,72`. Both map to
physical package zero, core 16, so this allocation has two SMT threads of one
physical core. It correctly fails the capacity preflight.

Slurm job `141944` subsequently completed GPU-CPU parity and all four CPU vision demand profiles on a
CPU-rich node. Use the checked analyzer rather than hand-pairing log lines:

```bash
python -m examples.visual_xor.analyze_cpu_vision_demand \
  --profile prompts8_samples8=benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand_d0.console.log \
  --profile prompts16_samples4=benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand_d1.console.log \
  --profile prompts32_samples2=benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand_d2.console.log \
  --profile prompts64_samples1=benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand_d3.console.log \
  --output benchmark_results/cpu_vision/slurm_141944_cpu_vision_demand/cpu_vision_demand.json
```

The global CPU vision demand peak is `5.8588356399` unique images/s and the CPU scaling target is
`7.3235445499` images/s. prompts32_samples2 peaked at `3.2404551625` images/s and its maximum
steady actor-wait ratio was only `0.6549997%`, so prompts32_samples2 did not reach the 5%
saturation gate. prompts8_samples8-prompts32_samples2 have exact request/image accounting. prompts64_samples1 generated the
required 64 accepted responses per cycle but its last interval encoded 29
extra refill/filter candidates; the artifact flags this rather than treating
prompts64_samples1 as an RL-valid accounting pass. Corrected CPU scaling Slurm job `142801` completed
all 36 matrix points inside a 16-core Slurm step. The `1x1` batch-one result
already sustains `17.433295` images/s. Explicit batch eight gains only `7.955%`
at that smallest passing layout, so live dynamic batching is not eligible for
the selected configuration.

Corrected CPU worker layouts Slurm job `142804` completed the equal-four-CPU prompts32_samples2 comparison.
`1x4`, `2x2`, and `4x1` averaged `1.613`, `1.405`, and `1.454` seconds of
steady rollout time, respectively, while all mean actor-wait ratios remained
below `0.62%`. `2x2` had the lowest rollout time, but used approximately
`4.41 GiB` aggregate encoder RSS; `4x1` used `8.58 GiB` and did not improve on
`2x2`; `1x4` used only `2.27 GiB`. These results do not displace `1x1` as the
system recommendation because its offline batch-one capacity is already
`2.975` times the measured peak demand and the live actor is not starved.

The cache-reuse study isolates the 1 GiB CPU cache from a 1 GiB SGLang cache
using the selected one-replica, one-thread CPU setting. Attempts `142805` and `142809` found, respectively, a
tensor-valued grid that was not JSON serializable and an early SGLang media
loader that rejected the ID-only dictionary before cache resolution. Both
boundaries now have regression tests. The cache-reuse Slurm wrapper requires exact
vision, rollout, and training cycles 0 through 11 and runs the checked analyzer
before reporting completion. Job `142817` completed all required cycles. The SGLang cache reduced request-body
bytes by `99.824%`, serialization time by `77.939%`, mean per-cycle p95 service RTT by `29.303%`, and rollout time by
`6.172%`. The rollout reduction remains below the `10%` adoption gate and must
not be presented as a training-cycle speedup.

The matched paper experiment compares GPU vision, CPU vision without caches, CPU vision with the 1 GiB CPU cache,
and CPU vision with both 1 GiB caches. Its primary boundary is `perf/step_time`, defined as
`perf/train_wait_time + perf/train_time`. The train scope already includes
Megatron training, two optimizer intervals, service checks, weight update, and
barriers; adding optimizer or weight-update sub-timers to `perf/step_time`
would double-count them. Four order-balanced repeats use run means as
the experimental units. Steady rollout cycles are repeated observations
inside a run, not independent replicates.

Job `154079` completed the four-repeat study. CPU vision without caches averaged
`11.056925 s` per training cycle, while CPU vision with both caches enabled lazily averaged
`11.212820 s`. Enabling both caches was `1.504%` slower, with a 95%
confidence interval from `-10.170%` to `7.161%`; caching therefore has no
demonstrated training-cycle benefit in this workload. Moving frozen vision computation to CPU remains the positive
matched result: CPU vision without caches was `7.328%` faster than GPU vision, with a 95% confidence interval from
`3.711%` to `10.944%`.

The remaining cache question is cold-start amortization. The opt-in
`--preload-vision-features` path snapshots the finite eager dataset,
encodes every image into the CPU cache, and admits every immutable feature to
the process-local SGLang cache before cycle 0. Admission uses the internal
`POST /relax/vision-features/cache` endpoint and never enters generation.
The controller releases service execution only after both caches retain the
entire feature set with zero eviction. The direct runner compares no-cache execution with a completely preloaded
64-image working set and reports both cycle-0-inclusive training-cycle time and time including the one-time preload.

Slurm job `160198` completed four matched repeats. The paired raw training-cycle speedup was `0.324%`, with a 95%
confidence interval from `-4.100%` to `4.748%`.
After amortizing the mean `8.660052 s` preload cost over 22 cycles, the paired
speedup was `-2.889%`, with a 95% confidence interval from `-7.340%` to
`1.562%`. Perfect residency reduced rollout time by `61.806%`, mean service
RTT by `97.724%`, request-body bytes by `99.824%`, and serialization time by
`71.158%`, but it did not move the complete training critical path. Treat
preloading as an opt-in correctness and upper-bound experiment, not a demonstrated
training-throughput optimization for `prompts32_samples2`. The checked artifact is
`benchmark_results/cpu_vision/full_dataset_preload_retry2_20260827_081348/full_dataset_preload_analysis.json`.

## Execution plan

Development proceeds through explicit gates so performance work cannot hide a
representation mismatch:

1. Add immutable feature and visual-revision identities, propagate them to
   SGLang and Megatron, and fail on mismatches.
2. Compare CPU and GPU final/DeepStack features on fixed visual-XOR
   images, then compare GPU and precomputed logits inside each backend.
3. Compare synchronized SGLang and Megatron response logits while holding the
   feature identity, policy version, prompt, and tokens fixed.
4. Run a two-cycle Relax smoke that proves one CPU encode feeds rollout,
   actor-forward, and actor training.
5. Only after those checks pass, enable the option to skip the GPU vision
   encoder and measure the actual VRAM reduction.
6. Compare GPU vision, CPU vision with the GPU encoder kept, and CPU
   vision with the GPU encoder skipped before tuning CPU threads, explicit image
   batches, or independent Ray Serve replicas.

For that GPU/CPU comparison, freeze the GPU visual tower and
projector too. The GPU debug launcher sets `FREEZE_VISION_MODEL=1`, which
keeps vision computation on GPU while avoiding trainable visual gradients and
optimizer state. Run `examples.visual_xor.monitor_rocm_vram` around each mode
sequentially from an idle baseline and compare the per-device peak deltas, not
only process-reported allocator values.

The completed 2026-07-31 comparison is recorded in
`training_reports/2026-07-31-qwen3-vl-gpu-cpu-vision-vram.md`. The isolated
GPU-encoder-kept to GPU-encoder-skipped reduction was 155.73 MiB at the simultaneous four-card
peak. The CPU evaluation cache achieved 640 hits from 768 requests with 128
actual encodes and no evictions. However, the CPU modes were approximately
1.53 times the GPU wall time and produced an action-A-biased, chance-level
held-out result while GPU vision was balanced and scored 0.6816. Treat
the live GPU-versus-CPU-precomputed SGLang and Megatron logit comparison as a
blocking P0 gate before tuning CPU capacity or making an optimization claim.

## Follow-up risk register

The baseline above is intentionally smaller than the complete production
design. The following work is part of the plan even when it is not required
for the first two-cycle smoke. Items are ordered by when they become blocking,
not by implementation difficulty.

### P0: correctness gates before a performance claim

1. **DeepStack semantic parity.** Compare the final projected stream and every
   selected, projected DeepStack stream independently. The parity artifact
   must name each stream, its shape, dtype, feature identity, cosine error,
   absolute error, and elementwise close fraction. Do not treat successful
   final-stream parity as proof that the three intermediate streams reached
   the correct language layers.
2. **One immutable bundle for every consumer.** Prove that one
   content-addressed bundle can feed SGLang rollout, Megatron actor-forward,
   and Megatron actor training without re-encoding or changing the tensor
   values. Record the same `feature_id` and `vision_revision` at each boundary.
3. **Frozen does not mean permanently valid.** Vision features have no policy
   staleness while the complete visual module is frozen, but they become
   invalid when the visual checkpoint, projection/merger, processor, grid
   layout, output dtype, or feature schema changes. Preserve those inputs in
   the cache identity and add an explicit schema version before features are
   persisted outside one process lifetime.
4. **Skipping the GPU encoder remains a separate gate.** First prove parity with the GPU encoder kept,
   then omit the SGLang and Megatron visual weights, repeat parity, and measure
   actual per-device allocator usage. Parameter counts alone are not evidence
   that temporary buffers and duplicate model copies were freed.
5. **Concurrency correctness.** Feature tensors are immutable after
   publication. Any future in-flight request coalescing or reusable-buffer
   implementation must prove that concurrent requests cannot overwrite,
   mutate, or evict storage still used by SGLang or Megatron.

### P1: routing, transport, and capacity

1. **Replica routing and duplicate caches.** Ray Serve replicas currently own
   independent LRUs and requests are not sticky by `feature_id`. Aggregate
   response snapshots now calculate:

   ```text
   duplicate_encode_ratio =
       aggregate backend encode requests / unique feature IDs requested
   ```

   Restore the 1 GiB cache for the winning CPU worker layouts layout and repeat prompts32_samples2. If this
   ratio exceeds `1.25`, evaluate feature-ID-sticky routing next. Do not move
   directly to a shared object store or database; consider shared storage only
   if routing cannot provide sufficient locality or sharing is also required
   across actor and rollout processes.
2. **Concurrent-miss coalescing.** Two simultaneous misses for the same feature
   can both run the CPU backend before either inserts into its replica's LRU.
   Measure this cache-stampede case separately from cross-replica duplication.
   If it occurs under the real rollout concurrency, add a per-feature in-flight
   future so one encode serves all waiters.
3. **Grouped SGLang parallel sampling before a registry.** For eligible fresh
   CPU-precomputed Qwen3-VL groups, send one JSON feature with
   `sampling_params.n=N_SAMPLES_PER_PROMPT` and map the ordered response list
   back to Relax samples. Keep deterministic, partial, resumed, custom,
   Slime, routing-replay, and multi-engine non-round-robin cases on scalar
   requests with an explicit logged reason. A non-round-robin policy is safe
   when exactly one engine exists because there is no cross-engine routing
   decision. Measure request count and bytes again before adding a new storage
   layer.
4. **Stateless packed transport before stateful storage.** Encode each
   contiguous BF16 bundle as base64 in the existing JSON envelope, preserve its
   dtype, shape, feature ID, vision revision, and grid, and reconstruct it
   before SGLang's base processor reads `feature`. Require bit-exact BF16 wire
   round trips and live GPU-versus-CPU-precomputed parity. Keep legacy nested
   payloads readable during migration, and do not inspect GPU media objects
   as dictionaries.
5. **Bounded engine-local feature cache, now opt-in.** The implemented SGLang cache
   cache preserves the packed-inline first-use path, then uses ID-only lookup
   for the same schema/revision/feature identity. It is disabled by default,
   byte bounded, sticky-routed, and recovers one marked miss by republishing
   inline. Measure packing, HTTP bytes, reconstruction, H2D, temporary RSS,
   hit/miss/republish counts, and rollout time before adopting it. Do not call
   it a shared database or cross-job registry.
6. **Request batching is not configured batching.**
   `VISION_ENCODER_MAX_IMAGES_PER_REQUEST` only rejects oversized requests; it does not
   coalesce independent requests. Measure queue delay and batch-size-one
   capacity first. Add bounded dynamic batching only if an explicit batch size
   above one improves CPU scaling throughput by at least 20% at the same replica/thread
   layout. The live path must then pass its separate rollout-time, CPU
   reservation, parity, and p95 service-latency adoption gates.
7. **Backpressure and in-flight memory.** Cache capacity does not include JSON
   copies, Ray object copies, pinned staging buffers, GPU copies, or queued
   requests. Record peak process RSS and in-flight feature bytes, then bound
   admission or queue depth if the rollout CPU cache can outrun the encoder or
   either consumer.
8. **CPU allocation and oversubscription.** Ray reserves
   `VISION_ENCODER_NUM_CPUS` per replica and the backend uses that value for
   PyTorch CPU threads. Benchmark total node throughput while actor, rollout,
   Ray Serve, and transfer services are active; isolated encoder throughput is
   not enough. Add replicas or threads only while end-to-end throughput scales
   and other services retain their latency budget.
9. **Overlap proof.** Produce a common timeline containing queue, encode,
   serialization, transfer, H2D, SGLang prefill, actor-forward, and actor
   training intervals. A fast isolated encoder is useful only if the CPU work
   overlaps GPU work and reduces or preserves end-to-end step time.
10. **Cold start and recovery.** Measure model load, first encode, warm encode,
   cache rebuild after replica restart, and retry behavior. A content-addressed
   request may be retried safely, but the capacity plan must include the cold
   cache period.

### P2: graph-safe and allocation-safe execution

1. **Replace the temporary Megatron method substitution.** The current adapter
   temporarily replaces `vision_model.forward` and supplies a dummy pixel
   tensor. This is acceptable for the eager correctness baseline, but it is
   not the target interface for `torch.compile`, CUDA/HIP graph capture, or
   concurrent calls on one model instance. Before enabling any of those paths,
   add an explicit Qwen3-VL forward contract for `vision_embeds`,
   `deepstack_visual_embeds`, and `image_grid_thw`, then remove the per-call
   method replacement.
2. **Validate SGLang graph behavior explicitly.** The current debug recipe uses
   `--sglang-disable-cuda-graph`, so it does not prove graph compatibility.
   Before enabling graphs, distinguish multimodal prefill from decode,
   enumerate supported visual-shape buckets, and record graph capture,
   fallback, and recapture counts. Do not silently fall back to eager
   execution for an unrecognized image grid.
3. **Shape buckets before buffer pools.** The current visual-XOR task has one
   stable grid, but general images vary by grid, image count, token count,
   hidden size, DeepStack count, and dtype. Define an explicit shape signature
   and either reject unsupported shapes or route them to documented buckets
   before assuming static graphs or fixed buffers.
4. **No single reusable feature buffer.** Fully asynchronous execution has
   several requests in flight. If profiling shows allocation or H2D cost is
   meaningful, use a leased ring/pool of pinned host and optional fixed-address
   device buffers keyed by the shape signature. Release a slot only after its
   consumer event/future completes.
5. **Evaluate contiguous packing without changing semantics.** A complete
   Qwen3-VL bundle may be transported as one contiguous `[N, 4 * H_text]`
   BF16 tensor while Megatron consumes zero-copy views for the final and three
   DeepStack streams. Adopt this only after parity and lifetime tests prove
   that packing does not reorder streams or introduce copies at the consumers.

Separate buffers do not by themselves prevent graph breaks. Dynamic Python
control flow, method replacement, changing tensor shapes, and changing storage
addresses are independent concerns and must be measured independently.

### P3: scale-dependent extensions

1. **Shared or persistent feature store.** The implemented byte-bounded
   in-process LRU is sufficient for the current 64-image training working set.
   Design an ID-addressed shared store only when cross-replica duplication,
   cross-job reuse, restart recovery, or a working set larger than RAM creates
   a measured need. Such a store must define schema/revision identity, atomic
   publication, tensor dtype/layout, eviction, ownership, and cleanup; a
   generic relational database is not assumed to be the right tensor store.
2. **Pipeline/context parallel DeepStack routing.** The baseline deliberately
   requires PP=1 and CP=1. Supporting PP or CP requires an explicit owner for
   each DeepStack injection layer plus tested scatter/split behavior. It must
   not be inferred from the PP=1 adapter.
3. **Alternative CPU runtimes.** Evaluate Prima.cpp, llama.cpp, or another
   parallel CPU runtime only after the PyTorch implementation supplies the
   correctness and throughput baseline. Runtime dispatch must preserve the
   exact final and DeepStack contract rather than merely producing embeddings
   of the same shape.

The observability work shared by all priorities is to aggregate unique feature
IDs, duplicate encodes, cache hits/misses/evictions, in-flight coalescing,
queue latency, backend encode time, serialization/transfer/H2D time, process
RSS, per-device VRAM, graph fallback/recapture counts, and consumer wait time.
Without these measurements, routing, storage, batching, and buffer changes
remain hypotheses rather than planned optimizations.

Benchmark artifacts belong in `benchmark_results/cpu_vision/<timestamp>/`.
They must report feature error, response-logit error, cache behavior, payload
bytes, queue/encode/transfer/H2D time, per-device VRAM, and end-to-end
throughput. The CPU configuration is acceptable only when it sustains at least
1.25 times the rollout pipeline's peak unique-image demand.

`RESOURCE_JSON["vision_encoder"]` remains `[1,0]`: Relax currently requires one
logical service entry. Future request-level data parallelism belongs in a
separate Ray Serve replica setting; it is not DDP and must not be expressed by
changing the first resource value.

Prima.cpp, llama.cpp, a persistent database, and PP-aware DeepStack routing are
deferred. They are considered only after the PyTorch CPU baseline is correct
and measured. The implemented job-local SGLang cache is deliberately smaller
than those proposals: it is process local, disabled by default, and retains an
explicit inline publication/republication path. A shared or persistent store
still requires measured cross-process or cross-job reuse.

## Current performance boundary

CPU vision with the GPU encoder kept bypasses GPU visual computation while retaining the
visual modules on the GPU. Skipping the GPU encoder avoids constructing or materializing
those modules. It remains opt-in and should be enabled only after both local
HF and live SGLang parity pass on the target node.

With SGLang cache disabled, the SGLang router transports contiguous BF16 bytes as
base64 inside the existing JSON envelope. This avoids nested numeric JSON but
retransmits one feature per distinct HTTP request and retains base64's 4/3
expansion. With SGLang cache enabled, the first grouped request remains inline and
later sticky-routed requests are ID-only; a marked miss republishes inline
once. Measure request serialization, reconstruction/H2D, SGLang prefill time,
actor time, cache hit rate, and end-to-end throughput before calling the design
a general performance win.

The measured scaling result narrows that performance claim. One CPU thread
already supplies nearly three times peak observed feature demand, and every
CPU vision demand/CPU worker layouts actor-wait mean is below 1%. More ViT CPUs can reduce request latency,
but they do not remove the present end-to-end bottleneck; multiple replicas
also multiply model and cache RSS. The main research value is therefore
frozen compute and transport reuse with explicit invalidation, plus
a demand-before-supply method that identifies when extra CPU capacity should
not be reserved. SGLang cache must still demonstrate request-byte reduction and pass
the rollout-time and p95-latency adoption gates before it is enabled by
default.

The local CPU smoke on 2026-07-28 successfully loaded the real refinement
checkpoint, encoded one image, and wrote a scaling artifact. With one replica,
two threads, batch size one, and a warmed worker, the single measured encode
took approximately 0.057 seconds and the complete measured call took
approximately 0.060 seconds. Its demand input was deliberately only
0.01 images/second, so its reported capacity pass is a harness smoke, not a
performance recommendation.

The eight-image local Hugging Face parity gate passed on one MI210 on
2026-07-30. All four feature streams had cosine similarity above `0.999993`,
maximum absolute error `0.03125`, and at least `99.9979%` elementwise
agreement under `rtol=0.01`, `atol=0.01`. Mean full-vocabulary response KL was
approximately `1.65e-8`, every top token matched, and the maximum A/B
probability delta was below `8e-7`. The artifact is
`benchmark_results/cpu_vision/20260730_hf_parity/parity.json`.
The interpretation and reusable parity methodology are recorded in
`training_reports/2026-07-30-qwen3-vl-cpu-gpu-vision-parity.md`.

The live Ray/SGLang/Megatron two-cycle smoke with GPU vision passed on four MI210s
on 2026-07-30. Both 64-sample rollouts completed, actor-forward consumed the
final and all three numbered DeepStack streams, the DP2 actor completed four
successful optimizer updates, final weights reached actor-forward and
SGLang, and the job exited successfully without writing a checkpoint. A
later audit found that this historical SGLang build received but ignored the
precomputed embedding tensor, so this run is transport/Megatron evidence, not
SGLang semantic evidence. The complete timeline and correction are recorded
in
`training_reports/2026-07-30-qwen3-vl-cpu-vision-two-cycle-smoke.md`.

On 2026-07-31, the corrected SGLang adapter passed the eight-image live gate:
all generated tokens matched, maximum A/B log-probability delta was
`1.430511474609375e-06`, and maximum action-margin delta was
`1.043081283569336e-07`. A one-rollout skipped-weight Relax run then scored
`0.69921875` held out, completed actor-forward and two optimizer steps, and
kept Megatron/SGLang sampled-token log-probability differences below `5e-7`.
Its baseline cache served 768 requests with 128 encodes and 640 hits. The
evidence and scope limits are recorded in
`training_reports/2026-07-31-qwen3-vl-live-precomputed-deepstack-parity.md`.

This is now a correctness result for single-image PP1/CP1. The corrected
GPU/CPU repeat on 2026-08-01 measured JSON feature
transport directly: a 524,312-byte raw feature expanded to a 2.907 MB request,
the 768-request evaluation sent 2.233 GB, and a 64-sample rollout sent about
185 MB for eight unique features. The CPU backend needed only 49-54 seconds to
encode the 128 unique evaluation images.

The 2026-08-03 live SGLang gate then proved the smaller first transport step:
one precomputed `n=8` request was 3,179,557 bytes versus 25,436,392 bytes for
eight scalar-equivalent requests, an 87.50% reduction. All eight greedy
outputs matched the scalar request and maximum A/B log-probability delta was
`2.98e-7`. Relax therefore groups only fresh homogeneous CPU-precomputed
Qwen3-VL samples. Deterministic inference remains scalar because GPU
SGLang parallel sampling does not assign distinct branch seeds.

The matched two-rollout run passed on 2026-08-03. Each 64-sample rollout used
eight SGLang requests, reducing mean request traffic from 185.40 MB to 23.25
MB and mean rollout time from 21.51 seconds to 5.82 seconds. This is an 87.46%
traffic reduction and 3.70x rollout speedup. Per-device VRAM peaks were
unchanged, as expected for a transport optimization.

On 2026-08-04, evaluation grouping was decoupled from `group_rm`. The 128
held-out prompts now each use one `n=4` request, while deterministic controls
remain scalar, cutting nested-list evaluation from 768 requests and 2.233 GB
to 384 requests and 1.117 GB. Stateless packed BF16 then reduced the same
evaluation to 268.9 MB. The live GPU-versus-packed gate matched every token
with maximum A/B log-probability delta `1.43e-6`; the matched two-rollout run
sent 5.60 MB per 64 samples and completed all four optimizer updates. Defer a
general registry until the remaining cross-request traffic is measured as the
next bottleneck. CPU replica scaling still comes after that boundary. See
`training_reports/2026-08-04-qwen3-vl-packed-cpu-vision-transport.md` and
`training_reports/2026-08-03-qwen3-vl-sglang-parallel-sampling.md`.

On 2026-08-06, Relax extended the narrow grouped request contract to GPU
processor-backed image-only groups and ran matched 20-cycle GPU and
CPU with GPU encoder skipped jobs with evaluation disabled. The CPU cache was limited to one
byte, producing 160 misses, 160 evictions, and zero hits. Even under that
worst-case cache condition, the CPU backend encoded each eight-image rollout
batch in 1.904 seconds on average, every rollout interval overlapped actor
work, and steady actor data wait averaged only 0.238 seconds. CPU rollout time
was 4.789 seconds versus 3.263 seconds GPU, but removing frozen visual work
reduced actor compute by 13.68%, complete training-cycle time by 9.67%, and the
simultaneous four-card peak by 738.03 MiB. Packed feature transport remained
383.5 times GPU request bytes without pacing the actor.

The GPU `n=8` probe matched all 64 generated tokens and texts but failed
the strict branch-score gate: maximum action-log-probability, margin, and
probability deltas were `0.03938`, `0.06250`, and `0.01457`. Preserve that
failure. The GPU run is a performance counterfactual, not strict scalar
logit-parity evidence. See
`training_reports/2026-08-06-qwen3-vl-cpu-vision-steady-state-overlap.md`.

CPU capacity under a workload that actually starves the actor, replica-level
cache routing, video, multi-image requests, context/pipeline parallelism, and
chunked multimodal prefill remain open. A shared or persistent feature store
remains deferred until cross-process or cross-job reuse is measured to justify
it.

## Plan-based CPU/GPU device choice

The 2026-08-27 implementation adds an explicit device choice layer above the
existing GPU and CPU-precomputed execution paths. The default remains
`fixed`, so existing launchers retain their previous behavior. The
`automatic` policy is opt-in and requires a versioned vision-device plan with
an exact workload record for every rollout cycle.

For workload $x=(1,N_{image},N_{visual},B_{feature})$, the policy predicts
GPU time and total CPU-path time with independently fitted non-negative
linear models. It removes only overlap measured for that workload:

$$
T_{CPU,after\ overlap} = \max(0, T_{CPU,total} - T_{overlap})
$$

The plan selects CPU when its predicted relative improvement exceeds the
configured minimum gap, selects GPU when the opposite gap exceeds that value, and otherwise preserves the
preceding device. Missing cycles, invalid coefficients, and unsupported combinations fail before work is generated.
There is no default workload or silent device fallback.

Automatic mode deliberately retains the visual weights on the GPU because a
later cycle may select GPU execution. Therefore automatic device choice is a
throughput result, not a GPU-memory result. CPU-only execution with
`SKIP_GPU_VISION_ENCODER=1` remains a separate fixed-device memory
profile.

The first controlled study alternates 112-by-112 and 448-by-448 visual-XOR
workloads and compares fixed GPU, fixed CPU, and automatic choice in
five counterbalanced matched repeats. The acceptance gates are at least 90%
match rate among cycles whose measured CPU/GPU gap exceeds the minimum gap, at least half of cycles outside
that band, matched generation-request fanout, and less than 5% mean extra time
relative to the faster device for each measured cycle.

This is not yet an online policy: it selects from an exact predeclared cycle
map before fetching rollout data. It therefore rejects staleness, shuffle,
dynamic filtering, unequal oversampling, partial rollout, and full-dataset preloading.
The Qwen3-VL-4B Geo3K study currently compares fixed GPU, CPU with GPU encoder kept, and
CPU with GPU encoder skipped profiles. Automatic Geo3K remains disabled until actual fetched-batch
descriptors drive the decision.

See `docs/draft/choose_vision_device_per_cycle.md`,
`training_reports/2026-08-27-iclr-claim-evidence.md`, and
`training_reports/2026-08-27-qwen3-vl-choose-vision-device-per-cycle.md`.
