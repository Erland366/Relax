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
  `--vision-encoder-backend pytorch` is set.

## Data flow

```text
raw image
  -> Relax Hugging Face processor
  -> CPU vision_encoder Ray Serve service
       -> final projected visual embeddings
       -> three ordered DeepStack embedding streams
       -> byte-bounded, revisioned LRU cache
  -> SGLang precomputed_embedding request
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
lets the existing Qwen3-VL forward perform text embedding replacement, MRoPE,
THD packing, and DeepStack injection. Raw pixel tensors are removed from the
actor payload.

SGLang receives the same final and DeepStack streams concatenated in its
Qwen3-VL precomputed-embedding layout. Relax installs a gated compatibility
patch in SGLang processes so the embeddings are not mistaken for raw visual
features and `image_grid_thw` remains available for MRoPE.

## Resource and cache configuration

`RESOURCE_JSON` keeps its existing `[num_services, num_gpus]` meaning. A CPU
encoder therefore uses:

```bash
export RESOURCE_JSON='{"actor":[1,2],"rollout":[1,1],"actor_fwd":[1,1],"advantages":[1,0],"vision_encoder":[1,0]}'
```

CPU reservation and cache sizing are separate:

```bash
export VISION_ENCODER_BACKEND=pytorch
export VISION_ENCODER_NUM_CPUS=8
export VISION_ENCODER_NUM_REPLICAS=1
export VISION_ENCODER_CACHE_MAX_BYTES=1073741824
export VISION_ENCODER_MAX_BATCH_SIZE=8
export VISION_ENCODER_OMIT_GPU_WEIGHTS=0
```

The cache key includes the processed pixels, grid, output dtype, visual
configuration, and a hash of every visual tensor. Reusing a feature from a
different checkpoint revision is therefore not allowed.

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
written by that recipe. W&B and the ordinary run log remain enabled.

## Correctness and parity gates

Run the fixed-image Hugging Face parity gate before enabling GPU-weight
omission:

```bash
python -m examples.visual_xor.validate_cpu_vision_parity \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/parity.json \
  --device cuda:0 \
  --num-images 8
```

The diagnostic compares the CPU encoder against the native GPU visual tower
for the final projected stream and all three DeepStack streams. It then
compares the native raw-image Hugging Face forward with a language-only
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

Every CPU feature bundle carries an immutable content-addressed `feature_id`
and `vision_revision`. They are retained in the SGLang payload and parity
artifacts. Megatron's batched multimodal dictionary remains tensor-only;
putting strings into that dictionary would break its concatenate-and-broadcast
path.

## GPU-weight omission

After the parity artifact passes, omission is enabled explicitly:

```bash
export VISION_ENCODER_OMIT_GPU_WEIGHTS=1
```

The default remains `0`. In omission mode:

- Megatron-Bridge receives `add_encoder=False` before provider finalization,
  so it never constructs the GPU visual tower. A parameterless sentinel is
  installed afterward and raw-image forwards fail with an actionable error.
- SGLang's Transformers Qwen3-VL model replaces its meta-device visual module
  with a parameterless sentinel before recursive GPU materialization.
  `model.visual.*` checkpoint tensors are intentionally skipped.
- Precomputed final and DeepStack streams continue through the same language
  model paths as resident mode.

For the local 0.37B visual-XOR checkpoint, the visual subtree contains
27,084,160 parameters: 51.66 MiB in BF16 (103.32 MiB in FP32) per otherwise
complete GPU model instance. This is the theoretical parameter-storage
reduction, not a measured allocator delta. Measure per-device peak and steady
VRAM because framework buffers, allocator rounding, and the number of actor,
actor-forward, and rollout copies also matter.

An omitted-vision Megatron checkpoint is configuration-specific. Do not resume
it in resident-vision mode or expect a complete standalone Hugging Face VLM
export from it; the omitted visual weights are still sourced from the original
frozen checkpoint used by the CPU service.

## CPU scaling benchmark

Use the local multiprocessing benchmark to choose threads, explicit image
batch size, and replica count against a measured rollout demand:

```bash
python -m examples.visual_xor.benchmark_cpu_vision_scaling \
  --checkpoint "$HF_CHECKPOINT" \
  --dataset "$REFINEMENT_DATA/refinement_rl_train.parquet" \
  --output benchmark_results/cpu_vision/scaling.json \
  --peak-unique-images-per-second 20 \
  --replica-counts 1,2 \
  --thread-counts 2,4,8 \
  --batch-sizes 1,4,8 \
  --num-images 64 \
  --repeats 3
```

The benchmark creates one frozen backend per spawned process, warms each
worker before timing, reuses pools across batch-size trials, and records wall
time, summed backend encode time, emitted feature bytes, completed images, and
throughput. A configuration passes only if it sustains at least `1.25` times
the supplied peak unique-image demand. The demand value must come from the
rollout workload; inventing a small value only proves the harness, not
capacity.

The batch-size dimension is currently a backend capability/what-if sweep.
Visual-XOR rollout sends one image per service request and the service does not
yet coalesce independent Ray Serve requests into a larger visual batch.
`VISION_ENCODER_MAX_BATCH_SIZE` caps images already present in one request; it
does not create dynamic batching. Use the batch-size-one result for the
current end-to-end recipe. Treat larger-batch results as motivation for a
separate request-batching change, not as throughput the live path already
achieves.

Each live service replica exposes raw counters. The rollout path snapshots
them after every eval and rollout, writes them to the normal log, and forwards
them to W&B with cumulative (`*_total`) and since-last-snapshot
(`*_interval`) values:

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
```

Cache hits increment total requests but do not increment backend work or
backend time. These counters are per replica, not a deployment-wide aggregate.
The current correctness launcher uses one replica, so its snapshots are also
deployment totals. Aggregate replica snapshots explicitly before interpreting
a multi-replica experiment.

## Execution plan

Development proceeds through explicit gates so performance work cannot hide a
representation mismatch:

1. Add immutable feature and visual-revision identities, propagate them to
   SGLang and Megatron, and fail on mismatches.
2. Compare CPU and native-GPU final/DeepStack features on fixed visual-XOR
   images, then compare native and precomputed logits inside each backend.
3. Compare synchronized SGLang and Megatron response logits while holding the
   feature identity, policy version, prompt, and tokens fixed.
4. Run a two-cycle Relax smoke that proves one CPU encode feeds rollout,
   actor-forward, and actor training.
5. Only after those checks pass, enable the implemented opt-in omission of
   frozen visual weights and measure the actual VRAM reduction.
6. Compare native GPU vision, CPU vision with resident GPU weights, and CPU
   vision with omitted GPU weights before tuning CPU threads, explicit image
   batches, or independent Ray Serve replicas.

For that three-mode comparison, freeze the native GPU visual tower and
projector too. The native debug launcher sets `FREEZE_VISION_MODEL=1`, which
keeps vision computation on GPU while avoiding trainable visual gradients and
optimizer state. Run `examples.visual_xor.monitor_rocm_vram` around each mode
sequentially from an idle baseline and compare the per-device peak deltas, not
only process-reported allocator values.

The completed 2026-07-31 comparison is recorded in
`training_reports/2026-07-31-qwen3-vl-vision-three-mode-vram.md`. The isolated
resident-to-omitted reduction was 155.73 MiB at the simultaneous four-card
peak. The CPU evaluation cache achieved 640 hits from 768 requests with 128
actual encodes and no evictions. However, the CPU modes were approximately
1.53 times the native wall time and produced an action-A-biased, chance-level
held-out result while native GPU vision was balanced and scored 0.6816. Treat
the live native-versus-precomputed SGLang and Megatron logit comparison as a
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
4. **GPU omission remains a separate gate.** First prove resident-mode parity,
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
   per-replica metrics to calculate:

   ```text
   duplicate_encode_ratio =
       aggregate backend encode requests / unique feature IDs requested
   ```

   First evaluate feature-ID-sticky routing. Move to a shared object store only
   if routing cannot provide sufficient locality or sharing is also required
   across actor and rollout processes. Treat more than twice the unique
   feature work, or duplicated work that prevents the 1.25-times capacity
   target, as a reason to change the design.
2. **Concurrent-miss coalescing.** Two simultaneous misses for the same feature
   can both run the CPU backend before either inserts into its replica's LRU.
   Measure this cache-stampede case separately from cross-replica duplication.
   If it occurs under the real rollout concurrency, add a per-feature in-flight
   future so one encode serves all waiters.
3. **Binary feature transport.** The current SGLang request converts roughly
   512 KiB of BF16 tensors per image into Python lists and JSON. Measure
   packing, serialization, transfer, deserialization, and temporary RSS
   separately. Replace JSON with a binary, shared-memory, or Ray object
   reference path when serialization and transfer exceed 10% of rollout wall
   time or duplicate payload bytes exceed twice the unique feature bytes.
4. **Request batching is not configured batching.**
   `VISION_ENCODER_MAX_BATCH_SIZE` only rejects oversized requests; it does not
   coalesce independent requests. Measure queue delay and batch-size-one
   capacity first. Add bounded dynamic batching only if the larger-batch
   benchmark materially improves throughput without violating rollout latency.
5. **Backpressure and in-flight memory.** Cache capacity does not include JSON
   copies, Ray object copies, pinned staging buffers, GPU copies, or queued
   requests. Record peak process RSS and in-flight feature bytes, then bound
   admission or queue depth if the rollout producer can outrun the encoder or
   either consumer.
6. **CPU allocation and oversubscription.** Ray reserves
   `VISION_ENCODER_NUM_CPUS` per replica and the backend uses that value for
   PyTorch CPU threads. Benchmark total node throughput while actor, rollout,
   Ray Serve, and transfer services are active; isolated encoder throughput is
   not enough. Add replicas or threads only while end-to-end throughput scales
   and other services retain their latency budget.
7. **Overlap proof.** Produce a common timeline containing queue, encode,
   serialization, transfer, H2D, SGLang prefill, actor-forward, and actor
   training intervals. A fast isolated encoder is useful only if the CPU work
   overlaps GPU work and reduces or preserves end-to-end step time.
8. **Cold start and recovery.** Measure model load, first encode, warm encode,
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
   correctness and throughput baseline. Runtime replacement must preserve the
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
and measured. Inline feature transport is replaced by an ID-addressed bounded
in-memory store only if serialization/transfer exceeds 10% of rollout wall time
or duplicate payloads exceed twice the unique feature bytes.

## Current performance boundary

Resident CPU-vision mode bypasses GPU visual computation while keeping the
visual modules resident. Omission mode can now avoid constructing or
materializing those modules, but it remains opt-in until the parity gate
passes on the target node.

The SGLang router currently transports precomputed tensors as JSON lists.
That is larger than the original PNG and can make this experimental path
slower even when CPU encoding itself is fast. Measure request serialization,
SGLang prefill time, actor time, cache hit rate, and end-to-end throughput
before calling the design a performance win.

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

The GPU-resident live Ray/SGLang/Megatron two-cycle smoke passed on four MI210s
on 2026-07-30. Both 64-sample rollouts completed, actor-forward consumed the
final and all three numbered DeepStack streams, the DP2 actor completed four
successful optimizer updates, final weights reached actor-forward and
SGLang, and the job exited successfully without writing a checkpoint. The
complete timeline, metrics, and two integration bugs found during bring-up are
recorded in
`training_reports/2026-07-30-qwen3-vl-cpu-vision-two-cycle-smoke.md`.

This is a correctness result, not a performance result. Warm request latency
showed qualitative cache reuse, but the run did not collect
`VisionEncoder.get_metrics()` before shutdown. Exact cache behavior, overlap,
GPU VRAM comparison, fixed-input SGLang/Megatron logit parity, and the
GPU-weight-omission repeat remain open gates.
