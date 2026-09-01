# Qwen3-VL Corrected GPU and CPU Performance Boundary

- **Date:** 2026-08-01
- **Status:** matched GPU, CPU-with-GPU-encoder-kept, and CPU-with-GPU-encoder-skipped runs passed;
  transport is the next optimization boundary
- **Hardware:** four AMD MI210 GPUs; one Ray CPU reserved for one CPU vision
  replica
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **Workload:** baseline evaluation, two fully asynchronous rollouts, four
  optimizer updates, final weight synchronization, and no checkpoint saving

## Objective

Repeat the GPU, CPU-precomputed with GPU vision-encoder weights kept, and
CPU-precomputed with GPU vision-encoder weights skipped comparison after correcting
SGLang's consumption of final plus DeepStack features. Add client-side stage
and payload instrumentation so the result can distinguish CPU encoding from
Ray/Serve scheduling, tensor conversion, JSON construction, HTTP transport,
and SGLang execution.

This supersedes the performance interpretation of the 2026-07-31 GPU/CPU
run. That historical run remains useful as bring-up evidence, but its SGLang
path received and ignored the precomputed feature tensor.

## Matched configuration

All modes used:

```text
execution mode                  fully_async
NUM_ROLLOUT                     2
NUM_STEPS_PER_ROLLOUT           2
ROLLOUT_BATCH_SIZE              8
N_SAMPLES_PER_PROMPT            8
GLOBAL_BATCH_SIZE               32
MICRO_BATCH_SIZE                1
MAX_STALENESS                   4
UPDATE_WEIGHTS_INTERVAL         1
actor                           DP2 on cards 0-1
rollout                         SGLang on card 2
actor_fwd                       Megatron on card 3
EVAL_INTERVAL                   4
SAVE_CHECKPOINTS                0
```

The CPU modes used one CPU vision replica, one reserved Ray CPU, a 1 GiB LRU,
and maximum encoder batch size eight. The GPU mode froze the GPU vision
tower and projection. Every run started from `13,135,872` bytes used on each
GPU and returned to that value at shutdown.

## Run inventory

| Mode | Ray job | W&B | Exit | Wall time |
|---|---|---|---:|---:|
| GPU | `raysubmit_Gr5FC3WM75vx2MHc` | `z695pdde` | 0 | 704.4 s |
| CPU with GPU encoder kept | `raysubmit_4EiVbt6dG5Dk3EMn` | `w7n3s64f` | 0 | 1024.7 s |
| CPU with GPU encoder skipped | `raysubmit_dC75CREAA7syfy5a` | `dlwserrn` | 0 | 1047.1 s |

The first GPU monitor attempt used an unrelated inherited virtual
environment and failed before Ray startup. Its `gpu_vram.json` artifact
is preserved but excluded. `gpu_vram_retry1.json` is the successful
matched GPU result.

All three accepted runs completed the 768-request baseline evaluation, two
64-sample rollouts, actor-forward, four successful optimizer updates, final
rollout and actor-forward weight synchronization, and clean Ray shutdown.

## Behavioral result

| Mode | Held-out reward | Valid action | Action A | Permuted control | Constant control |
|---|---:|---:|---:|---:|---:|
| GPU | 0.7090 | 1.0 | 0.4785 | 0.5234 | 0.5000 |
| CPU with GPU encoder kept | 0.6816 | 1.0 | 0.4785 | 0.5234 | 0.5000 |
| CPU with GPU encoder skipped | 0.7031 | 1.0 | 0.4531 | 0.5234 | 0.5000 |

These stochastic evaluations are consistent with the deterministic live
SGLang parity gate: both corrected CPU modes recover GPU-like task
behavior, and skipping the GPU encoder does not change the representation. The
fixed-input logit comparison remains the actual semantic parity proof.

## VRAM result

Values are peak deltas above the common idle baseline. Per-device peaks can
occur at different samples.

| Mode | Card 0 actor | Card 1 actor | Card 2 rollout | Card 3 actor-fwd | Simultaneous total |
|---|---:|---:|---:|---:|---:|
| GPU | 2.944 GiB | 2.511 GiB | 7.778 GiB | 7.257 GiB | 20.155 GiB |
| CPU with GPU encoder kept | 2.899 GiB | 2.443 GiB | 7.262 GiB | 7.228 GiB | 19.093 GiB |
| CPU with GPU encoder skipped | 2.840 GiB | 2.418 GiB | 7.215 GiB | 7.173 GiB | 19.279 GiB |

CPU with GPU encoder skipped reduced every per-device peak relative to CPU with GPU encoder kept:

| Device | Kept minus skipped |
|---|---:|
| Card 0 actor | 60.30 MiB |
| Card 1 actor | 26.21 MiB |
| Card 2 rollout | 48.03 MiB |
| Card 3 actor-fwd | 56.07 MiB |
| Sum of independent per-device reductions | 190.61 MiB |

The simultaneous total did not isolate the VRAM saved by skipping the GPU encoder in this fully
asynchronous repeat. Its peak was 190.18 MiB higher in the skipped run even
though every individual device peak was lower. The peak happened at a
different overlap of actor, rollout, and actor-forward allocations. The prior
matched run showed the opposite simultaneous ordering while producing very
similar per-device reductions from skipping the GPU encoder. Therefore:

- per-device VRAM savings are repeatable and role-local;
- a single fully asynchronous simultaneous maximum is schedule-sensitive;
  and
- a stronger cluster-level VRAM number requires repeated runs or a
  synchronized phase-specific memory probe, not selection of the lower of two
  asynchronous samples.

GPU-to-CPU comparisons include execution-path and allocation-timing
changes, not only visual weights. They must not be labeled as skipped-weight
savings.

## Stage instrumentation

The rollout path now records count, total, mean, p50, p95, and maximum for
CPU-service round trip, feature preparation, tensor-to-list conversion, HTTP
request construction, HTTP response wait/read/decode, and the existing image
processor, visual encode, generation, and post-generation phases. It also
records precomputed feature bytes and exact HTTP request/response body bytes.
Shared CPU feature preparation is counted once per prompt group.

### Baseline evaluation

The following values cover the same 768 generation requests. Totals across
concurrent requests are work/latency sums, not wall time.

| Metric | GPU | CPU with GPU encoder kept | CPU with GPU encoder skipped |
|---|---:|---:|---:|
| Mean generation request | 1.123 s | 4.502 s | 4.389 s |
| Mean HTTP request build | 0.00018 s | 0.3665 s | 0.3739 s |
| Total HTTP request build | 0.14 s | 281.48 s | 287.13 s |
| Mean HTTP response wait | 1.123 s | 4.135 s | 4.015 s |
| Mean image processing | 0.206 s | 0.804 s | 0.846 s |
| Mean CPU service round trip | n/a | 1.869 s | 2.069 s |
| Mean tensor-to-list conversion | n/a | 0.0206 s | 0.0241 s |
| Actual CPU backend encode time | n/a | 49.42 s | 53.56 s |
| HTTP request bytes, total | 1.40 MB | 2.233 GB | 2.233 GB |
| HTTP request bytes, mean | 1.82 KB | 2.907 MB | 2.907 MB |
| Raw precomputed feature, mean | n/a | 524,312 B | 524,312 B |
| SGLang router send warnings | 0 | 13 | 5 |

`httpx.send()` is intentionally left in its normal eager-response mode, so
`http_response_wait` includes response body transfer as well as SGLang queue,
feature reconstruction/H2D, prefill, and decode. The response bodies average
only about 682 bytes, so this does not affect the transport conclusion.

### Interpretation

The CPU backend is not the first bottleneck:

- it encoded only 128 unique images and spent 49-54 seconds doing so;
- its LRU served the other 640 evaluation requests from cache;
- tensor-to-list conversion averaged only 21-24 milliseconds; but
- every generation request still carried a JSON-expanded feature averaging
  2.907 MB.

The raw feature is 524,312 bytes. JSON expands it by about 5.5 times per
request. More importantly, the same feature is retransmitted for every sample
from a prompt. With eight samples per prompt, a 64-sample rollout transports
about 185 MB of request bodies even though its eight unique raw features total
only 4.19 MB. Evaluation similarly transmits 2.233 GB despite only 128 unique
features and a working CPU-side LRU.

Relative to GPU evaluation, CPU request bodies increased from 1.40 MB to
2.233 GB, approximately 1,599 times. JSON construction blocks the Python event
loop, and the large concurrent requests also stress the SGLang router. The
router warnings did not create client retries or incomplete samples in these
runs, but they are another symptom of the current boundary.

## Runtime result

| Mode | Wall time | Relative to GPU |
|---|---:|---:|
| GPU | 704.4 s | 1.000x |
| CPU with GPU encoder kept | 1024.7 s | 1.455x |
| CPU with GPU encoder skipped | 1047.1 s | 1.487x |

Skipping the GPU encoder changed CPU-vision wall time by only 2.2%. This confirms that
skipping the GPU encoder is a memory feature, not a throughput optimization. The corrected
CPU modes remain about 46-49% slower than GPU under this workload.

## Decision and next gate

Do not scale CPU replicas yet. The next implementation should remove repeated
feature transport:

1. add an SGLang-side precomputed-feature registry keyed by immutable
   `feature_id` plus `vision_revision`;
2. upload each feature once through a binary or same-node shared-memory path;
3. send only the feature identity in repeated generation requests;
4. fail loudly on a missing ID, revision mismatch, shape mismatch, or expired
   entry;
5. instrument registry upload, hit/miss, fetch, reconstruction/H2D, prefill,
   and eviction separately; and
6. rerun this matched GPU/CPU workload before testing more CPU threads or
   replicas.

The first target is not a general database. A bounded engine-local registry
with explicit lifecycle and LRU accounting is enough to test whether removing
JSON duplication closes the performance gap. Multi-engine and multi-replica
routing require `feature_id`-sticky placement or explicit replication and
remain a later gate.

## Evidence

### Logs

- GPU: `log/visual-xor-refinement-gpu-20260801_141157.log`
- CPU with GPU encoder kept: `log/visual-xor-refinement-cpu-20260801_142408.log`
- CPU with GPU encoder skipped: `log/visual-xor-refinement-cpu-skip-gpu-encoder-20260801_144228.log`

### VRAM artifacts

- `benchmark_results/cpu_vision/20260801_gpu_cpu_vision_performance/gpu_vram_retry1.json`
- `benchmark_results/cpu_vision/20260801_gpu_cpu_vision_performance/cpu_vram.json`
- `benchmark_results/cpu_vision/20260801_gpu_cpu_vision_performance/cpu_skip_gpu_encoder_vram.json`

### W&B

- GPU: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/z695pdde`
- CPU with GPU encoder kept: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/w7n3s64f`
- CPU with GPU encoder skipped: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/dlwserrn`
