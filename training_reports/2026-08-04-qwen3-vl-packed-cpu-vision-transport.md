# Qwen3-VL Packed CPU Vision Transport

- **Date:** 2026-08-04
- **Status:** passed serialization benchmark, live native-versus-precomputed
  parity, grouped evaluation, and a fully asynchronous two-rollout training run
- **Hardware:** four AMD MI210 GPUs; one Ray CPU reserved for one frozen CPU
  vision replica
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **GPU vision state:** omitted from SGLang, actor, and actor-forward
- **Checkpoint saving:** disabled

## Result

Relax still uses a JSON HTTP envelope for precomputed vision requests, but it
no longer represents a BF16 feature as a nested array of JSON numbers. The
immutable final-plus-DeepStack bundle is serialized as contiguous BF16 bytes,
base64-encoded in the existing `precomputed_embedding` object, and decoded
before SGLang's base multimodal processor reads `feature`.

This stateless change reduced the accepted grouped evaluation from
1,117,116,704 to 268,913,280 request bytes, a 75.93% reduction. Combined with
evaluation grouping, it reduced the original scalar nested-list evaluation
from 2,232,651,920 bytes to 268,913,280 bytes, an 87.96% reduction. Both
64-sample training rollouts used eight SGLang requests and 5,602,384 request
bytes. Actor-forward, all four optimizer updates, final weight synchronization,
and shutdown passed.

## Evaluation grouping correction

Evaluation branching had been coupled to `group_rm`. That flag controls reward
semantics, not whether one prompt's sampled branches can share one generation
request. A failing test first established that an evaluation prompt with
`n_samples_per_eval_prompt > 1` must form one generation group even when
`group_rm` is false. The implementation now always dispatches one per-prompt
group while preserving per-sample reward calculation.

The immutable multimodal object is shared across the branches so the existing
SGLang native-parallel eligibility proof applies. Deterministic `n=1` controls
remain scalar. The accepted nested-list measurement therefore used:

- 128 held-out SGLang requests with `n=4`, returning 512 samples; and
- 256 scalar deterministic control requests, returning 256 samples.

That cut evaluation from 768 to 384 HTTP requests and from 2,232,651,920 to
1,117,116,704 request bytes before changing the feature representation.

## Transport benchmark and decision

Artifact:
`benchmark_results/cpu_vision/20260804_packed_transport/serialization.json`

The local benchmark used the real `[64, 4096]` BF16 final-plus-DeepStack
feature, whose raw payload is 524,312 bytes.

| Representation | Request or upload bytes | Encode median | HTTP body build median | Decode median |
|---|---:|---:|---:|---:|
| Nested float JSON | 3,017,467 | 7.507 ms | 66.816 ms | 7.659 ms |
| Inline BF16 base64 JSON | 700,301 | 0.522 ms | 15.068 ms | 0.971 ms |
| Binary registry upload | 524,457 | N/A | N/A | N/A |
| Registry ID request | 1,155 | N/A | N/A | N/A |

For the grouped 384-request evaluation, the benchmark projected 1,158,707,328
bytes for nested JSON, 268,915,584 bytes for inline BF16, and 67,574,016 bytes
for one binary upload per unique feature plus ID requests.

Inline BF16 was selected because it removes 76.79% of projected nested-JSON
traffic without adding mutable engine state, ownership, eviction, or
missing-ID recovery. A registry could remove another roughly 201 MB in this
evaluation, but that remaining saving does not yet justify its larger routing
contract. The live run must be measured again before revisiting that decision.

## Wire contract

The existing `format: precomputed_embedding` object now contains:

- `feature_b64`: base64 over the exact contiguous CPU BF16 bytes;
- `feature_dtype: bfloat16`;
- `feature_shape`;
- the existing grid, feature ID, and vision revision metadata.

The decoder fails loudly on unsupported dtype, invalid base64, negative or
malformed shapes, and shape/byte-count mismatch. It reconstructs the original
BF16 bit pattern exactly. Legacy nested-list payloads remain readable during
the transition, and native PIL/media objects bypass the packed decoder.

Two live failures tightened the integration seam:

1. Decoding after SGLang's base processor was too late; the processor had
   already indexed `dict_item["feature"]` and raised `KeyError`.
2. Treating every multimodal item as a mapping broke native PIL input with
   `AttributeError: PngImageFile has no attribute get`.

The final decoder runs before that base read and checks that an item is a
precomputed mapping before examining its format. Transport-only packed fields
are removed after reconstruction so downstream model-specific data keeps the
established contract.

The W&B metric name `precomputed_to_list_time` is retained for dashboard
continuity, but it now measures packed feature serialization rather than
tensor-to-list conversion.

## Live parity gate

Artifact:
`benchmark_results/cpu_vision/20260804_packed_transport/sglang_live_parity.json`

The fixed eight-image native-GPU versus packed-CPU comparison passed inside
one live SGLang transformers server.

| Gate | Result |
|---|---:|
| Generated token match | 8 / 8 |
| Text match | 8 / 8 |
| Maximum A/B log-probability delta | `1.430511474609375e-06` |
| Maximum action-margin delta | `1.043081283569336e-07` |
| Maximum action-probability delta | `1.7429432008775336e-08` |

The same server also completed eight `n=8` packed requests:

| Gate | Result |
|---|---:|
| Generated outputs | 64 |
| Generation requests | 8 |
| Output/text match | 64 / 64 |
| Maximum grouped A/B log-probability delta | `9.5367431640625e-07` |
| Grouped request bytes | 5,602,616 |
| Scalar-equivalent request bytes | 44,820,416 |
| Within-prompt traffic reduction | 87.50% |

## Accepted matched Relax run

| Item | Value |
|---|---|
| Ray job | `raysubmit_c3YzyTcrTuidT6us` |
| W&B | `kjp5zomx` |
| Log | `log/visual-xor-refinement-cpu-omitted-20260804_133243.log` |
| VRAM artifact | `benchmark_results/cpu_vision/20260804_packed_transport/cpu_omitted_packed_two_rollout_vram.json` |
| Exit code | 0 |
| Monitor wall time | 680.384 s |

Timeline:

- `+00:00` — monitor started at 13:32:43 UTC;
- `+06:57` — SGLang reported ready;
- `+07:03` — rollout manager was attached to the actor;
- `+09:47` — grouped evaluation completed and rollout 0 started;
- `+09:51` — rollout 0 completed and rollout 1 started;
- `+09:56` — rollout 1 completed;
- `+10:16` to `+10:54` — four optimizer updates completed successfully;
- `+11:08` — controller shutdown completed;
- `+11:12` — Ray shutdown completed; and
- `+11:20` — the monitor wrote the final idle-memory artifact.

### Evaluation

| Metric | Packed result | Grouped nested-list result | Change |
|---|---:|---:|---:|
| HTTP requests | 384 | 384 | unchanged |
| Returned samples | 768 | 768 | unchanged |
| SGLang grouped requests | 128 | 128 | unchanged |
| SGLang grouped samples | 512 | 512 | unchanged |
| Request-body bytes | 268,913,280 | 1,117,116,704 | 75.93% lower |
| Request-build mean | 24.275 ms | 336.2 ms | 92.78% lower |
| Packed serialization mean | 5.469 ms | 64.5 ms | 91.52% lower |
| Response-wait mean | 3.240 s | 7.071 s | contextual only |
| Held-out reward | 0.7109375 | 0.693359375 | stochastic |
| Permuted control | 0.5234375 | 0.5234375 | equal |
| Constant control | 0.5 | 0.5 | equal |
| Valid-action rate | 1.0 | 1.0 | equal |

The CPU LRU received 384 evaluation feature requests, encoded 128 unique
images, served 256 hits, held 128 entries (67,111,936 bytes), and evicted
nothing. Grouping changed request fanout; it did not change the 128 unique
encoder inputs.

The response-wait change is not isolated enough to claim an inference speedup:
both runs were fully asynchronous and server scheduling varied. Request bytes
and client-side construction/serialization are the direct transport evidence.

### Rollout and training

| Metric | Rollout 0 | Rollout 1 |
|---|---:|---:|
| Generated samples | 64 | 64 |
| SGLang requests | 8 | 8 |
| Parallel samples per request | 8 | 8 |
| Request-body bytes | 5,602,384 | 5,602,384 |
| Request-build mean | 13.566 ms | 17.340 ms |
| Rollout wall time | 3.777 s | 5.142 s |
| Reward | 0.6875 | 0.671875 |
| Valid-action rate | 1.0 | 1.0 |
| CPU encodes | 8 | 8 |

Against the same-day grouped nested-list rollouts, mean request traffic fell
from 23,238,097 to 5,602,384 bytes, a 75.89% reduction. All four optimizer
updates reported `update_successful=True`; gradient norms were `1.0058`,
`1.3580`, `4.9491`, and `1.3629`. Actor-forward completed both steps, the final
rollout and actor-forward weight markers arrived, and no checkpoint was saved.

### VRAM

Peak deltas above the common idle baseline were:

| Role/device | Packed peak delta |
|---|---:|
| Actor card 0 | 2.838 GiB |
| Actor card 1 | 2.418 GiB |
| SGLang card 2 | 7.215 GiB |
| Actor-forward card 3 | 7.173 GiB |
| Simultaneous total | 18.859 GiB |

These per-device peaks match the grouped nested-list run within measurement
noise. Packing changes host serialization and on-wire traffic, not model or
activation allocation. Final memory returned exactly to the 52,543,488-byte
four-card baseline, and no Ray or SGLang process remained.

## Decision and remaining work

Use stateless packed BF16 as the default precomputed Qwen3-VL transport. Do not
add a feature registry or scale CPU replicas yet. The feature still crosses
the router once per distinct HTTP request, and base64 has its expected 4/3
size expansion, but the accepted workload is no longer dominated by nested
numeric JSON.

Reconsider binary upload plus ID-only generation only if a matched measurement
shows the remaining 268.9 MB evaluation traffic or reconstruction/H2D is the
next limiting phase. Such a design must first solve multi-engine ownership,
routing affinity or replication, bounded LRU eviction, feature revision,
missing-ID recovery, and cleanup. Multiple images, video, PP/CP greater than
one, and chunked multimodal prefill remain outside this gate.

## Validation

```text
54 passed in 6.14s
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

The focused suite includes bit-exact BF16 round trips, malformed wire payloads,
legacy nested lists, native-media bypass, early SGLang processor materialization,
evaluation grouping independent of reward grouping, shared multimodal identity,
HTTP metrics, and the live parity harness.
