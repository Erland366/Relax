# Qwen3-VL Live Precomputed DeepStack Parity

- **Date:** 2026-07-31
- **Status:** Passed
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **Checkpoint saving:** disabled

## Result

The live Qwen3-VL CPU-vision correctness blocker is closed for the tested
single-image, PP1/CP1 path.

GPU vision and CPU-precomputed final plus DeepStack features produced
the same decision on all eight fixed images inside the real SGLang
transformers backend:

| Metric | Result | Gate |
|---|---:|---:|
| Generated-token match rate | `1.0` | `1.0` |
| Decoded-text match rate | `1.0` | `1.0` |
| Maximum A/B log-probability absolute delta | `1.430511474609375e-06` | `0.01` |
| Maximum A-vs-B margin absolute delta | `1.043081283569336e-07` | `0.01` |
| Maximum normalized action-probability delta | `1.7429432036530912e-08` | `0.01` |

The subsequent skipped-GPU-weight Relax run completed evaluation, rollout,
Megatron actor-forward, two optimizer updates, final weight synchronization,
and clean shutdown. Actor-forward and SGLang agreed on the sampled-token
log-probabilities to below `5e-7` mean absolute error at both optimizer
updates.

## Root cause

Relax already serialized the complete Qwen3-VL feature bundle correctly:

```text
[final projection | DeepStack 0 | DeepStack 1 | DeepStack 2]
```

The mismatch was in SGLang's generic transformers multimodal wrapper. Its
normal forward path collected raw `item.feature` values for visual execution,
but did not consume `item.precomputed_embeddings`. The CPU feature payload was
therefore parsed and transported while the language model still received its
ordinary image-token embeddings. This explains the earlier chance-level CPU
evaluation and action bias without implicating the CPU encoder or GPU-weight
skipping the GPU encoder.

The Relax compatibility patch now detects an all-precomputed Qwen3-VL prefill
batch, validates the packed width, splits the final and three DeepStack
streams, scatters final visual embeddings into the processor-expanded image
token positions, and calls the Qwen3-VL language model with both
`visual_pos_masks` and `deepstack_visual_embeds`. Decode and raw-image
requests retain the original SGLang path. Mixed GPU/CPU-precomputed batches
fail explicitly rather than silently selecting one representation.

## Live skipped-weight Relax gate

- **Log:**
  `log/visual-xor-refinement-cpu-skip-gpu-encoder-20260731_115759.log`
- **Ray job:** `raysubmit_MS2w23VJ8shDwHtP`
- **Offline W&B:**
  `log/wandb/wandb/offline-run-20260731_115848-4ril6b79`
- **Workload:** one fully asynchronous rollout, two optimizer updates
- **Resources:** actor DP2, SGLang one GPU, actor-forward one GPU, CPU vision
  one replica
- **GPU visual weights:** skipped in SGLang, actor-forward, and actor

### Baseline evaluation

| Dataset/metric | Result |
|---|---:|
| Held-out reward | `0.69921875` |
| Held-out valid action | `1.0` |
| Held-out action A | `0.453125` |
| Permuted control | `0.5234375` |
| Constant-image control | `0.5` |

The earlier GPU run scored `0.6816` on the held-out set. These are
separate stochastic evaluations, so their difference is not a parity metric;
the important change is that the corrected CPU path recovered from `0.5000`
and the `0.7148` action-A collapse to GPU-like behavior. The deterministic
SGLang artifact below is the actual parity gate.

### Cache evidence

The baseline evaluation made 768 CPU-vision requests over 128 unique images:

| Metric | Result |
|---|---:|
| Requests | `768` |
| Cache hits | `640` |
| Cache misses / backend encodes | `128` |
| Hit rate | `0.8333333333333334` |
| Evictions | `0` |
| Resident feature bytes | `67,111,936` |
| Backend encode time | `37.9273 s` |

### Megatron/SGLang sampled-token agreement

| Optimizer step | Train-vs-rollout log-probability absolute difference | Probability absolute difference |
|---:|---:|---:|
| 0 | `4.160376931849896e-07` | `2.2724270820617676e-07` |
| 1 | `4.769898396261851e-07` | `2.7474015951156616e-07` |

The final rollout aggregates also differed by only
`abs(-0.33503444492816925 - -0.33503396809101105) = 4.76837158203125e-07`.
This proves that the precomputed feature representation used for rollout is
compatible with the Megatron actor-forward/training boundary for the actual
sample batch. It is a cross-backend sampled-token gate, not a second
within-Megatron GPU-versus-CPU-precomputed full-vocabulary comparison.

## Deterministic live SGLang gate

The reusable probe launches one SGLang server with the visual weights
resident, sends every fixed image once through GPU vision and once
through CPU-precomputed vision, flushes the cache between paths, and requests
the exact next-token scores for action tokens A and B:

```bash
python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
  --checkpoint "$VISUAL_REFINEMENT_SFT" \
  --dataset "$VISUAL_REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/sglang_live_parity.json \
  --num-images 8 \
  --host 127.0.0.1 \
  --port 31000 \
  --base-gpu-id 0
```

The command must run with the same SGLang Python and ROCm `sgl_kernel` paths
used by the Relax launcher. It installs no packages and terminates the server
process before writing the final result.

## Evidence

- Deterministic artifact:
  `benchmark_results/cpu_vision/20260731_sglang_live_parity/parity.json`
- Deterministic server log:
  `log/sglang-cpu-vision-parity-20260731.log`
- Live Relax log:
  `log/visual-xor-refinement-cpu-skip-gpu-encoder-20260731_115759.log`
- Local HF parity artifact:
  `benchmark_results/cpu_vision/20260730_hf_parity/parity.json`

## Remaining scope limits

This result does not validate video, multiple images per request, context
parallelism, pipeline parallelism greater than one, chunked multimodal
prefill, mixed GPU/CPU-precomputed batching, or throughput scaling. SGLang
disabled chunked prefill for this multimodal transformers run. Those remain
separate gates; none should weaken the now-passing single-image PP1/CP1
correctness result.
