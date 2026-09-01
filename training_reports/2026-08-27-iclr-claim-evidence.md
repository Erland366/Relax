# ICLR 2027 claim-evidence table

- **Frozen:** 27 August 2026
- **Rule:** A claim enters the paper only after its required artifact passes the stated gate.

| Candidate claim | Current evidence | Required gate before publication |
|---|---|---|
| Frozen CPU Qwen3-VL features preserve GPU language-model computation. | Supported for the local 0.37B model. | Final and every DeepStack stream, standalone and live logits, actor-forward log probabilities, loss, and trainable-gradient parity. Repeat on 4B before generalizing across model scale. |
| Moving frozen vision computation to CPU improves complete training-cycle time under `prompts32_samples2`. | Supported. | Matched job `154079`: `7.328%` mean speedup with paired 95% confidence interval `3.711%` to `10.944%`. |
| A precomputed plan can select CPU or GPU vision for a controlled rollout workload. | Implemented; live evidence pending. | Five matched repeats. Require at least 90% match rate among cycles over the 5% minimum gap, at least half of cycles over that gap, less than 5% mean extra time, identical request fanout, and both device choices in every repeat. |
| CPU-only deployment can skip GPU vision-encoder weights in rollout and actor-forward services. | Small-model correctness supported; 4B memory evidence pending. | Compare CPU vision with the GPU encoder kept against CPU vision with the GPU encoder skipped, using parity plus synchronized per-device VRAM. Do not attribute unrelated allocator changes to the vision parameters. |
| Full-dataset preload removes repeated feature work but does not improve the current training critical path. | Supported negative result. | Job `160198`: raw training-cycle speedup `0.324%`, 95% confidence interval `-4.100%` to `4.748%`; 22-cycle preload-inclusive speedup `-2.889%`, 95% confidence interval `-7.340%` to `1.562%`. |
| Online automatic device choice improves real multimodal RL time-to-quality without reducing quality. | Not implemented; do not claim. | First implement and validate runtime batch-workload observation. Then run three Geo3K seeds per setting with identical sample and update budgets, final accuracy within two absolute percentage points, and a supported wall-time gain. |
| Measured CPU/GPU time models predict a useful device crossover. | Pending. | Rank-aware timing grid, held-out prediction error, crossover heatmap, and matched extra time. Report any predictor removed because Qwen3-VL feature bytes are collinear with visual-token and image counts. |
| The implementation is a generic VLM-offload framework. | Rejected for this submission. | The current backend is Qwen3-VL image-only. Do not claim another model family, video, a trainable vision encoder, pipeline parallelism above one, or cross-node feature storage. |

## Language guardrails

- Say **packed BF16 base64 in a JSON HTTP envelope**, not binary transport.
- Say **safe while the complete vision encoder and merger remain frozen**, not universally stale-safe.
- Say **one logical feature contract consumed by SGLang and Megatron**, not one shared physical tensor.
- Say **cache mechanism benefit** for byte, serialization, and RTT reductions; do not turn them into a throughput
  claim.
- Report automatic device choice with GPU weights kept separately from CPU-only deployment with the GPU encoder
  skipped.
- Call the implemented method **plan-based automatic device choice**, not online adaptation. Geo3K currently supports
  fixed GPU, fixed CPU with the GPU encoder kept, and fixed CPU with the GPU encoder skipped. Automatic Geo3K choice
  remains disabled until runtime batch descriptors drive the decision.
