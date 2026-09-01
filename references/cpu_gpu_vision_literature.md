# Frozen vision-feature path literature map

- **Prepared:** 27 August 2026
- **Paper target:** ICLR 2027
- **System scope:** Frozen Qwen3-VL image encoder shared by asynchronous Relax rollout and training

## Paper-Anatomy Anchor

[Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783) is the structural
anchor. It makes one exact execution problem legible, supplies two alternatives, predicts their crossover with a
cost model, and validates the model with end-to-end scaling and operation-level breakdowns. This paper
paper should follow the same sequence:

1. Establish the policy-invariance observation and duplicated visual work.
2. Define exact GPU and CPU-precomputed execution paths.
3. Model exposed CPU work rather than total eliminated work.
4. Select the device from measured time models.
5. Validate semantic parity, time-prediction error, device-choice match rate, extra time, training-cycle
   time, and learning quality.

The anchor is a paper-structure model, not a technical baseline: it addresses long-context inference rather than
multimodal RL.

## Closest Systems Work

| Work | Mechanism and evaluation lesson | Boundary relative to this project |
|---|---|---|
| [HydraInfer](https://arxiv.org/abs/2505.12658) | Disaggregates multimodal encode, prefill, and decode; evaluates heterogeneous stage scheduling, batching, and SLO throughput. | Serves independent requests; it does not exploit a frozen representation shared across policy versions, rollout, and training. |
| [HeteroServe](https://arxiv.org/abs/2603.12707) | Places multimodal stages across heterogeneous GPU tiers using a cost model and the modality boundary. | Uses cross-tier GPUs for serving; this work uses CPU/GPU execution inside asynchronous RL and validates Megatron as a second consumer. |
| [HeteGen](https://arxiv.org/abs/2403.01164) | Overlaps heterogeneous CPU/GPU inference under constrained GPU memory. | Primarily offloads model execution or parameters; this work produces immutable frozen vision features. |
| [vLLM-Omni](https://arxiv.org/abs/2602.02204) | Disaggregates multimodal stages and studies stage-specific scheduling. | Inference-only; no shared training consumer or policy-version contract. |

## RL Systems Evaluation Models

| Work | Evaluation pattern to reuse | Difference |
|---|---|---|
| [AReaL](https://arxiv.org/abs/2505.24298) | Pair throughput and staleness with learning curves and matched final quality; isolate system and algorithm ablations. | Language reasoning rather than multimodal perception device choice. |
| [HybridFlow](https://arxiv.org/abs/2409.19256) | Separates end-to-end throughput, resource allocation, transition, and engine effects. | Places RL model roles rather than frozen submodel computation. |
| [RLHFuse](https://arxiv.org/abs/2409.13221) | Show critical-path changes with inter- and intra-stage fusion. | Training-stage fusion rather than modality-level heterogeneous execution. |
| [ReaL](https://proceedings.mlsys.org/paper_files/paper/2025/hash/3b3889d313ba9476c12c2d77ea66b24f-Abstract-Conference.html) | Evaluate automatic resource allocation against fixed device choices and report end-to-end throughput. | Reallocates GPUs among RL roles; the frozen vision-feature path introduces CPU work and a representation contract. |

## Learning-Quality Context

[VL-DAC](https://arxiv.org/abs/2508.04280) motivates using a controlled synthetic visual environment for causal
debugging while still requiring real multimodal evaluation. Here, visual XOR supplies exact controls and a workload
trace; [Geo3K](https://huggingface.co/datasets/hiyouga/geometry3k) supplies real visual reasoning, held-out accuracy,
and time-to-quality evidence.

## Intended Related-Work Claim

> Multimodal serving systems disaggregate inference stages, and RL systems separately schedule rollout and
> training. The Relax frozen vision-feature path instead exploits a frozen modality computation that is invariant across
> policy updates, produces one revisioned representation, and supplies it to both SGLang inference and Megatron
> training. Its device-choice rule is governed by exposed critical-path cost, not merely eliminated computation or
> transfer bytes.

## Resource-Import Status

The seven-paper `$add-resource` import is specified for the global `LLM-RL` domain with human-written summaries.
The 27 August compute allocation cannot reach MLSys or arXiv through its HTTPS proxy (`403` tunnel denial), so no
incomplete entries were written. Run the recorded imports from a network-enabled shell; retain Markdown only and
use the summaries in this map.
