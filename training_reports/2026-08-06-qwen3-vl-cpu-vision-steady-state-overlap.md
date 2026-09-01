# Qwen3-VL CPU Vision Steady-State Overlap

- **Date:** 2026-08-06
- **Status:** passed matched 20-rollout fully asynchronous CPU with GPU encoder skipped and
  GPU performance runs; GPU grouped sampling remains a qualified
  performance counterfactual because its strict branch-logit gate did not pass
- **Hardware:** four AMD MI210 GPUs; one Ray CPU reserved for one frozen CPU
  vision replica in the CPU mode
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **Evaluation:** disabled in both runs
- **Checkpoint saving:** disabled in both runs

## Result

CPU vision offload is already useful for this workload even before replacing
the remaining packed JSON transport. With the CPU cache deliberately made too
small to retain any feature, one CPU encoder still kept rollout production
well ahead of the actor. Every one of the 105 seconds occupied by CPU-mode
rollout intervals overlapped an actor optimizer interval, and the actor waited
only 0.238 seconds per steady cycle for data.

The CPU path made each rollout slower than GPU vision, but removed the
frozen visual computation from SGLang, actor-forward, and actor training. The
net result was a 13.68% reduction in measured actor compute time, a 9.67%
reduction in complete steady training-cycle time, and a 738.03 MiB reduction at
the simultaneous four-card VRAM peak. The complete monitored CPU job finished
44.50 seconds sooner despite the extra CPU-service startup.

This result answers the immediate design question: a binary feature registry
is not required to demonstrate useful overlap or an end-to-end gain on the
current visual-XOR workload. The packed CPU request is still 383 times larger
than the GPU raw-image request, so transport remains inefficient, but it is
not currently pacing training. Actor compute and per-cycle weight propagation
are the larger boundaries.

## Matched workload

Both runs used the same:

- fully asynchronous execution with `MAX_STALENESS=4`;
- actor data parallel size two on cards 0-1;
- one SGLang rollout engine on card 2 and actor-forward on card 3;
- 20 rollout cycles and two optimizer steps per cycle;
- rollout batch size eight and eight stochastic samples per prompt;
- global batch size 32, micro batch size one, and learning rate `3e-6`;
- visual-XOR refinement training data and output-only A/B system prompt;
- evaluation disabled with `ENABLE_EVAL=0`; and
- checkpoint saving disabled with `SAVE_CHECKPOINTS=0`.

The CPU run additionally set:

```bash
SKIP_GPU_VISION_ENCODER=1
VISION_ENCODER_CACHE_MAX_BYTES=1
```

One byte cannot retain a 524,312-byte feature bundle. The final counters prove
that the run did not benefit from cache reuse: 160 misses, 160 evictions, zero
hits, zero resident entries, and 160 actual image encodes.

## Run identities and artifacts

| Mode | Ray job | W&B run | Log | VRAM/timeline directory |
|---|---|---|---|---|
| CPU with GPU encoder skipped, no cache | `raysubmit_fahWwVRtsD5QD6An` | `drbdd32z` | `log/visual-xor-refinement-cpu-skip-gpu-encoder-20260806_081340.log` | `benchmark_results/cpu_vision/20260806_overlap_cpu_skip_gpu_encoder_nocache/` |
| GPU, grouped | `raysubmit_Eu9kS3dHqyZJ37rX` | `k936t7l6` | `log/visual-xor-refinement-gpu-20260806_083125.log` | `benchmark_results/cpu_vision/20260806_overlap_GPU_grouped/` |

Both jobs exited with code zero, completed 20 rollout intervals and 40
optimizer intervals, performed their final actor-forward and SGLang weight
updates, shut down Ray, and returned all four GPUs to the common 52,543,488
byte idle baseline.

## GPU grouped correctness qualification

GPU raw-image generation was extended through the same narrow fresh-group
contract as CPU-precomputed generation: one prompt/media object, one
processor pass, one raw-media encoding, and one SGLang request with `n=8`.
The standalone live artifact is
`benchmark_results/cpu_vision/20260806_GPU_grouped_n8/probe.json`.

The probe returned the same generated token and text for all 64 GPU grouped
branches, and the ordinary scalar GPU-versus-packed comparison still
passed with a maximum action-log-probability delta of `1.4305e-6`. However,
GPU grouped versus GPU scalar exceeded the strict `0.01` branch-score
gates:

| GPU grouped gate | Observed | Limit |
|---|---:|---:|
| Maximum action-log-probability delta | `0.0393803` | `0.01` |
| Maximum action-margin delta | `0.06249997` | `0.01` |
| Maximum action-probability delta | `0.01457146` | `0.01` |

The gate therefore correctly remained failed. The GPU run is a matched
throughput and memory counterfactual with token-level agreement, not strict
evidence that stochastic `n=8` GPU branching reproduces scalar GPU
logits. The thresholds were not loosened.

## Overlap and capacity evidence

The timeline analyzer pairs `Start/Finish rollout` events and every Megatron
`starting forward_backward`/`finished optimizer.step` interval. Its artifacts
contain no unmatched events.

| Timeline metric | CPU with GPU encoder skipped | GPU grouped |
|---|---:|---:|
| Complete rollout intervals | 20 | 20 |
| Complete optimizer intervals | 40 | 40 |
| Rollout interval wall | 105 s | 79 s |
| Actor optimizer interval wall | 299 s | 344 s |
| Rollout time overlapping actor work | 105 s / 100% | 79 s / 100% |

The CPU backend spent 38.079 seconds actually encoding 160 images: mean 1.904
seconds, median 1.762 seconds, and maximum 2.883 seconds per eight-image
rollout batch. The complete steady CPU rollout took 4.789 seconds on average.
It repeatedly filled the five-partition staleness window and waited for the
actor. All 20 CPU rollouts were generated by 08:27:25 UTC while training-cycle 15
was still running.

The GPU CPU cache showed the same scheduling relationship. Its steady
rollout averaged 3.263 seconds, filled the same staleness window, and generated
all 20 rollouts while training-cycle 15 was still running. In both modes the
actor's steady data wait stayed near zero.

The overlap percentage has one-second log resolution and describes paired
rollout versus optimizer intervals; it does not isolate SGLang reconstruction,
H2D, or prefill sub-intervals. The independent CPU backend counters and actor
wait time establish the stronger capacity result: uncached CPU vision supplied
data substantially faster than the consumer used it.

## Steady-state timing

Cycle zero contains initialization and queue warmup and is excluded. The table
summarizes cycles 1-19.

| Metric | CPU with GPU encoder skipped mean | GPU grouped mean | CPU change |
|---|---:|---:|---:|
| Actor data wait | 0.238 s | 0.232 s | +0.006 s |
| Actor compute | 13.531 s | 15.674 s | 13.68% lower |
| Fully-async weight update | 12.712 s | 13.464 s | 5.59% lower |
| Training portion | 26.591 s | 29.469 s | 9.77% lower |
| Complete training-cycle | 26.829 s | 29.700 s | 9.67% lower |
| Actor wait ratio | 0.880% | 0.786% | both negligible |
| Steady rollout time | 4.789 s | 3.263 s | 46.77% higher |

CPU rollout is slower, but it remains about 5.6 times shorter than the CPU
training-cycle. Offloading the frozen visual computation from the actor path more
than compensates for the slower CPU cache. The 2.14-second actor-compute
reduction is consistent with skipping the frozen visual tower, while the
0.75-second weight-update reduction is consistent with the smaller skipped
model state; repeat measurements are still required before assigning every
second causally.

The monitor wall times were 960.778 seconds for CPU and 1005.279 seconds for
GPU. CPU service startup delayed rollout 0 by about 16 seconds relative to
GPU, but the faster steady drain recovered that cost and finished the job
44.501 seconds earlier. The two stochastic trajectories were comparable but
not identical: mean rollout reward was 0.7203 CPU versus 0.7188 GPU, valid
action rate was 1.0 in both, and response length was exactly two tokens in
both.

## Transport

| Per 64-sample rollout | CPU with GPU encoder skipped | GPU grouped |
|---|---:|---:|
| SGLang generation requests | 8 | 8 |
| Parallel samples returned | 64 | 64 |
| HTTP request-body bytes | 5,602,384 | 14,609 mean |
| Raw precomputed feature bytes | 4,194,496 | N/A |

Packed CPU features therefore used about 383.5 times the GPU request bytes,
or 112,047,680 bytes over the 20 rollouts. That is a genuine inefficiency, but
the zero actor-starvation result shows that it is not the next blocking
optimization for this workload. A binary upload/ID registry should remain
deferred until a larger rollout workload, evaluation, more engines, or remote
transport makes these bytes pace the consumer.

## VRAM

Peak deltas above the common idle baseline were sampled every 0.5 seconds.

| Role/device | CPU with GPU encoder skipped | GPU grouped | CPU reduction |
|---|---:|---:|---:|
| Actor card 0 | 2.900 GiB | 3.006 GiB | 108.09 MiB |
| Actor card 1 | 2.455 GiB | 2.531 GiB | 77.39 MiB |
| SGLang card 2 | 7.215 GiB | 7.702 GiB | 498.22 MiB |
| Actor-forward card 3 | 7.317 GiB | 7.392 GiB | 76.27 MiB |
| Simultaneous four-card peak | 19.841 GiB | 20.562 GiB | 738.03 MiB |

The rollout card has the clearest isolated saving because SGLang no longer
materializes or executes the visual tower. The smaller actor and actor-forward
deltas combine skipped frozen parameters with changes in temporary visual
activations. Peak sampling is process-level evidence, not a precise allocator
decomposition.

## Decision

Keep the current stateless packed CPU path and skip the GPU encoder for the next
scale experiment. Do not implement the feature registry or add CPU replicas
yet. The next performance experiment should first increase rollout demand
(larger prompt batch, more engines, or a larger/faster actor configuration)
until actor wait rises or the CPU CPU cache no longer reaches staleness
backpressure. That identifies the actual CPU/transport capacity boundary.

Separately, profile the 12-13 second weight-update phase: it consumes almost
half of each steady training-cycle and is now a larger optimization target than
CPU vision transport. Before using grouped GPU generation as a correctness
baseline, investigate its branch-logit difference rather than weakening the
strict gate.
