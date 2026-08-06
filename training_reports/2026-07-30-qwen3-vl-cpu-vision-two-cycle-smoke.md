# Qwen3-VL CPU Vision Two-Cycle Relax Smoke

- **Date:** 2026-07-30
- **Status:** Passed as a transport/Megatron smoke; SGLang semantic claim was
  superseded by the 2026-07-31 DeepStack fix
- **Ray job:** `raysubmit_sZm1DdH3fZ9bqGLj`
- **W&B:** `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/3rm56cjz`
- **Log:** `log/visual-xor-refinement-cpu-vision-20260730_122136.log`
- **Launcher:**
  `scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh`

## Objective

Prove that the frozen Qwen3-VL CPU visual path works through two complete
fully-asynchronous Relax cycles while the GPU visual weights remain resident:

```text
image
  -> CPU visual tower + merger + three DeepStack projections
  -> one precomputed feature bundle
  -> SGLang rollout
  -> transfer queue
  -> Megatron actor-forward
  -> Megatron DP2 backward and optimizer
  -> actor-forward and SGLang weight synchronization
```

This is a transport and Megatron correctness smoke. A later investigation
showed that the SGLang transformers wrapper in this run parsed but ignored the
precomputed embedding field. Therefore this historical run proves CPU
encoding, caching, transfer-queue transport, Megatron actor-forward/training,
and synchronization, but not SGLang semantic consumption. The corrected live
gate is documented in
[Qwen3-VL Live Precomputed DeepStack Parity](2026-07-31-qwen3-vl-live-precomputed-deepstack-parity.md).

## Configuration

The run used the visual-XOR refinement SFT checkpoint with:

```text
execution mode                 fully_async
NUM_ROLLOUT                    2
NUM_STEPS_PER_ROLLOUT          2
ROLLOUT_BATCH_SIZE             8
N_SAMPLES_PER_PROMPT           8
GLOBAL_BATCH_SIZE              32
MICRO_BATCH_SIZE               1
MAX_STALENESS                  4
UPDATE_WEIGHTS_INTERVAL        1
actor                          DP2 on GPUs 0-1
rollout                        SGLang on GPU 2
actor_fwd                      Megatron on GPU 3
vision_encoder                 one replica, one CPU thread
VISION_ENCODER_CACHE_MAX_BYTES 1 GiB
VISION_ENCODER_OMIT_GPU_WEIGHTS 0
SAVE_CHECKPOINTS               0
```

The Slurm allocation exposed four MI210 GPUs but only two physical CPU cores.
The CPU service therefore reserved one logical Ray CPU rather than the
eight-CPU development default that had made the first launch unschedulable.

## Failures exposed before the passing run

### 1. CPU resource request prevented service scheduling

The first launch requested eight CPUs for the vision replica. Ray had only a
small CPU allocation left after its control and service actors, so the vision
replica remained pending. The debug launcher now defaults to one CPU and
requires CPU-rich scaling experiments to override
`VISION_ENCODER_NUM_CPUS`.

### 2. SGLang received tokenizer IDs for processor-expanded visual features

The first request that reached SGLang failed MRoPE construction:

```text
prompt IDs:                51 tokens
processor-expanded IDs:   88 tokens
precomputed feature grid:  matched the 88-token representation
```

Raw-image SGLang expands the visual placeholder internally. The precomputed
path bypasses that work, so it must send the processor-expanded prompt IDs.
The rollout adapter now uses `processor_prompt_ids` for both the SGLang
payload and `sample.rollout_tokens` only when precomputed visual data is
present. Raw-image and text-only paths retain tokenizer IDs.

### 3. Megatron received numbered DeepStack streams

The next run completed evaluation and both rollouts, then actor-forward failed
because the transfer batch contained:

```text
deepstack_visual_embeds_0
deepstack_visual_embeds_1
deepstack_visual_embeds_2
```

while the model-instance wrapper expected one synthetic
`deepstack_visual_embeds` tuple. The wrapper now assembles the numbered keys
in the order defined by the model's `deepstack_visual_indexes`, removes them
before the original forward call, rejects mixed tuple/numbered input, and
reports missing numbered streams explicitly.

## Passing-run timeline

Offsets use the tmux launcher start at approximately `12:21:36 UTC`.

| Offset | Event |
|---:|---|
| `+00:00` | Debug launcher started |
| `+00:24` | Ray job submitted; runtime environment setup began |
| `+01:18` | Coordinator ready |
| `+01:32` | Metrics service ready; Relax began creating services |
| `+02:52` | CPU encoder loaded revision `7230823af0f8` |
| `+02:53` | CPU vision service ready |
| `+04:17` | Megatron actor-forward ready |
| `+05:03` | DP2 Megatron actor ready |
| `+05:35` | SGLang rollout ready; all five services registered |
| `+05:52` | Initial actor-to-actor-forward and actor-to-SGLang sync completed |
| `+05:53` | All services started; actor-forward waited on step `0/2` |
| `+10:43` | Baseline evaluation completed; rollout 0 started |
| `+11:05` | Rollout 0 completed; rollout 1 started |
| `+11:30` | First two optimizer updates completed |
| `+11:37` | Rollout 1 completed; all rollouts finished |
| `+12:02` | Final two optimizer updates and final weight sync completed |
| `+12:14` | Relax main function completed successfully |
| `+12:18` | Ray shutdown completed; job reported `succeeded` |

## Results

### Baseline evaluation

| Dataset | Reward/accuracy | Valid action rate |
|---|---:|---:|
| Held-out visual XOR | `0.466796875` | `1.0` |
| Permuted-label control | `0.5` | `1.0` |
| Constant-image control | `0.5` | `1.0` |

The baseline was biased toward action A (`0.724609375`) and was not expected
to converge during a two-cycle smoke. Both controls remained exactly at
chance and every response was a valid action.

### Rollout and optimizer

| Rollout ID | Samples | Reward mean | Valid action rate |
|---:|---:|---:|---:|
| 0 | 64 | `0.53125` | `1.0` |
| 1 | 64 | `0.5` | `1.0` |

Each rollout supplied two optimizer updates. All four updates were successful:

| Global train step | Rollout/local step | Loss | Gradient norm |
|---:|---|---:|---:|
| 0 | `0/0` | `-0.1890152` | `0.8668980` |
| 1 | `0/1` | `-0.0369092` | `1.2483702` |
| 2 | `1/0` | `-0.0607604` | `3.8410137` |
| 3 | `1/1` | `-0.2277948` | `3.1961463` |

Actor-forward log-probability work completed for both Relax steps. The final
actor weights reached actor-forward and SGLang before shutdown.

### Cache and CPU-service observations

The live log demonstrated both cold and warm behavior:

- uncached or concurrently queued CPU calls commonly took hundreds of
  milliseconds and reached approximately two seconds during rollout bursts;
- repeated-image calls commonly fell to approximately `4-20 ms`;
- the one-thread service sustained the correctness smoke but emitted Ray
  queue-length deadline warnings under bursts.

The cache exposes exact hits, misses, evictions, resident bytes, backend work,
and backend time through `VisionEncoder.get_metrics()`. This run did not
collect that method before service shutdown, so no exact cache-hit rate should
be inferred from request latency. Wiring these counters into W&B or the final
shutdown report is still required before a throughput or capacity claim.

### Nonfatal warnings

SGLang's Rust router intermittently logged `Failed to send typed request`
during high-concurrency evaluation and rollout bursts. Its retry path
completed every requested sample, all services remained healthy, and the job
succeeded. These warnings should still be counted in a later reliability
benchmark rather than treated as desirable behavior.

## Acceptance gates

- [x] CPU encoder loaded the expected frozen visual revision.
- [x] SGLang received processor-expanded IDs with the precomputed feature grid.
- [ ] This historical SGLang build consumed the precomputed feature tensor.
- [x] Baseline evaluation completed all `768` samples.
- [x] Both rollout IDs completed all `64` samples.
- [x] Megatron actor-forward consumed final plus all three DeepStack streams.
- [x] The DP2 actor completed four backward/optimizer updates.
- [x] Final weights synchronized to actor-forward and SGLang.
- [x] The job exited successfully and released all four GPUs.
- [x] No model or optimizer checkpoint was written.
- [ ] Exact cache counters were captured at shutdown.
- [x] A later corrected SGLang build passed fixed-image native-versus-
      precomputed next-token parity.
- [x] A later one-cycle smoke passed with GPU visual weights omitted and
      Megatron/SGLang sampled-token differences below `5e-7`.
- [ ] Native GPU, CPU-resident, and CPU-omitted VRAM/throughput were compared.

## Next gate

The original next gates were completed on 2026-07-31: GPU-weight omission,
per-device VRAM, cache counters, the SGLang DeepStack consumption fix, and
live parity. The remaining work is performance-oriented: replace JSON tensor
transport, scale CPU replicas on a CPU-rich allocation, and validate the
currently unsupported multimodal/parallelism variants.
