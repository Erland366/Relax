# Qwen3-VL Three-Mode Vision VRAM and Cache Comparison

- **Date:** 2026-07-31
- **Status:** GPU-weight omission passed end to end; performance claims remain
  blocked on live native-versus-precomputed semantic parity
- **Hardware:** four AMD MI210 GPUs; two allocated physical CPU cores
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **Workload:** baseline evaluation, two fully asynchronous rollouts, four
  optimizer updates, and final weight synchronization
- **Checkpoint saving:** disabled in every mode

## Objective

Compare the same visual-XOR Relax workload in three modes:

1. native GPU vision with the visual tower and projector frozen;
2. CPU precomputed vision while GPU visual weights remain resident; and
3. CPU precomputed vision with GPU visual weights omitted.

The comparison answers three separate questions:

- does the omitted-weight path still complete rollout and training correctly;
- how much real per-device and simultaneous VRAM does omission save; and
- does the CPU cache avoid repeated encoding under the real evaluation and
  rollout workload?

It does not yet establish that the CPU path is a behaviorally equivalent or
faster replacement for native GPU vision.

## Equalized configuration

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
SAVE_CHECKPOINTS                0
```

The native launcher also passed `--freeze-vision-model` and
`--freeze-vision-projection`. Without that control, visual gradients and
optimizer state would make the actor-side comparison invalid. Both CPU modes
used one CPU vision replica with one reserved Ray CPU, a 1 GiB LRU, and a
configured maximum request batch size of eight.

Each mode started from the same idle baseline of `13,135,872` bytes on every
card. `monitor_rocm_vram.py` polled `rocm-smi` through the complete launcher
lifetime and recorded per-card maxima as well as the maximum simultaneous
four-card total.

## Run inventory

| Mode | Ray job | W&B | Workload status | Monitor status |
|---|---|---|---|---:|
| Native GPU | `raysubmit_4TE3SETiNUQZGarm` | `we20dh4y` | Passed | `0` |
| CPU resident | `raysubmit_fZDv79ENHUtecm2g` | `qioqj49h` | Passed | `127` anomaly |
| CPU omitted | `raysubmit_xkETPSVC1jMVYcG8` | `uusr01l5` | Passed | `0` |

The CPU-resident log explicitly reaches all four optimizer updates,
`All training steps finished`, `Main func successfully`, clean SGLang and Ray
shutdown, and Ray's final `Job ... succeeded`. The monitor nevertheless
observed exit status `127` from the outer `env ... bash <launcher>` process.
There is no corresponding command-not-found or workload failure in the Ray
log. The JSON artifact is preserved unchanged and this result is treated as a
successful workload with an unresolved wrapper-status anomaly, not as a clean
launcher exit.

## VRAM results

Values below are peak deltas over the idle baseline. Per-card peaks need not
occur in the same sample, so their sum is not the simultaneous total.

| Mode | Card 0 actor | Card 1 actor | Card 2 rollout | Card 3 actor-fwd | Simultaneous total |
|---|---:|---:|---:|---:|---:|
| Native GPU | 2.933 GiB | 2.511 GiB | 7.778 GiB | 7.257 GiB | 20.144 GiB |
| CPU resident | 2.896 GiB | 2.443 GiB | 7.278 GiB | 7.228 GiB | 19.090 GiB |
| CPU omitted | 2.837 GiB | 2.418 GiB | 7.215 GiB | 7.173 GiB | 18.938 GiB |

### Clean omission effect

CPU omitted versus CPU resident isolates the omission knob while retaining
the same precomputed-feature path:

| Device/aggregate | Reduction |
|---|---:|
| Card 0 actor | 60.54 MiB |
| Card 1 actor | 26.18 MiB |
| Card 2 rollout | 64.02 MiB |
| Card 3 actor-fwd | 56.02 MiB |
| Sum of independent per-card peak reductions | 206.76 MiB |
| Simultaneous four-card peak reduction | 155.73 MiB |

The simultaneous reduction is the defensible cluster-level answer for this
workload. The independent peak sum is also useful for locating memory on each
role, but it must not be presented as a simultaneous saving.

CPU omitted was 1.206 GiB below native GPU vision at the simultaneous peak.
Only 0.152 GiB of that difference is isolated by the resident-versus-omitted
comparison. The remainder includes changes in visual execution, temporary
buffers, allocation timing, and serving behavior, so it cannot be labeled
"omitted visual weights."

All four cards returned exactly to their starting VRAM values after every
run.

## Runtime results

| Mode | Launcher wall time | Relative to native | Services ready to baseline eval complete |
|---|---:|---:|---:|
| Native GPU | 658.8 s | 1.000x | 138 s |
| CPU resident | 1006.9 s | 1.528x | 436 s |
| CPU omitted | 1011.5 s | 1.535x | 438 s |

Omission itself had no measurable speed benefit: resident and omitted were
within 4.6 seconds across approximately 17 minutes. Both CPU modes were about
53% slower end to end than native GPU vision on this allocation.

This is a deliberately CPU-starved correctness run, not a CPU scaling result.
The allocation exposed only two physical CPU cores and the vision service used
one. Even so, the CPU backend reported only 52-54 seconds of actual encode
time during the 768-request baseline evaluation. The much larger end-to-end
gap therefore cannot be assigned only to visual computation; current JSON
feature serialization, Ray/Serve transport, request scheduling, SGLang
consumption, and generation are part of the measured path.

## Cache counters

The rollout manager now snapshots `VisionEncoder.get_metrics()` at evaluation
and after every rollout. The same values are included in normal Relax metric
batches, so they appear in both the log and W&B.

### Baseline evaluation

| Metric | CPU resident | CPU omitted |
|---|---:|---:|
| Requests | 768 | 768 |
| Cache hits | 640 | 640 |
| Cache misses | 128 | 128 |
| Hit rate | 83.33% | 83.33% |
| Actual backend encode requests/images | 128 | 128 |
| Evictions | 0 | 0 |
| Entries after evaluation | 128 | 128 |
| Resident feature bytes | 67,111,936 | 67,111,936 |
| Backend encode time | 54.03 s | 52.60 s |
| Backend images/second | 2.369 | 2.434 |

The evaluation contains 128 unique images and 768 total requests. These
counters prove that each unique image was encoded once by the one-replica CPU
service and the remaining 640 requests were cache hits. No LRU pressure was
observed.

Each later rollout introduced eight new images. Consequently, both rollout
snapshots reported eight requests, eight misses, eight encodes, zero hits, and
zero evictions. The cache grew to 144 entries and 75,500,928 resident feature
bytes after the second rollout. This is expected reuse behavior for the
present data order, not evidence that the rollout cache is broken.

## Behavioral results

The baseline evaluation is also a live-system correctness signal:

| Mode | Held-out reward | Valid action | Held-out action A | Permuted control | Constant control |
|---|---:|---:|---:|---:|---:|
| Native GPU | 0.6816 | 1.0 | 0.4980 | 0.5234 | 0.5000 |
| CPU resident | 0.5039 | 1.0 | 0.6875 | 0.5000 | 0.5000 |
| CPU omitted | 0.5000 | 1.0 | 0.7148 | 0.5000 | 0.5000 |

CPU resident and CPU omitted agree closely: both retain an action-A bias and
score at chance. That makes GPU-weight omission an unlikely cause of the
behavioral difference. Native GPU vision is balanced between A and B and is
materially above chance on the same held-out evaluation.

The independent runs use stochastic sampling, so this table alone does not
locate the mismatch. The gap is nevertheless too large to support a
performance-equivalence claim. Earlier local Hugging Face feature and logit
parity passed, which narrows the next gate to the live serving/training
integration rather than the standalone CPU encoder.

## Conclusions

1. **GPU-weight omission works end to end.** The omitted mode completed all
   evaluation samples, both rollouts, actor-forward work, four optimizer
   updates, final synchronization, and shutdown without checkpoints.
2. **The cache instrumentation works.** W&B and logs now expose total and
   interval requests, hits, misses, evictions, entries, resident bytes,
   backend work, feature bytes, backend time, hit rate, and throughput.
3. **The measured omission saving is real but modest.** The clean comparison
   is 155.73 MiB at the simultaneous four-GPU peak, not the full 1.206 GiB
   native-to-omitted difference.
4. **This CPU path is currently slower on the tested allocation.** It is about
   1.53x native wall time, and omission does not improve throughput.
5. **Semantic parity is the P0 blocker.** Before CPU scaling, binary transport,
   more replicas, or throughput claims, compare native-image and precomputed-
   feature logits inside the live SGLang transformers backend with identical
   checkpoint revision, processor-expanded prompt IDs, position IDs, feature
   bundle, sampling parameters, and policy version. Repeat the same fixed-
   input gate at the Megatron actor-forward boundary.

## Evidence

### Logs

- Native GPU: `log/visual-xor-refinement-native-gpu-20260731_100025.log`
- CPU resident: `log/visual-xor-refinement-cpu-resident-20260731_101159.log`
- CPU omitted: `log/visual-xor-refinement-cpu-omitted-20260731_093951.log`

### VRAM artifacts

- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/native_gpu_vram.json`
- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/cpu_resident_vram.json`
- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/cpu_omitted_vram.json`

### W&B

- Native GPU: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/we20dh4y`
- CPU resident: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/qioqj49h`
- CPU omitted: `https://wandb.ai/erlandpg/relax-amd-visual-xor-refinement/runs/uusr01l5`
