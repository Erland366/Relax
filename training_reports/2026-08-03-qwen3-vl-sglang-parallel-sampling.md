# Qwen3-VL SGLang Parallel Sampling

- **Date:** 2026-08-03
- **Status:** passed standalone parity and a fully asynchronous two-rollout
  CPU-skipped training run
- **Hardware:** four AMD MI210 GPUs; one Ray CPU reserved for one CPU vision
  replica
- **Checkpoint:**
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- **Workload:** baseline evaluation, two 64-sample rollouts, four optimizer
  updates, final rollout and actor-forward synchronization, and no checkpoint
  saving

## Objective

The corrected 2026-08-01 comparison showed that one 524,312-byte CPU vision
feature became an approximately 2.907 MB JSON generation request. Relax sent
the same feature eight times for `N_SAMPLES_PER_PROMPT=8`, making downstream
transport—not CPU encoding—the first performance boundary.

Test the smallest optimization before introducing an SGLang feature registry:
use SGLang SGLang multi-sample sampling to send one precomputed final-plus-
DeepStack bundle per prompt with `sampling_params.n=8`, then map the ordered
outputs back to the original Relax samples.

## Implementation and safety boundary

The rollout engine groups samples only when all of these conditions hold:

- the group contains multiple fresh, pending samples;
- every sample shares the same prompt and media object;
- CPU-precomputed Qwen3-VL features are active;
- inference is stochastic;
- generation uses Relax's built-in path, without Slime middleware, routing
  replay, partial output, or resumed generation; and
- routing is round-robin, or exactly one SGLang engine exists so no
  cross-engine choice remains.

One request carries one final feature stream, all numbered DeepStack streams,
and `sampling_params.n=<group size>`. Relax requires a response list with the
exact expected cardinality and maps it in order. Request construction, timing,
and body sizes are counted once per request; response tokens, log-probabilities,
finish metadata, reward, and Megatron visual inputs remain per sample.

Unsupported groups retain the scalar path and log one explicit reason.
Deterministic generation remains scalar because this SGLang interface does
not expose a distinct seed for each SGLang multi-sample branch. Multi-engine,
non-round-robin routing also remains scalar because grouping would erase the
existing per-sample routing decision.

## Standalone live parity gate

Artifact:
`benchmark_results/cpu_vision/20260803_sglang_parallel_n8/probe.json`

One live SGLang request produced eight ordered outputs from one precomputed
feature bundle. It was compared with eight scalar-equivalent requests using
the same fixed image and greedy scores.

| Gate | Result |
|---|---:|
| Ordered output match | 8 / 8 |
| Text match | 8 / 8 |
| Maximum A/B log-probability delta | `2.980232238769531e-07` |
| Maximum action-margin delta | `0.0` |
| Maximum action-probability delta | `0.0` |
| One `n=8` request | 3,179,557 B |
| Eight scalar-equivalent requests | 25,436,392 B |
| Request-body reduction | 87.50% |

The gate passed and the standalone SGLang server shut down cleanly.

## Relax integration correction

The first full attempt, Ray job `raysubmit_GYunVFxDkBBayez9` and W&B run
`bl753v4s`, exposed an eligibility mistake rather than a training failure.
Relax's router used `cache_aware`, so the initial round-robin-only guard kept
all rollout requests scalar. Rollout 0 completed with 64 SGLang requests,
185,397,880 request bytes, and 21.512 seconds wall time. The run was stopped
before rollout 1 and is excluded from the accepted benchmark.

A failing test was added before changing the guard. The fix permits any
router policy only when the resource configuration proves that exactly one
SGLang engine exists. The multi-engine non-round-robin restriction remains.

## Accepted two-rollout run

| Item | Value |
|---|---|
| Ray job | `raysubmit_GLUEF5ZUBeq5S13S` |
| W&B | `lr67sp9v` |
| Log | `log/visual-xor-refinement-cpu-skip-gpu-encoder-20260803_174541.log` |
| VRAM artifact | `benchmark_results/cpu_vision/20260803_sglang_parallel_n8/cpu_skipped_two_rollout_vram.json` |
| Exit code | 0 |
| Monitor wall time | 771.621 s |

Timeline:

- `+00:00` — monitor started at 17:45:40 UTC;
- `+05:39` — SGLang became ready;
- `+05:44` — rollout service became ready;
- `+06:02` — initial rollout weight synchronization completed;
- `+10:52` — baseline evaluation completed and rollout 0 started;
- `+10:58` — rollout 0 completed;
- `+11:04` — rollout 1 completed;
- `+11:16` to `+11:46` — four optimizer updates completed successfully;
- `+11:57` — final weight synchronization completed;
- `+11:58` — controller shutdown completed; and
- `+12:52` — the monitor wrote the final idle-memory artifact.

The baseline evaluation remained scalar because evaluation dispatch was still
coupled to `group_rm`, which was disabled. The held-out dataset actually
requested four samples for each of 128 prompts; the two deterministic controls
requested one sample for each of 128 prompts. It completed 768 requests,
scored `0.703125` held-out reward with valid-action rate `1.0`, and sent
2,232,651,920 request bytes. Its CPU LRU recorded 128 encodes and 640 hits.
Transient router send warnings recovered through the normal SGLang router
path; client attempt counts remained one and all evaluation samples completed.

### Grouped rollout metrics

| Metric | Rollout 0 | Rollout 1 | Mean |
|---|---:|---:|---:|
| Generated samples | 64 | 64 | 64 |
| SGLang generation requests | 8 | 8 | 8 |
| Parallel samples per request | 8 | 8 | 8 |
| Request-body bytes | 23,169,839 | 23,323,962 | 23,246,900.5 |
| Rollout time time | 6.069 s | 5.562 s | 5.816 s |
| Reward | 0.7500 | 0.6875 | 0.71875 |
| Valid action | 1.0 | 1.0 | 1.0 |
| Action A | 0.46875 | 0.4375 | 0.453125 |
| CPU encode requests | 8 | 8 | 8 |
| CPU encode time | 1.488 s | 1.985 s | 1.737 s |

Compared with the rejected same-day scalar rollout 0:

| Metric | Scalar | Grouped mean | Change |
|---|---:|---:|---:|
| SGLang requests | 64 | 8 | 87.50% fewer |
| Request-body bytes | 185,397,880 | 23,246,900.5 | 87.46% lower |
| Rollout time time | 21.512 s | 5.816 s | 72.97% lower; 3.699x faster |

The comparison isolates the within-prompt `n=8` amplification on the same
day and execution path. Reward is stochastic task behavior, not a transport
performance gate.

### Training correctness

Both rollouts reached actor-forward and the actor. All four optimizer updates
reported `update_successful=True`; gradient norms were `3.8432`, `0.9394`,
`1.1531`, and `1.8288`. Actor-forward computed both steps, the final rollout
and actor-forward synchronization markers were present, and the controller
shut down cleanly. No checkpoint was written because `SAVE_CHECKPOINTS=0`.

## VRAM comparison

Peak deltas above the common idle baseline:

| Run | Card 0 actor | Card 1 actor | Card 2 rollout | Card 3 actor-fwd | Simultaneous total |
|---|---:|---:|---:|---:|---:|
| 2026-08-01 CPU skipped scalar | 2.840 GiB | 2.418 GiB | 7.215 GiB | 7.173 GiB | 19.279 GiB |
| 2026-08-03 CPU skipped grouped | 2.838 GiB | 2.418 GiB | 7.215 GiB | 7.173 GiB | 18.786 GiB |

The per-device peaks are effectively unchanged, as expected: grouping removes
repeated transport but does not omit additional model weights. The 0.493 GiB
lower simultaneous peak is schedule-sensitive fully asynchronous overlap and
must not be reported as a memory saving.

The complete monitor duration was 26.3% shorter than the 2026-08-01 skipped
run, but baseline evaluation also ran faster because of runtime variance.
Therefore the direct same-day scalar-versus-grouped rollout measurements—not
whole-run duration—support the 3.699x speedup claim.

## Decision

SGLang multi-sample sampling closes the repeated `N_SAMPLES_PER_PROMPT=8`
transport amplification for this training workload. Defer a general SGLang
feature registry and do not scale CPU vision replicas yet.

The remaining large boundary is cross-request reuse, especially scalar
evaluation: the accepted run still sent 2.233 GB across 768 evaluation
requests despite a working CPU LRU. The next experiment should target
evaluation-side feature fanout or a bounded binary/engine-local registry only
for repeated distinct requests. It must preserve explicit feature revision,
routing, eviction, and missing-ID failures. Multi-engine routing must be
tested separately before widening grouped eligibility.

Follow-up on 2026-08-04: evaluation dispatch is now independent of `group_rm`,
so the stochastic held-out dataset uses 128 `n=4` requests while deterministic
controls remain scalar. Stateless packed BF16 transport then reduced the
grouped evaluation from 1.117 GB to 268.9 MB and each 64-sample rollout to
5.60 MB. This supersedes the recommendation to jump from evaluation fanout
directly to a registry. See
`training_reports/2026-08-04-qwen3-vl-packed-cpu-vision-transport.md`.

## Validation

```text
23 passed in 3.24s
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

Focused tests cover ordered mapping, malformed cardinality, skip reasons,
single-engine cache-aware routing, live parity request accounting, and the
shared HTTP metrics path.
