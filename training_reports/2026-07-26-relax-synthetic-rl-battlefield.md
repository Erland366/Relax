# Experiment: Relax Synthetic RL Battlefield

- **Date**: 2026-07-26
- **Author**: Codex-assisted
- **Goal**: Establish a trustworthy progression of synthetic text and vision
  tasks that proves reward routing, GRPO learning, multimodal conditioning,
  Megatron actor optimization, SGLang weight synchronization, and fully
  asynchronous execution on the ROCm Relax stack before moving to harder
  research tasks.
- **General description**: Relax can now learn both controlled text objectives
  and a genuinely image-conditioned XOR policy. The main unresolved problem is
  no longer basic learnability; it is preserving and interpreting the best
  policy under fully asynchronous training.
- **Models**:
  - Random-initialized, reduced Qwen3 model with tied embeddings for the joint
    EOS and two-action bandit task.
  - Random-initialized Qwen3-VL 0.37B model with 371,438,976 parameters, 12
    language layers, and a six-layer vision tower.
- **Datasets**:
  - Joint two-row EOS/two-action-bandit prompt set.
  - Visual-XOR discovery data with thousands of rendered images.
  - Visual-XOR refinement data with noisy 70%-correct SFT targets, 64 fixed RL
    training images, 128 held-out images, and permuted/constant controls.

---

## 1. What we tried

### 1.1 Text-only learning-signal ladder

1. Added a completion-length reward whose optimum is immediate EOS.
2. Replaced an overly deterministic initializer with a random-initialized,
   tied-embedding SFT model that deliberately assigns probability to both
   valid responses.
3. Added a two-action bandit where `A + EOS` receives `+1`, `B + EOS`
   receives `0`, and invalid responses receive `-1`.
4. Combined EOS and bandit rows in one Relax run, routed reward by
   `metadata.task`, and logged task-specific metrics separately.
5. Ran the joint task synchronously and fully asynchronously without saving
   Relax checkpoints.

### 1.2 Startup profiling

1. Timed the original in-framework EOS workload from Ray setup through actor
   step 0.
2. Launched the same small checkpoint with standalone SGLang using both
   defaults and Relax-compatible ROCm flags.
3. Located an existing vLLM environment without installing or upgrading
   packages and ran a compatible standalone comparison.
4. Separated model weight-loading time from imports, process startup, engine
   initialization, warmup, Ray Serve orchestration, and Megatron startup.

### 1.3 Multimodal visual-XOR ladder

1. Generated a large, variable visual-XOR dataset with opaque row identifiers
   and strict private labels.
2. Trained a compact Qwen3-VL model from random initialization. Only pinned
   processor/tokenizer metadata came from Qwen3-VL; no pretrained language,
   vision, projection, embedding, or output-head weights were loaded.
3. Added a discovery SFT that teaches the A/B protocol and a deliberately
   easier refinement SFT with 70%-correct XOR targets.
4. Added held-out, permuted-image, constant-image, and glyph-flip
   counterfactual gates so a global action bias cannot be mistaken for visual
   learning.
5. Integrated Qwen3-VL into the ROCm Megatron path, including a non-Transformer
   Engine vision fallback and explicit block-diagonal visual attention masks.
6. Ran refinement synchronously on two GPUs and fully asynchronously on four
   GPUs with DP2 actor training and a separate actor-forward service.

### 1.4 Operational debugging

1. Diagnosed stale Ray WorkerID messages and actor-death hangs.
2. Distinguished an actor failure from the downstream rollout staleness wait.
3. Verified that DP2 is viable after a clean launch; an earlier step-0 RCCL
   OOM was not the model's normal DP2 memory requirement.
4. Diagnosed cross-project GPU collisions caused by a long-lived default tmux
   server inheriting an older Slurm cgroup.

---

## 2. Setup and accepted initializers

### 2.1 Joint EOS and two-action bandit SFT

The 128-example tied-embedding SFT dataset is exactly balanced:

```text
32 EOS prompt    -> EOS
32 EOS prompt    -> OK + EOS
32 bandit prompt -> A + EOS
32 bandit prompt -> B + EOS
```

The accepted 100-step checkpoint produced:

| Gate | Result |
|---|---:|
| EOS prompt: immediate EOS rate | 47.3% |
| EOS prompt: `OK` rate | 50.4% |
| EOS/OK exact response validity | 98.4% |
| Bandit prompt: A rate | 51.2% |
| Bandit prompt: B rate | 46.1% |
| A/B exact response validity | 97.3% |
| Mixed eight-sample groups | 100% for both tasks |

This initializer supplies within-prompt reward variance without making either
RL objective already solved.

### 2.2 Visual-XOR refinement SFT

The refinement SFT is continued from the accepted local random-initialized
visual bootstrap. It uses 8,000 noisy training examples and 1,000 noisy
evaluation examples, with each XOR combination exactly 70% correct and 30%
flipped.

The accepted checkpoint produced:

| Gate | Result |
|---|---:|
| Held-out XOR sampled accuracy | 68.16% |
| Held-out valid response rate | 100% |
| Held-out action A/B rates | 48.44% / 51.56% |
| Held-out combination accuracies `00/01/10/11` | 60.94% / 65.23% / 74.22% / 72.27% |
| Permuted-image control accuracy | 49.80% |
| Constant-image control accuracy | 50.10% |
| Left-glyph flip directional probability shift | 0.3601 |
| Right-glyph flip directional probability shift | 0.3628 |
| Truncation rate | 0% |

The controls show that the initializer uses image content but does not already
solve the held-out task.

---

## 3. Runs and results

### 3.1 Run table

| Run | Mode and topology | Key configuration | Result |
|---|---|---|---|
| `raysubmit_Wq9zeKxuivPyZMYf` | Fully async joint text; actor 1 GPU, rollout 1 GPU, advantages CPU | 200 rollout cycles, batch `2 x 8 = 16`, GBS 16, LR `1e-6`, staleness 0, TIS | Succeeded and solved both tasks |
| `raysubmit_B8GMLKtLuAku4yFn` | Sync visual refinement; actor 1 GPU, rollout 1 GPU | RBS 4, `n=8`, GBS 32, one optimizer step/cycle, LR `3e-6`, eval every 8 | Reached 94.92% held-out at step 231; run was interrupted before the requested 500 cycles |
| `raysubmit_LTvz6LeADDCf9R4P` | Fully async visual refinement; DP2 actor, rollout 1 GPU, actor-forward 1 GPU | RBS 8, `n=8`, GBS 32, two optimizer steps/cycle, staleness 1 | Actor rank 0 OOMed in first RCCL reduce-scatter; top-level job remained stale-live |
| `raysubmit_2AzBgsgvzg9KbxAT` | Fully async visual refinement; same DP2 topology | Same batch geometry, LR `3e-6`, staleness 1 | Reached 93.55% held-out at step 91; interrupted after step 163 |
| `raysubmit_neq5BVaTTwaZtMYn` | Fully async visual refinement; same DP2 topology | 250 cycles, 500 optimizer steps, RBS 8, `n=8`, GBS 32, LR `3e-6`, staleness 4, TIS, weight update every cycle | Succeeded; crossed the full convergence gate near step 147, then regressed late |

All synthetic RL launchers kept Relax checkpoint saving disabled. W&B and
ordinary logs remained enabled.

### 3.2 Joint text multitask result

The successful 200-cycle fully-async run changed:

| Metric | Step 0 | Step 199 |
|---|---:|---:|
| EOS reward | -1.375 | -1.000 |
| EOS response length | 1.375 | 1.000 |
| Immediate EOS rate | 62.5% | 100% |
| Bandit score | 0.500 | 1.000 |
| Action A rate | 50.0% | 100% |
| Action B rate | 50.0% | 0% |

This is the strongest text-only result: one shared policy learned two
independently routed reward functions in the same fully-async run.

The aggregate mean reward is not a valid interpretation metric here. The EOS
optimum is `-1`, while the bandit optimum is `+1`; their average can remain
near zero even when both tasks improve. Task-specific metrics are mandatory.

### 3.3 Synchronous visual refinement

The two-GPU synchronous run changed held-out reward from `0.6582` at step 0 to
`0.9492` at step 231:

| Metric at step 231 | Result |
|---|---:|
| Held-out reward | 0.9492 |
| Valid response rate | 1.0000 |
| Combination `00` accuracy | 1.0000 |
| Combination `01` accuracy | 0.8672 |
| Combination `10` accuracy | 0.9375 |
| Combination `11` accuracy | 0.9922 |
| Permuted-image control | 0.5234 |
| Constant-image control | 0.5000 |

This exceeds the benchmark convergence criteria and keeps both anti-shortcut
controls near chance. It is direct evidence that Relax improved a genuinely
image-conditioned policy rather than learning a constant A/B response.

### 3.4 Fully asynchronous visual refinement

The final four-GPU run used:

```bash
RESOURCE_JSON='{"actor":[1,2],"rollout":[1,1],"actor_fwd":[1,1],"advantages":[1,0]}'
NUM_ROLLOUT=250
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
NUM_STEPS_PER_ROLLOUT=2
UPDATE_WEIGHTS_INTERVAL=1
MAX_STALENESS=4
LR=3e-6
ENABLE_RECOMPUTE=1
```

Each cycle generated 64 samples and performed two 32-sample optimizer steps.
With DP2, each optimizer step used 16 microbatches per actor rank. The job
completed 250 actor service cycles and 500 optimizer steps.

Key evaluations:

| Eval step | Held-out | Valid | Min combination | Permuted | Constant | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.6816 | 1.0000 | 0.6250 | 0.5234 | 0.5000 | Accepted SFT baseline |
| 143 | 0.9414 | 0.9922 | 0.8438 | 0.5156 | 0.5000 | Best mean, one combination slightly below the 0.85 gate |
| 147 | 0.9355 | 0.9902 | 0.8516 | 0.5156 | 0.5000 | Full benchmark convergence gate passed |
| 247 | 0.8652 | 0.9648 | 0.8047 | 0.5078 | 0.5000 | Late regression despite successful job completion |

The run therefore proves that fully asynchronous DP2 visual RL learns. It also
shows that the last policy is not necessarily the best policy. The scientific
problem has shifted from basic plumbing to asynchronous optimization
stability, policy-version lag, and model selection.

### 3.5 Why rollout reward sometimes drops sharply

Individual rollout points are sampled estimates from a small response set.
Even after the underlying policy improves, stochastic generation can produce
a temporarily poor batch. In fully-async mode, boundedly stale rollout policy
versions add another source of variance. A single rollout drop does not imply
that SGLang reset its weights.

For this task, the source of truth is the held-out evaluation trajectory plus
the four combination accuracies and controls. The final fully-async result
also proves that smoothing alone is insufficient: evaluation genuinely
regressed after the peak, so long-run drift must be treated as a real training
effect.

---

## 4. Startup findings

The original end-to-end workload timeline was:

```text
+00:00  Ray runtime setup began
+00:51  Ray initialized
+01:23  Relax began creating services
+04:05  SGLang finished loading and became ready
+04:09  rollout service became ready
+04:12  Megatron actor deployment began
+06:34  actor became ready
+06:40  training step 0 started
```

Standalone comparison on the same small tied SFT checkpoint:

| Engine/path | Approximate observed startup | Important qualification |
|---|---:|---|
| SGLang with Relax-compatible ROCm flags and local kernel path | 51 seconds from logged server arguments to first successful generation | Weight loading itself took only 1.46 seconds |
| vLLM eager with a compatible tokenizer | 34 seconds from first logged startup warning to application readiness | Different existing environment; no packages were installed |
| Full Relax workload | 400 seconds to actor step 0 | Includes Ray, Serve, rollout wiring, Megatron construction, and initial weight synchronization |

SGLang default flags were not a valid performance baseline on this ROCm stack:
the default AITER path failed with an undefined
`paged_attention_ragged`. Relax-compatible flags without the local kernel path
also failed because `sgl_kernel` was unavailable. The successful standalone
comparison therefore used the same conservative ROCm-compatible choices as
Relax.

The main startup lesson is that model weight loading is not the dominant cost.
Most time is in imports, worker/process creation, engine warmup, Ray Serve
orchestration, Megatron initialization, and service-to-service weight wiring.

---

## 5. What worked

1. **SFT acceptance gates before RL.** Short valid-response gates prevented
   invalid or over-deterministic initializers from wasting full Relax runs.
2. **Random initialization is viable.** Both reduced Qwen3 and Qwen3-VL models
   were trained without pretrained model weights.
3. **Per-task reward routing and logging.** `metadata.task` plus task-specific
   metrics made joint training interpretable.
4. **Strict multimodal controls.** Held-out, permuted, constant, per-combination,
   and counterfactual metrics separated genuine image use from action bias.
5. **Synchronous visual baseline.** It gives the clearest learning curve and is
   the correct control before interpreting async behavior.
6. **Fully-async DP2 execution.** A clean 250-cycle/500-update run completed,
   synchronized weights, evaluated controls, and reached the convergence gate.
7. **No debug checkpoint accumulation.** Relax checkpoints remained disabled
   while W&B and logs provided observability.

---

## 6. What failed and why

| Failure | Cause | Lesson |
|---|---|---|
| Original EOS initializer already strongly preferred immediate EOS | The SFT objective nearly solved the reward before RL | An RL smoke initializer must be valid but deliberately non-optimal |
| First visual discovery attempts stayed noisy/static | Discovering XOR from a weak visual bootstrap was a harder problem than validating the RL stack | Use refinement first to separate framework learning from task discovery |
| Aggregate joint reward obscured progress | EOS and bandit have different reward scales and optima | Log and judge each task independently |
| `--balance-data` in pure fully-async mode | Relax explicitly rejects this combination | Preserve deterministic composition through the dataset and batch geometry instead |
| DP2 actor OOMed during the first RCCL reduce-scatter | GPU-wide usage was 62.44 GiB while the failing rank reported only 2.77 GiB allocated; the device was already abnormally occupied | Do not conclude DP2 is intrinsically too large; inspect global GPU owners and retry only from a clean allocation |
| Actor failure became an apparent rollout deadlock | `train_0` was never cleared, `MAX_STALENESS` backpressured rollout, and the top-level job stayed alive | The first actor exception is causal; repeated rollout wait warnings are downstream |
| Different request IDs still collided on GPUs | A shared default tmux server was rooted in Slurm job `94808`; panes for another allocation inherited that cgroup and GPU mask | Use one tmux server socket per Slurm job |
| Fully-async policy regressed after reaching the benchmark gate | Continued updates under bounded staleness and fixed LR moved away from the best evaluated policy | Successful job completion and best-policy quality are different outcomes |
| No checkpoint represented the best async policy | Checkpointing was intentionally disabled for debugging | Keep it disabled for plumbing smokes, but use a bounded best-eval artifact for optimization studies |

---

## 7. Operational battlefield rules

### 7.1 Before every GPU run

```bash
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS}"
cat /proc/$$/cgroup

TMUX_SERVER_PID="$(tmux display-message -p '#{pid}')"
cat "/proc/${TMUX_SERVER_PID}/cgroup"

fuser -v /dev/kfd
rocm-smi --showpids --showmemuse --showuse
```

The shell and tmux server must belong to the same Slurm job. Start isolated
tmux servers with:

```bash
tmux -L "slurm-${SLURM_JOB_ID}" new -s main
```

### 7.2 Minimal validation ladder

1. Reward unit tests and dataset preflight.
2. SFT acceptance evaluation.
3. Two-rollout optimizer and weight-sync smoke.
4. Synchronous learning baseline with held-out metrics.
5. Fully-async run with actor-forward, explicit staleness, and TIS.
6. Only then expand model size, task difficulty, or runtime length.

### 7.3 Metrics that define visual success

Never accept training reward alone. Require:

```text
held-out reward >= 0.90
all four held-out combinations >= 0.85
valid response rate >= 0.99
both actions remain represented
permuted control near 0.50
constant control near 0.50
```

---

## 8. Open questions for the hard-work phase

1. **Async stability:** Which factor caused the drop after step 147:
   `MAX_STALENESS=4`, two optimizer steps per rollout, constant `3e-6` LR,
   actor-forward mismatch, or their interaction?
2. **Model selection:** What is the smallest bounded checkpoint policy that
   captures only eval-best state without returning to high-volume debug
   checkpointing?
3. **Staleness observability:** Log generated-policy version, actor update
   version, accepted sample age, TIS statistics, and eval policy version in one
   aligned table.
4. **Controlled ablations:** Compare staleness `0/1/2/4`,
   `NUM_STEPS_PER_ROLLOUT=1/2`, and LR `1e-6/3e-6` while holding the SFT,
   prompt order, seeds, and evaluation set fixed.
5. **Discovery tier:** After the refinement path is stable, can RL discover
   XOR from the weaker visual bootstrap without noisy XOR SFT labels?
6. **Failure propagation:** Make an actor fatal exception fail or restart the
   whole job instead of leaving rollout in a stale-live wait.
7. **Startup:** Reduce the Ray/Serve/Megatron orchestration component now that
   standalone engine startup is known not to explain the full 6:40 delay.
8. **Allocation isolation:** Add a launcher preflight that rejects a tmux
   server from another Slurm cgroup and reports unexpected `/dev/kfd` owners.

---

## 9. Candidate reusable knowledge

Prefer updating existing knowledge rather than growing the default skill
profile:

1. Extend `ray-stale-live-state-triage` with the observed chain
   `actor failure -> uncleared train_N -> MAX_STALENESS backpressure`.
2. Add a troubleshooting entry for default tmux servers crossing Slurm
   allocations and inheriting the wrong GPU cgroup.
3. Add the visual-XOR refinement recipe and sync/fully-async convergence
   thresholds to a result skill only if this benchmark becomes a standing
   regression rather than a one-off research task.

Approved follow-up completed on 2026-07-27:

- Extended `ray-stale-live-state-triage` with partial distributed-rank failure,
  uncleared partition, bounded-staleness backpressure, and global GPU ownership
  checks.
- Added the tmux/Slurm GPU-cgroup failure pattern to
  `references/troubleshooting.md`.
- Deliberately did not create a visual-XOR result skill; the benchmark remains
  documented here until it becomes a standing regression gate.

---

## 10. References

- Joint task guide: `examples/eos_two_action_bandit/README.md`
- Visual task guide: `examples/visual_xor/README.md`
- Joint fully-async log:
  `log/eos-two-action-bandit-20260716_092253.log`
- Visual refinement SFT acceptance:
  `log/visual-xor-refinement-sft-acceptance-20260720.log`
- Best synchronous visual run:
  `log/visual-xor-refinement-20260722_085536.log`
- First DP2 stale-live failure:
  `log/visual-xor-refinement-20260722_105748.log`
- Successful fully-async DP2 run:
  `log/visual-xor-refinement-20260722_174431.log`
- Startup artifacts: `benchmark_results/startup_profile/`
- Startup architecture note:
  `resources/modules/actor-rollout-startup-debugging.md`
