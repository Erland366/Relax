---
name: completion-length-reward-smoke-test
description: >
  Validate Relax reward plumbing and GRPO learning signal with a toy completion
  length reward. Use when testing whether reward metrics, W&B logging, rollout,
  advantage computation, actor training, checkpointing, and sync versus
  fully-async execution work before debugging a real task.
metadata:
  short-description: "Toy EOS reward e2e validation"
  tags:
    - relax
    - reward
    - grpo
    - rocm
    - wandb
    - transfer-queue
  domain: research
  created: 2026-06-20
  author: Codex
---

# Completion Length Reward Smoke Test

## General Description

This skill captures the toy reward task where reward is
`-len(completion_tokens)`. The optimal policy emits EOS immediately, so this
task is useful for separating reward plumbing and training-loop behavior from
real task complexity.

The validated Qwen3-0.6B ROCm runs showed that the Relax stack can execute this
reward end-to-end and log namespaced reward metrics to W&B. They also showed
two distinct issues: fully async can fail from TransferQueue backlog, and the
current prompt/sampling setup often produces no reward variance because the
model generates to `rollout_max_response_len`.

## When to Apply

Use this knowledge when:
- Adding or changing simple scalar reward functions.
- Checking whether rollout reward metrics reach W&B as `rollout/reward/*`.
- Verifying GRPO advantage computation with a known reward surface.
- Comparing synchronous, hybrid, and fully async Relax execution modes.
- Debugging fully async TransferQueue capacity or producer/consumer behavior.

Do not use this as proof that a model learned the EOS policy unless the run
shows nonzero reward variance, nonzero advantages, and nonzero gradients.

## Results Summary

| Run | Mode | Result |
|-----|------|--------|
| `m9qfd2j5` | fully async, `max_len=32`, 4 rollouts | Completed, but reward was flat at `-32.0` |
| `palkq31u` | fully async, `max_staleness=1`, `max_len=64` | Failed with TransferQueue capacity `32 existing + 16 new > 32` |
| `z0m8zdv7` | fully async, `max_staleness=3`, `max_len=64` | Failed later with TransferQueue capacity `64 existing + 16 new > 64` |
| `lwrhk4rw` | fully async short6, `max_staleness=3` | Completed; produced occasional shorter completions |
| `ougxjccy` | sync/non-fully-async short6, `max_staleness=0` | Completed; no queue failure, but reward stayed flat at `-64.0` |
| `8a5r2uzv` | real Qwen3-4B sync colocate short6 | Failed before rollout; SGLang loaded weights, then hit ROCm `HIP error: named symbol not found` on first prefill |
| `t1i8zbrt` | real Qwen3-4B sync non-colocate short6 | Reached rollout and training; rollout reward flat at `-64.0`, then actor OOM at first `optimizer.step()` |
| `rplfx3eg` | Qwen3-0.6B sync colocate short6 | Reproduced the same SGLang ROCm `HIP error: named symbol not found` before rollout |
| `jidwd60n` | Qwen3-0.6B sync colocate, low SGLang pressure, `AMD_SERIALIZE_KERNEL=3` | Reproduced the same first-prefill HIP named-symbol error on both SGLang engines after server ready |
| `2usrz00j` | Qwen3-Mock-0.5B sync colocate mini baseline | Reproduced the same first-prefill HIP named-symbol error with `enable_memory_saver=True` |
| `kse80ff1` | Qwen3-Mock-0.5B sync colocate, `RELAX_SGLANG_DISABLE_MEMORY_SAVER=1` | Foreground validation reached SGLang weight/KV load with `enable_memory_saver=False`; no immediate named-symbol crash before the 5-minute cutoff |
| `tyhcen6i` | Qwen3-Mock-0.5B sync colocate, no SGLang memory saver | Completed rollout 0 with reward `-64.0`, then failed in actor log-prob forward with ROCm HIP named-symbol error |
| `oc9gqu3i` | Qwen3-Mock-0.5B sync non-colocate A/B | Succeeded one rollout, one actor train step, weight update, and checkpoint save |
| `frx2vwtb` | Qwen3-Mock-0.5B sync colocate, actor sleep-after-weight-update patch | Completed rollout 0, then failed in actor log-prob forward with the same HIP named-symbol error |
| `6dq4eeav` | Qwen3-Mock-0.5B sync colocate, sleep-after-weight-update patch, `AMD_SERIALIZE_KERNEL=3` | Completed rollout 0, then failed in actor log-prob forward; serialization did not improve stack attribution |
| `jeid5rq1` | real Qwen3-0.6B sync non-colocate sampled GRPO, `n=2`, `micro_batch=2`, `max_len=8` | Completed 10/10 and checkpointed, but reward stayed flat at `-8.0`; advantages and grad norm stayed zero |
| `wqqv6si0` | real Qwen3-0.6B sync non-colocate sampled GRPO validation, short prompt, `max_len=4`, `temperature=2.0` | Foreground validation reached SGLang ready and actor initialization before the 5-minute cutoff |
| `raysubmit_3xnDf2D9uMNdkPT4`, `raysubmit_AgeAN98TtC6tMu9M` | same short-prompt production config | Both production attempts hung before W&B startup, Ray service actors, or GPU allocation; root cause was a stale Torch extension lock for Megatron `managed_alloc_runtime` |
| `6edey3v7` | same short-prompt production config after removing stale Torch extension lock | Completed 10/10 rollout/train steps, but all sampled completions still hit `max_len=4`; reward stayed flat at `-4.0`, advantages and grad norm stayed zero |
| `0cp1qu3s` | real Qwen3-4B sync non-colocate TP2 sampled GRPO, short prompt, `max_len=4`, `temperature=2.0` | Completed 10/10 rollout/train steps and saved torch_dist checkpoint iteration 9; reward still stayed flat at `-4.0`, advantages and grad norm stayed zero |
| `4cz43iek` | real Qwen3-4B sync non-colocate TP2 sampled GRPO, 50 rollouts, short prompt, `max_len=4`, `temperature=2.0` | Completed 50/50 and saved torch_dist checkpoint iteration 49; rollout 40 produced one shorter completion (`mean_len=3.75`, `reward=-3.75`, `grad_norm=61.13`), but the signal did not persist and final rollout 49 returned to flat `-4.0` |

W&B links:

```text
baseline flat async: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/m9qfd2j5
async stale1 fail:   https://wandb.ai/erlandpg/relax-amd-completion-length/runs/palkq31u
async stale3 fail:   https://wandb.ai/erlandpg/relax-amd-completion-length/runs/z0m8zdv7
async short6 pass:   https://wandb.ai/erlandpg/relax-amd-completion-length/runs/lwrhk4rw
sync short6 pass:    https://wandb.ai/erlandpg/relax-amd-completion-length/runs/ougxjccy
4B collocate fail:   https://wandb.ai/erlandpg/relax-amd-completion-length/runs/8a5r2uzv
4B non-colocate oom: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/t1i8zbrt
0.6B colocate fail:  https://wandb.ai/erlandpg/relax-amd-completion-length/runs/rplfx3eg
0.6B colocate low-pressure/serialize fail:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/jidwd60n
mock colocate baseline fail:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/2usrz00j
mock colocate no-memory-saver probe:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/kse80ff1
mock colocate no-memory-saver rollout-then-actor fail:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/tyhcen6i
mock non-colocate A/B pass:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/oc9gqu3i
mock colocate sleep-after-weight-update fail:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/frx2vwtb
mock colocate serialize-kernel fail:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/6dq4eeav
sampled sync n2/micro2 pass:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/jeid5rq1
short-prompt max4/temp2 validation:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/wqqv6si0
short-prompt max4/temp2 post-lock run:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/6edey3v7
4B TP2 short-prompt max4/temp2 pass:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/0cp1qu3s
4B TP2 short-prompt max4/temp2 long50 pass:
                     https://wandb.ai/erlandpg/relax-amd-completion-length/runs/4cz43iek
```

## Recommended Practice

### Step 1: Use the toy completion-length reward

The reward function should return the negative response token count:

```python
reward = -float(sample.response_length)
```

A healthy rollout log includes:

```text
rollout/reward/mean
rollout/reward/median
rollout/reward/max
rollout/reward/min
rollout/response_len/mean
rollout/truncated_ratio
```

### Step 2: Start with synchronous/non-fully-async execution

Use sync first when the goal is validating reward plumbing without fully async
queue pressure. The validated short run used:

```bash
source /vast/users/qirong.ho/miniforge3/etc/profile.d/conda.sh
conda activate relaxrl_rocm_after_fix
unset ROCR_VISIBLE_DEVICES

export ASSET_DIR=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets
export PROMPT_SET=$PWD/examples/completion_length/prompts.jsonl
export RELAX_EXECUTION_MODE=sync
export USE_COLLOCATE=0
export RM_TYPE=completion_length
export REWARD_KEY=
export USE_KL_LOSS=0
export USE_BALANCE_DATA=0
export NO_SAVE_RNG=1
export WANDB_MODE=online
export WANDB_ANONYMOUS=allow
export WANDB_PROJECT=relax-amd-completion-length
export WANDB_GROUP=completion-length-sync-short6-YYYYMMDD
export WANDB_ENTITY=
export CONDA_ENV_NAME=relaxrl_rocm_after_fix
export NUM_ROLLOUT=6
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=8
export NUM_STEPS_PER_ROLLOUT=2
export MAX_STALENESS=0
export SGLANG_SERVER_CONCURRENCY=16
export SGLANG_MAX_RUNNING_REQUESTS=128
export SGLANG_MAX_TOTAL_TOKENS=65536
export SEQ_LENGTH=1024
export ROLLOUT_MAX_RESPONSE_LEN=64
export ROLLOUT_TEMPERATURE=1.2

bash scripts/training/multimodal/amd_qwen3_0_6b_4gpu_e2e.sh
```

Important nuance: `RELAX_EXECUTION_MODE=sync` removes `--fully-async`. Set
`USE_COLLOCATE=1` when the run should also add `--colocate`; otherwise parsed
args show `fully_async=False` and `colocate=False`. The non-colocate sync path
was still useful because it avoided the fully async TransferQueue backlog and
completed all rollout/training steps.

The colocate probes confirmed that both Megatron and SGLang can load the model
weights, but the colocated rollout path can crash before reward metrics.
Both real Qwen3-4B colocate and Qwen3-0.6B colocate reproduced the same SGLang
first-prefill failure:

```text
torch.AcceleratorError: HIP error: named symbol not found
```

The Qwen3-4B non-colocate control did not hit that SGLang failure. It reached
rollout 0 and actor training, then failed later at the first actor
`optimizer.step()` with HIP OOM. Treat the named-symbol error as a colocated
SGLang/ROCm runtime failure, not as evidence about model quality or reward
learning.

For colocated ROCm diagnostics, set `AMD_SERIALIZE_KERNEL=3` in the shell. The
AMD launcher forwards this variable into the Ray runtime env, and the rollout
engine forwards it into SGLang engine actors only when it is explicitly set.
In the `jidwd60n` run, this confirmed the env reached SGLang, but PyTorch also
logged:

```text
Ignoring invalid value for boolean flag AMD_SERIALIZE_KERNEL: 3valid values are 0 or 1.
```

So `AMD_SERIALIZE_KERNEL=3` did not produce a better stack on this stack. The
failure still landed at `schedule_batch.py:1510` during
`new_batch.prepare_for_extend()`.

For quick colocate debugging, use the mini mock model wrapper:

```bash
bash scripts/training/multimodal/amd_qwen3_mock_2gpu_e2e.sh
```

The Qwen3-Mock-0.5B baseline reproduced the same first-prefill
`HIP error: named symbol not found` with `enable_memory_saver=True`, so the
failure is not specific to 0.6B or 4B model size. The diagnostic override:

```bash
export RELAX_SGLANG_DISABLE_MEMORY_SAVER=1
```

is forwarded through the AMD launcher into Ray runtime env and makes Relax pass
`enable_memory_saver=False` to SGLang while leaving colocated actor offload
enabled. The first foreground validation with this override reached SGLang
weight and KV-cache loading without the immediate named-symbol failure before
the 5-minute validation cutoff.

The follow-up mini A/B now separates the failure more sharply:

- No-memory-saver colocate runs can complete rollout and log
  `rollout/reward/mean=-64.0`, but then fail when the Megatron actor computes
  log-probs on the same GPU after SGLang has run.
- The same Qwen3-Mock-0.5B setup succeeds in non-colocated sync mode
  (`oc9gqu3i`), completing rollout, actor training, rollout weight update, and
  checkpoint save.
- Returning the actor to `sleep()` after sync `update_weights()` fixes the
  earlier torch-memory-saver state error (`Cannot resume allocation that is not
  paused`) and verifies actor memory is released before rollout, but it does
  not fix the actor-side HIP named-symbol failure.
- `AMD_SERIALIZE_KERNEL=3` did not improve stack attribution for the actor
  failure; the reported frame still landed in Megatron attention query reshape.

Interpretation: once SGLang memory saver is disabled, the remaining colocate
blocker is same-GPU SGLang/Megatron ROCm handoff after rollout, not the mock
checkpoint or the completion-length reward.

### Step 3: Read the reward signal before interpreting training

The sync short6 run completed, but all rollouts were flat:

```text
rollout 0: rollout/reward/mean=-64.0, rollout/response_len/mean=64.0
rollout 1: rollout/reward/mean=-64.0, rollout/response_len/mean=64.0
rollout 5: rollout/reward/mean=-64.0, rollout/response_len/mean=64.0
rollout/advantages=0.0
train/grad_norm=0.0
train/loss=0.0
train/pg_loss=0.0
```

This means the framework ran correctly, but the policy had no learning signal.
Do not call this learned behavior.

### Step 4: Use shorter max response lengths to force signal

If the model generates to the cap, reduce `ROLLOUT_MAX_RESPONSE_LEN` before
debugging GRPO. Candidate next configs:

```bash
ROLLOUT_MAX_RESPONSE_LEN=8
ROLLOUT_TEMPERATURE=1.5
```

or:

```bash
ROLLOUT_MAX_RESPONSE_LEN=16
ROLLOUT_TEMPERATURE=2.0
```

The goal is to make early EOS or short completions appear often enough that
groups have nonzero reward standard deviation.

### Step 4b: Use sampled GRPO for convergence probes

The `../verl` length-penalty skill reproduced the same pattern: random-init
mini models completed 100 steps but stayed at the response-length cap, so they
are useful for runtime smoke tests but not for proving learnability. A
pretrained Qwen2.5-3B run also drifted to the cap when it used the baseline
recipe. The best verl diagnostic used sampled GRPO:

```bash
ROLLOUT_N=4
ROLLOUT_DO_SAMPLE=True
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
```

The Relax equivalent is:

```bash
N_SAMPLES_PER_PROMPT=4
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
```

Run this with a real pretrained checkpoint, not the Qwen3-Mock random-weight
checkpoint. The mock checkpoint is still the right fast path for reproducing
ROCm/SGLang/Megatron runtime failures, but a random policy can fail to discover
EOS often enough to create a useful GRPO signal.

For a small, synchronous Relax probe, use one prompt group per train step:

```bash
RELAX_EXECUTION_MODE=sync
USE_COLLOCATE=0
ROLLOUT_BATCH_SIZE=2
N_SAMPLES_PER_PROMPT=2
NUM_STEPS_PER_ROLLOUT=1
GLOBAL_BATCH_SIZE=4
MICRO_BATCH_SIZE=2
ROLLOUT_MAX_RESPONSE_LEN=8
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
```

The actor-side TransferQueue sampler must receive a batch size divisible by
`n_samples_per_prompt`. On the 2-rank actor setup, `n_samples_per_prompt=4`
failed with `batch_size (2) must be a multiple of n_samples_per_prompt (4)`;
`n_samples_per_prompt=2` with `MICRO_BATCH_SIZE=1` failed with
`batch_size (1) must be a multiple of n_samples_per_prompt (2)`. The first
validated sampled shape is therefore `ROLLOUT_BATCH_SIZE=2`,
`N_SAMPLES_PER_PROMPT=2`, `GLOBAL_BATCH_SIZE=4`, `MICRO_BATCH_SIZE=2`.

Judge the run by reward variance and clipping before judging convergence:

```text
rollout/reward/std > 0
rollout/advantages != 0
train/grad_norm > 0
rollout/truncated_ratio < 1
rollout/response_len/mean decreases or stays short
```

The `jeid5rq1` run completed all 10 rollout/train steps and saved a torch_dist
checkpoint, but all 40 sampled completions hit `ROLLOUT_MAX_RESPONSE_LEN=8`.
Reward stayed `-8.0`, `rollout/truncated_ratio` stayed `1.0`,
`rollout/advantages` stayed `0.0`, and `train/grad_norm` stayed `0.0`. This
proves the synchronous sampled path works, but not learnability. The next probe
should change the rollout distribution or prompt/stop behavior enough to
produce at least one shorter completion.

The first short-prompt probe used:

```bash
PROMPT_SET=examples/completion_length/prompts_end_response.jsonl
ROLLOUT_MAX_RESPONSE_LEN=4
ROLLOUT_TEMPERATURE=2.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
```

The foreground validation run (`wqqv6si0`) reached SGLang ready and actor
initialization. Two production attempts with the same config then hung before
W&B startup and before any Relax service actors were created. Ray state showed
only the `JobSupervisor` actor alive and `0/4` GPUs allocated; the driver log
stopped immediately after the Megatron FSDP fallback warning. A direct
faulthandler probe of `python -m relax.entrypoints.train --help` showed the
driver blocked in `torch.utils.file_baton.wait()` while importing
`ROCm-Megatron-LM/megatron/core/inference/unified_memory.py`, specifically
during `load_inline(name="managed_alloc_runtime", ...)`. Removing the stale
cache lock at `~/.cache/torch_extensions/py312_cpu/managed_alloc_runtime/lock`
allowed startup to continue. Treat the stopped production attempts as a stale
JIT-lock artifact, not as evidence about reward learning.

The post-lock production retry (`6edey3v7`) completed all 10 rollout/train
steps successfully with the same short-prompt config. It confirmed that the
stale lock was the startup blocker, but it did not produce a learning signal:
every sampled completion reached `ROLLOUT_MAX_RESPONSE_LEN=4`,
`rollout/reward/mean=-4.0`, `rollout/truncated_ratio=1.0`,
`rollout/advantages=0.0`, and `train/grad_norm=0.0`.

The real Qwen3-4B retry must use the known-good TP2 shape rather than the old
single-actor-GPU non-colocate shape that OOMed: `ACTOR_RESOURCE_GPUS=2`,
`ROLLOUT_RESOURCE_GPUS=2`, `TENSOR_MODEL_PARALLEL_SIZE=2`, and
`ENABLE_SEQUENCE_PARALLEL=1` on four MI210 GPUs. With that setup, run
`0cp1qu3s` completed all 10 rollout/train steps and saved torch_dist checkpoint
iteration 9. It proves the 4B operational path still works, including W&B and
checkpointing, but model size alone did not expose EOS-discovery signal:
response length stayed capped at 4, reward stayed `-4.0`, advantages stayed
zero, and actor gradients stayed zero.

The longer Qwen3-4B TP2 run (`4cz43iek`) used the same config for 50 rollouts.
It completed and checkpointed iteration 49. Unlike the 10-step run, it did find
one shorter sample at rollout 40: `rollout/response_len/mean=3.75`,
`rollout/reward/mean=-3.75`, `rollout/truncated_ratio=0.75`, and
`train/grad_norm=61.13`. This confirms that the longer run can occasionally
expose a real GRPO signal. However, the improvement did not persist: rollout 49
returned to `response_len=4.0`, `reward=-4.0`, `advantages=0.0`, and
`grad_norm=0.0`. Treat this as sparse-discovery evidence, not convergence.

### Step 5: Treat long fully async separately

The fully async path can produce some variance, but long runs failed from queue
capacity:

```text
max_staleness=1: Storage capacity exceeded: 32 existing + 16 new > 32
max_staleness=3: Storage capacity exceeded: 64 existing + 16 new > 64
```

Increasing `MAX_STALENESS` only delayed the failure. For long fully async toy
runs, address queue capacity or producer/consumer backpressure first.

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Flat reward at max response length | Model generated until `rollout_max_response_len` | Lower max response length or increase sampling variance |
| Zero advantages and gradients | All samples in a group received the same reward | Check reward variance before interpreting training curves |
| Sync colocate fails before rollout | SGLang loads weights, then crashes at first prefill with ROCm `HIP error: named symbol not found` | This reproduced on both Qwen3-4B and Qwen3-0.6B, so debug colocated SGLang/ROCm runtime before interpreting reward curves |
| Qwen3-Mock-0.5B sync colocate baseline | The mini model reproduced the same first-prefill HIP named-symbol failure with SGLang `enable_memory_saver=True` | Use the mock model for quick debugging; this is not a model-size-only failure |
| Qwen3-Mock-0.5B with `RELAX_SGLANG_DISABLE_MEMORY_SAVER=1` | Foreground validation reached SGLang load/KV allocation with `enable_memory_saver=False`, then timed out before first prefill observation | Continue this shape in tmux to test whether SGLang memory saver is the colocate trigger |
| Low SGLang pressure plus `AMD_SERIALIZE_KERNEL=3` | SGLang servers reach ready, then both engines crash at first prefill; PyTorch warns `AMD_SERIALIZE_KERNEL: 3` is an invalid boolean-style value | The failure is not caused by high request/token caps, and this stack does not accept `AMD_SERIALIZE_KERNEL=3` cleanly through PyTorch env parsing |
| Qwen3-4B sync non-colocate fails at first train step | Rollout completes, then actor OOMs while Adam initializes optimizer state at `optimizer.step()` | This is a separate actor memory issue, not the SGLang named-symbol failure |
| Fully async long run with `max_staleness=1` | TransferQueue filled at capacity 32 | This is a pipeline/backlog failure, not a reward failure |
| Fully async long run with `max_staleness=3` | TransferQueue filled at capacity 64 | Higher staleness only delayed the same failure |
| Prompt asked for EOS in natural language | Model explained or answered the instruction | Use less instruction-like prompts or generation settings that expose EOS |
| Short-prompt production retry after foreground timeout | Job remained RUNNING but Ray showed only `JobSupervisor`, `0/4` GPUs used, and no W&B startup | Check for stale Torch extension locks, especially `~/.cache/torch_extensions/py312_cpu/managed_alloc_runtime/lock`; this is before rollout/reward metrics exist |
| Short-prompt `max_len=4`, `temperature=2.0` after lock removal | Run completed, but every completion still reached the response cap | Startup was fixed, but rollout distribution still has zero reward variance; change the task prompt, stop behavior, or sampling setup rather than rerunning the same config |
| Qwen3-4B TP2 short-prompt retry | The 4B path completed and checkpointed, but every completion still reached the response cap | 4B size alone does not fix this toy task; keep the TP2 resource shape for operational validation, then change prompt/stop/sampling to create reward variance |
| Qwen3-4B TP2 long50 short-prompt retry | One rollout out of 50 produced a shorter completion and nonzero grad norm, but later rollouts returned to the response cap | Longer training can expose sparse signal, but this setup does not reliably sustain EOS learning; increase sample diversity or change prompt/stop behavior before expecting convergence |

## Configuration

```yaml
environment: relaxrl_rocm_after_fix
hardware: 4x MI210
model:
  config: qwen3-0.6B
  asset: /vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets/Qwen3-0.6B
reward:
  rm_type: completion_length
  reward: -len(completion_tokens)
validated_sync_run:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/ougxjccy
  job: raysubmit_9cffcK59j7Eqw3Dm
  log: log/amd-qwen3-0.6b-4gpu-20260620_085332.log
  num_rollout: 6
  rollout_batch_size: 2
  n_samples_per_prompt: 8
  global_batch_size: 8
  num_steps_per_rollout: 2
  rollout_max_response_len: 64
  rollout_temperature: 1.2
  max_staleness: 0
  result: completed but reward flat at -64.0
qwen3_4b_sync_colocate_probe:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/8a5r2uzv
  job: raysubmit_1RnUQtFDR8sGU5tv
  log: log/amd-qwen3-4b-2gpu-20260620_103742.log
  launcher: scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
  use_colocate: 1
  result: failed before rollout with SGLang ROCm HIP named-symbol error
qwen3_4b_sync_noncolocate_probe:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/t1i8zbrt
  job: raysubmit_Y8DAkNNcQMAW3wVY
  log: log/amd-qwen3-4b-2gpu-20260620_124613.log
  launcher: scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh
  use_colocate: 0
  result: reached rollout and training, then failed at first optimizer.step with HIP OOM
qwen3_0_6b_sync_colocate_probe:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/rplfx3eg
  job: raysubmit_89XcyQTG7A3N4uA7
  log: log/amd-qwen3-0.6b-4gpu-20260620_130721.log
  launcher: scripts/training/multimodal/amd_qwen3_0_6b_4gpu_e2e.sh
  use_colocate: 1
  result: failed before rollout with SGLang ROCm HIP named-symbol error
qwen3_0_6b_sync_colocate_low_pressure_serialize_probe:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/jidwd60n
  job: raysubmit_qFkGVzSh5WscQJKV
  log: log/amd-qwen3-0.6b-4gpu-20260620_234750.log
  launcher: scripts/training/multimodal/amd_qwen3_0_6b_4gpu_e2e.sh
  use_colocate: 1
  sglang_server_concurrency: 16
  sglang_max_running_requests: 16
  sglang_max_total_tokens: 16384
  amd_serialize_kernel: 3
  result: reproduced first-prefill HIP named-symbol error after SGLang server ready; stopped manually
validated_async_short_run:
  wandb: https://wandb.ai/erlandpg/relax-amd-completion-length/runs/lwrhk4rw
  job: raysubmit_LZMcVZPYfjdfSZYm
  log: log/amd-qwen3-0.6b-4gpu-20260620_082658.log
  result: completed with occasional shorter completions
```
