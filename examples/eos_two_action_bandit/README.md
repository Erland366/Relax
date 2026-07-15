# Joint EOS and Two-Action Bandit

This example trains one Qwen3 mock policy on two task environments in the same
Relax run. It is deliberately small enough to isolate multitask reward routing,
per-prompt GRPO grouping, actor training, and weight synchronization.

The tasks are:

```text
EOS prompt:    immediate EOS -> reward -1; every additional token lowers reward
Bandit prompt: A + EOS       -> reward +1
               B + EOS       -> reward  0
               invalid       -> reward -1
```

## Train one shared SFT initializer

Use the existing SFT environment without installing or upgrading anything. The
trainer reads only the reduced Qwen3 config and tokenizer and initializes model
weights randomly with tied input/output embeddings. It does not load either
single-task SFT checkpoint.

The 128-example dataset is split exactly as follows:

```text
32 EOS-prompt -> EOS
32 EOS-prompt -> OK + EOS
32 bandit-prompt -> A + EOS
32 bandit-prompt -> B + EOS
```

Run the CPU-only preflight first:

```bash
cd /path/to/SFT_training
source .venv/bin/activate
export RELAX_ROOT=/path/to/Relax-rocm-megatron_after_fix_profiled_20260610
PYTHONPATH="$RELAX_ROOT" python \
  "$RELAX_ROOT/examples/eos_two_action_bandit/train_sft.py" \
  --preflight-only --report-to none
```

Then train on four GPUs:

```bash
export JOINT_SFT="$PWD/mini-qwen3-0.5b_EOS-TWO-ACTION-BANDIT-SFT"
PYTHONPATH="$RELAX_ROOT" accelerate launch \
  --config_file accelerate_configs/sft.yaml \
  "$RELAX_ROOT/examples/eos_two_action_bandit/train_sft.py" \
  --output-dir "$JOINT_SFT"
```

The recipe uses `save_strategy="no"`; it writes no intermediate optimizer
checkpoints and saves only the final Hugging Face model required by Relax.

## Validate both tasks

The joint checkpoint must pass each task's acceptance gate independently:

```bash
PYTHONPATH="$RELAX_ROOT" python \
  "$RELAX_ROOT/examples/eos_two_action_bandit/evaluate_sft.py" \
  "$JOINT_SFT"
```

For both prompts, the two intended outputs must be the two highest-probability
first tokens with at least 90% combined probability. Each intended output must
sample at a 20-80% rate, exact valid responses must be at least 95%, EOS after a
content target must be at least 90%, at least half of eight-response groups must
contain both choices, and truncation must be at most 5%.

The validated 100-step run produced 47.3% immediate EOS versus 50.4% `OK`, and
51.2% A versus 46.1% B. Exact response validity was 98.4% for EOS/OK and 97.3%
for A/B; every reconstructed eight-response group was mixed.

## Why every Relax update contains both tasks

[`prompts.jsonl`](prompts.jsonl) contains exactly two rows: one tagged EOS row
and one tagged bandit row. The launcher uses `rollout_batch_size=2`, so shuffling
may change their order but cannot change the composition of a batch. Each prompt
is expanded into eight responses before GRPO normalization:

```text
EOS prompt    -> one eight-response GRPO group
bandit prompt -> one eight-response GRPO group
both groups   -> one 16-sample shared optimizer update
```

The reward function requires `metadata.task` and fails loudly for missing or
unknown tasks. Both task branches return a dictionary with the shared primary
key `score`, while task-specific fields provide separate W&B metrics.

## Run Relax without checkpoints

```bash
cd "$RELAX_ROOT"
HF_CHECKPOINT="$JOINT_SFT" \
  bash scripts/debug/qwen_eos_two_action_bandit_sync_2gpus.sh
```

Relax checkpoint saving remains disabled. W&B cache files and the run log are
written under `log/`, not `relax_assets`.

The most useful metrics are:

```text
rollout/task_eos/mean                 expected to stay 0.5
rollout/task_bandit/mean              expected to stay 0.5
rollout/eos_reward/mean               should rise toward -1
rollout/eos_response_length/mean      should fall toward 1
rollout/immediate_eos/mean            should rise
rollout/bandit_score/mean             should rise toward +1
rollout/action_a/mean                 should rise
rollout/action_b/mean                 should fall
rollout/valid_action/mean             should remain high
train/grad_norm                       should be nonzero early
```

Do not use aggregate `rollout/reward/mean` to judge multitask learning: the EOS
optimum is -1 while the bandit optimum is +1, so their average can hide progress.
