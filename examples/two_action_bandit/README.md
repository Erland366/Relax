# Two-Action Bandit Learning Test

This is the smallest Relax task that tests an actual GRPO learning loop rather
than only startup, termination, or constant-reward handling. The policy emits
one content token and then EOS:

```text
A + EOS -> reward +1
B + EOS -> reward  0
anything else -> reward -1
```

Both valid actions have the same response length. Improvement therefore cannot
come from learning EOS timing or completion length. It must come from shifting
probability from B to A after reward normalization, optimization, and actor to
rollout weight synchronization.

The task returns a reward dictionary with primary key `score` and auxiliary
metrics `action_a`, `action_b`, and `valid_action`. The debug launcher sets
`REWARD_KEY=score`, so W&B logs the normal reward statistics plus explicit
action-rate statistics.

## Why the recipe is Qwen3-only

The SFT workspace contains historical Qwen3, GLM-4-MoE, and MiniMax-M2 models,
but only Qwen3 has a validated reduced-model Relax ROCm path in this repository.
The GLM and MiniMax chat templates, embedding tying, model providers, and
conversion coverage are not interchangeable with the current mock Qwen3 path.
The SFT script therefore fails loudly unless `model_type` is `qwen3` and the
source config uses tied word embeddings.

## Train the balanced initializer

Use the existing SFT environment; do not install or upgrade anything. The
trainer loads a config and tokenizer only, constructs the model with random
weights, and trains 32 `A + EOS` and 32 `B + EOS` examples. It writes no
intermediate checkpoints and saves only the final Hugging Face model needed by
Relax.

First run the CPU-only preflight:

```bash
cd /path/to/SFT_training
source .venv/bin/activate
export RELAX_ROOT=/path/to/Relax-rocm-megatron_after_fix_profiled_20260610
PYTHONPATH="$RELAX_ROOT" python "$RELAX_ROOT/examples/two_action_bandit/train_sft.py" \
  --preflight-only
```

The expected Qwen result includes:

```text
model_type=qwen3, tie_word_embeddings=True, examples=64, choice_a=32,
choice_b=32, choice_token_ids=32/33, completion_tokens=3
```

Train on four GPUs with the SFT project's existing Accelerate config:

```bash
cd /path/to/SFT_training
source .venv/bin/activate
export RELAX_ROOT=/path/to/Relax-rocm-megatron_after_fix_profiled_20260610
export BANDIT_SFT="$PWD/mini-qwen3-0.5b_TWO-ACTION-BANDIT-SFT"
PYTHONPATH="$RELAX_ROOT" accelerate launch \
  --config_file accelerate_configs/sft.yaml \
  "$RELAX_ROOT/examples/two_action_bandit/train_sft.py" \
  --output-dir "$BANDIT_SFT"
```

The model is initialized through `AutoModelForCausalLM.from_config`; no
pretrained model weights are read. `save_strategy="no"` prevents intermediate
optimizer checkpoints.

## Validate the initializer

Before Relax training, sample 256 responses with the exact rollout prompt:

```bash
PYTHONPATH="$RELAX_ROOT" python "$RELAX_ROOT/examples/two_action_bandit/evaluate_sft.py" \
  "$BANDIT_SFT"
```

Acceptance requires:

- A and B are exactly the two highest-probability first tokens;
- their combined first-token probability is at least 0.90;
- each sampled action rate is between 0.20 and 0.80;
- EOS probability after each action is at least 0.90;
- at least 95% of samples are exactly action plus EOS;
- at least half of reconstructed eight-sample groups contain both actions;
- truncation is at most 5%.

Do not start GRPO if this gate fails. A saturated initial policy recreates the
zero-advantage failure mode of the immediate-EOS task.

## Run Relax without checkpoints

The launcher requires the accepted model path explicitly and keeps Relax model
and optimizer checkpointing disabled:

```bash
cd "$RELAX_ROOT"
HF_CHECKPOINT="$BANDIT_SFT" bash scripts/debug/qwen_two_action_bandit_sync_2gpus.sh
```

The default workload uses 30 synchronous updates, groups of eight samples,
16 samples per training step, a three-token generation cap, and weight sync
after every update. A valid trajectory still has exactly two tokens: action
plus EOS. The extra cap position lets SGLang report EOS completion as `stop`
instead of `length`; any actual third generated token remains invalid. W&B
cache data is written under `log/wandb`, not `relax_assets`.

The expected learning pattern is:

```text
early: A/B groups -> nonzero reward variance and nonzero gradient norms
middle: action_a rises, action_b falls, reward mean rises
late: mostly A groups -> zero-variance groups and gradients naturally return toward zero
```

Watch `rollout/action_a/mean`, `rollout/action_b/mean`,
`rollout/valid_action/mean`, `rollout/reward/mean`, and `train/grad_norm`.
Response-length variance should be zero; that is intentional for this task.
