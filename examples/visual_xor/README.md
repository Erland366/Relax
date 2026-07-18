# Random-Initialized Qwen3-VL Visual XOR

This example tests whether Relax can learn a genuinely image-conditioned
policy. It uses a compact, randomly initialized Qwen3-VL model and does not
load pretrained language, vision, projection, embedding, or output-head
weights.

The experiment deliberately has two stages:

1. Multimodal SFT teaches the model to read the left glyph and to emit a valid
   `A<|im_end|>` or `B<|im_end|>` response while preserving both actions.
2. Relax GRPO learns visual XOR: `A` when the two glyphs have the same
   orientation and `B` when their orientations differ.

The SFT task never contains the XOR rule or XOR labels. This example also does
not reproduce Engram-ViT's memory/reasoning split, lookup addresses, or memory
objectives.

## Task contract

Each 256x256 image contains two large abstract glyphs:

```text
bit 0: two vertical bars
bit 1: two horizontal bars

00 -> A
01 -> B
10 -> B
11 -> A
```

Colors, position, scale, width, and border distractors vary independently of
the bits. Dataset rows use opaque IDs. The private label and bit metadata are
stored outside the prompt.

The Relax reward is strict:

```text
correct action + EOS:  +1
wrong action + EOS:     0
invalid or truncated:  -1
```

## Prepare metadata and datasets

Use the existing SFT environment. Do not install or upgrade anything.

```bash
export RELAX_ROOT=/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/Relax-rocm-megatron_after_fix_profiled_20260610
export SFT_ROOT=/vast/users/qirong.ho/erland/Python_project/SFT_training
export PROCESSOR_DIR="$SFT_ROOT/qwen3-vl-processor-metadata"
export VISUAL_XOR_DATA="$SFT_ROOT/visual_xor_data"

cd "$SFT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$RELAX_ROOT"

python "$RELAX_ROOT/examples/visual_xor/prepare_metadata.py" \
  --output-dir "$PROCESSOR_DIR"

python "$RELAX_ROOT/examples/visual_xor/generate_data.py" \
  --output-dir "$VISUAL_XOR_DATA"
```

`prepare_metadata.py` is pinned to
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`.
It uses an explicit processor/tokenizer allowlist and rejects model weight
files. The final compact model is constructed from a new configuration with
random weights.

The generator refuses to overwrite an existing dataset. Its normal outputs
are:

```text
sft_train.parquet  8192 examples
sft_eval.parquet   1024 examples
xor_train.parquet  4096 examples
xor_eval.parquet   1024 examples
```

## Train the visual bootstrap

Run the processor/dataset preflight first:

```bash
python "$RELAX_ROOT/examples/visual_xor/train_sft.py" \
  --processor-path "$PROCESSOR_DIR" \
  --train-data "$VISUAL_XOR_DATA/sft_train.parquet" \
  --eval-data "$VISUAL_XOR_DATA/sft_eval.parquet" \
  --output-dir "$SFT_ROOT/qwen3-vl-0.37b-visual-bootstrap-sft" \
  --preflight-only --report-to none
```

Then train on exactly two GPUs:

```bash
export VISUAL_SFT="$SFT_ROOT/qwen3-vl-0.37b-visual-bootstrap-sft"

accelerate launch \
  --config_file "$SFT_ROOT/accelerate_configs/sft.yaml" \
  --num_processes 2 \
  "$RELAX_ROOT/examples/visual_xor/train_sft.py" \
  --processor-path "$PROCESSOR_DIR" \
  --train-data "$VISUAL_XOR_DATA/sft_train.parquet" \
  --eval-data "$VISUAL_XOR_DATA/sft_eval.parquet" \
  --output-dir "$VISUAL_SFT"
```

The model has 371,438,976 parameters: 12 language layers, a six-layer vision
tower, and tied token embeddings/LM head. SFT uses BF16, global batch 16,
`1e-4` learning rate, 1,024 optimizer steps, no packing, completion-only loss,
and no intermediate checkpoints. It writes one final Hugging Face checkpoint.

Validate that checkpoint before giving it to Relax:

```bash
python "$RELAX_ROOT/examples/visual_xor/evaluate_sft.py" "$VISUAL_SFT" \
  --sft-eval-data "$VISUAL_XOR_DATA/sft_eval.parquet" \
  --xor-eval-data "$VISUAL_XOR_DATA/xor_eval.parquet"
```

The gate requires at least 95% valid A/B+EOS responses, both actions at a
20-80% marginal rate, at least 90% mixed eight-sample XOR groups, a 60-85%
left-glyph preference rate, a 35-65% unsolved XOR baseline, at most 5%
truncation, and a measurable left-glyph counterfactual probability shift.

## Run Relax without checkpoints

The first command is a two-rollout startup and optimizer smoke test:

```bash
cd "$RELAX_ROOT"
HF_CHECKPOINT="$VISUAL_SFT" \
PROMPT_SET="$VISUAL_XOR_DATA/xor_train.parquet" \
HIP_VISIBLE_DEVICES=0,1 \
  bash scripts/debug/qwen3_vl_visual_xor_sync_2gpus.sh
```

After the smoke test completes an optimizer step and weight synchronization,
run the 100-rollout learning test:

```bash
HF_CHECKPOINT="$VISUAL_SFT" \
PROMPT_SET="$VISUAL_XOR_DATA/xor_train.parquet" \
HIP_VISIBLE_DEVICES=0,1 \
NUM_ROLLOUT=100 \
  bash scripts/debug/qwen3_vl_visual_xor_sync_2gpus.sh
```

The wrapper forcibly uses synchronous, non-colocated execution with one actor
GPU and one SGLang GPU. It clears inherited async/colocation settings, passes
`--multimodal-keys '{"image":"image"}'`, disables KL, and sets
`SAVE_CHECKPOINTS=0`. Relax writes no Megatron checkpoint and no final HF
export. W&B and the ordinary log remain under `log/`.

On ROCm, Relax also replaces Qwen3-VL's Transformer Engine-only vision stack
with Megatron's local attention, normalization, and tensor-parallel layers.
Packed image/frame groups are represented by an explicit bidirectional
block-diagonal boolean mask, so different images or video frames cannot attend
to one another. This debugging fallback has quadratic mask memory and does not
support vision CUDA-graph padding; it fails explicitly if that mode is enabled.

The most useful W&B metrics are:

```text
rollout/correct_action/mean
rollout/valid_action/mean
rollout/action_a/mean
rollout/action_b/mean
rollout/target_a_accuracy/mean
rollout/target_b_accuracy/mean
rollout/combo_00_accuracy/mean
rollout/combo_01_accuracy/mean
rollout/combo_10_accuracy/mean
rollout/combo_11_accuracy/mean
train/grad_norm
```

Treat the run as successful only if the last 20 rollouts have at least 0.80
mean reward, 95% valid responses, at least 75% accuracy for both targets, and
at least 65% accuracy for every input combination. Mean reward alone is not
enough: a constant A or B policy receives 50% on this balanced task.

## Failure localization

- Metadata preflight fails: processor/tokenizer files or special-token IDs do
  not match the pinned Qwen3-VL interface.
- SFT validity fails: the model has not learned the short A/B+EOS protocol.
- SFT XOR accuracy is above 65%: the initializer already solves too much of the
  RL task.
- Relax reports missing image placeholders: the Parquet schema or
  `MULTIMODAL_KEYS` mapping is wrong.
- Rewards stay homogeneous: inspect A/B rates and mixed-group rate before
  increasing the rollout budget.
- Aggregate reward reaches about 0.5 but one target/combo stays near zero: the
  policy collapsed to one global action instead of using the image.
