# Random-Initialized Qwen3-VL Visual XOR

This example tests whether Relax can learn a genuinely image-conditioned
policy. It uses a compact, randomly initialized Qwen3-VL model and does not
load pretrained language, vision, projection, embedding, or output-head
weights.

The experiment has two benchmark tiers that share the same renderer and strict
reward:

1. **Refinement (the default learning smoke test):** a short second SFT stage
   gives the model a deliberately noisy 70%-correct XOR policy. Relax only has
   to refine that real image-conditioned policy toward 100%.
2. **Discovery (the harder diagnostic):** the original visual bootstrap only
   learns to read the left glyph and emit valid `A<|im_end|>` or
   `B<|im_end|>`. Relax must discover XOR from reward alone.

Only the discovery SFT excludes the XOR rule and XOR labels. The refinement
SFT intentionally includes noisy XOR labels because its purpose is to answer a
much narrower question: can this Relax path improve a multimodal policy when
the correct visual feature and both actions are already represented? This
example does not reproduce Engram-ViT's memory/reasoning split, lookup
addresses, or memory objectives.

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

The refinement tier is the first test to run. It should improve quickly; if it
does not, the likely problem is in rollout, reward/advantage computation,
weight synchronization, or actor optimization rather than visual task
discovery.

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

## Build the easy refinement initializer

Generate the separate refinement datasets. This does not overwrite the
original discovery datasets:

```bash
export VISUAL_REFINEMENT_DATA="$SFT_ROOT/visual_xor_refinement_data"

python "$RELAX_ROOT/examples/visual_xor/generate_refinement_data.py" \
  --output-dir "$VISUAL_REFINEMENT_DATA"
```

The bundle contains an 8,000-row noisy SFT training set, a 1,000-row noisy SFT
evaluation set, 64 fixed ordered RL training images, 128 held-out RL images,
and permuted-image and constant-image control sets. Within each SFT split,
every one of `00`, `01`, `10`, and `11` has exactly 70% correct XOR targets
and 30% flipped targets. Render seeds do not overlap between splits.

Continue from the accepted local random-initialized visual bootstrap. Despite
the Transformers API method name used internally, this is a local checkpoint
continuation: it never loads an external pretrained model.

```bash
export VISUAL_REFINEMENT_SFT="$SFT_ROOT/qwen3-vl-0.37b-visual-xor-refinement-sft"

python "$RELAX_ROOT/examples/visual_xor/train_sft.py" \
  --processor-path "$PROCESSOR_DIR" \
  --initial-checkpoint "$VISUAL_SFT" \
  --train-data "$VISUAL_REFINEMENT_DATA/refinement_sft_train.parquet" \
  --eval-data "$VISUAL_REFINEMENT_DATA/refinement_sft_eval.parquet" \
  --output-dir "$VISUAL_REFINEMENT_SFT" \
  --max-steps 512 \
  --learning-rate 5e-5 \
  --run-name qwen3-vl-0.37b-visual-xor-refinement-sft \
  --preflight-only --report-to none

accelerate launch \
  --config_file "$SFT_ROOT/accelerate_configs/sft.yaml" \
  --num_processes 2 \
  "$RELAX_ROOT/examples/visual_xor/train_sft.py" \
  --processor-path "$PROCESSOR_DIR" \
  --initial-checkpoint "$VISUAL_SFT" \
  --train-data "$VISUAL_REFINEMENT_DATA/refinement_sft_train.parquet" \
  --eval-data "$VISUAL_REFINEMENT_DATA/refinement_sft_eval.parquet" \
  --output-dir "$VISUAL_REFINEMENT_SFT" \
  --max-steps 512 \
  --learning-rate 5e-5 \
  --run-name qwen3-vl-0.37b-visual-xor-refinement-sft
```

As in the bootstrap stage, SFT writes no intermediate checkpoints and saves
one final checkpoint because Relax needs that initializer. Apply the dedicated
gate before launching RL:

```bash
python "$RELAX_ROOT/examples/visual_xor/evaluate_refinement_sft.py" \
  "$VISUAL_REFINEMENT_SFT" \
  --xor-eval-data "$VISUAL_REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --permuted-eval-data "$VISUAL_REFINEMENT_DATA/refinement_rl_eval_permuted.parquet" \
  --constant-eval-data "$VISUAL_REFINEMENT_DATA/refinement_rl_eval_constant.parquet"
```

The accepted checkpoint must already respond validly, use both actions, score
60-80% on sampled held-out XOR, exceed 55% on every combination, react in the
correct direction when either glyph is flipped, and stay near chance when the
images are permuted or replaced by one constant image. Those controls reject a
global A/B bias or a label-order shortcut.

## Run the easy refinement benchmark

```bash
cd "$RELAX_ROOT"
HF_CHECKPOINT="$VISUAL_REFINEMENT_SFT" \
REFINEMENT_DATA="$VISUAL_REFINEMENT_DATA" \
HIP_VISIBLE_DEVICES=0,1 \
  bash scripts/debug/qwen3_vl_visual_xor_refinement_sync_2gpus.sh
```

The 64 RL training rows are deliberately eager-loaded and not shuffled. Every
consecutive rollout batch contains `00`, `01`, `10`, and `11` in that order,
and the dataset repeats exactly every 16 updates. Evaluation runs before step
0 and every eight updates against held-out, permuted-image, and constant-image
datasets. The wrapper uses batch size 4, eight samples per image, global batch
32, learning rate `3e-6`, and 50 rollouts by default. It saves no Relax model
or checkpoint.

This tier should normally show clear held-out improvement within roughly
10-50 updates. Call it converged only when held-out reward is at least 0.90,
all four combinations are at least 0.85, valid responses stay at least 0.99,
both actions remain present, and both controls remain near chance. A rising
training reward without those held-out/control conditions is not success.

## Run the experimental frozen CPU vision path

The optional CPU vision service can compute the frozen Qwen3-VL visual tower,
final projection, and all three DeepStack projections once, cache the result,
and give the same feature bundle to SGLang and Megatron:

```bash
cd "$RELAX_ROOT"
HF_CHECKPOINT="$VISUAL_REFINEMENT_SFT" \
REFINEMENT_DATA="$VISUAL_REFINEMENT_DATA" \
HIP_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh
```

The recipe adds `"vision_encoder":[1,0]` to `RESOURCE_JSON`, reserves one
logical CPU core per Ray Serve replica by default, freezes both the vision
tower and its projections, and writes no checkpoint. The one-core default
keeps this correctness smoke schedulable beside Ray and Serve control actors;
set `VISION_ENCODER_NUM_CPUS` explicitly for scaling experiments on a
CPU-rich allocation. It is Qwen3-VL image-only and requires context and
pipeline parallel sizes of one.

The recipe defaults to one CPU replica and keeps the GPU vision encoder.
Set `ENABLE_EVAL=0` on either fully asynchronous refinement launcher for a
training-only profiling run. This explicitly removes every evaluation config
and interval; `ENABLE_EVAL=1` remains the default and preserves the established
held-out and control evaluation behavior. First run the fixed-image parity
gate:

```bash
python -m examples.visual_xor.validate_cpu_vision_parity \
  --checkpoint "$VISUAL_REFINEMENT_SFT" \
  --dataset "$VISUAL_REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/parity.json \
  --device cuda:0 \
  --num-images 8
```

Only after that artifact passes should a run set
`SKIP_GPU_VISION_ENCODER=1`. When the GPU encoder is skipped, Megatron does not
construct its visual tower and SGLang replaces the meta-device visual module
before GPU materialization; raw-image fallbacks fail loudly. The local
checkpoint's theoretical BF16 parameter reduction is 51.66 MiB per full GPU
model instance, but actual VRAM must still be measured.

The local Hugging Face gate does not exercise SGLang. Run the live gate on one
visible GPU, with the same SGLang Python and ROCm `sgl_kernel` paths used by
the Relax launcher:

```bash
HIP_VISIBLE_DEVICES=0 \
python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
  --checkpoint "$VISUAL_REFINEMENT_SFT" \
  --dataset "$VISUAL_REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/sglang_live_parity.json \
  --num-images 8 \
  --parallel-samples 8 \
  --host 127.0.0.1 \
  --port 31000 \
  --base-gpu-id 0
```

This uses one SGLang server that keeps its GPU visual weights, sends each fixed
image through GPU vision and CPU-precomputed vision, flushes the cache
between paths, and compares the generated token plus exact A/B next-token
log-probabilities. With `--parallel-samples 8`, it also tests one precomputed
request and one GPU raw-image request returning eight ordered outputs and
records their scalar-equivalent versus actual request bytes. The precomputed
grouped gate is strictly score-matched. GPU grouped branches must pass their
own score, margin, and probability gates; matching generated tokens alone is
not a pass. The command installs nothing and shuts down the server before
writing the artifact.

Compare the stateless nested and packed wire representations locally before
proposing a stateful registry:

```bash
python -m examples.visual_xor.benchmark_precomputed_transport \
  --repetitions 5 \
  --output benchmark_results/cpu_vision/serialization.json
```

This benchmark starts neither Ray nor SGLang. It uses the production packed
format and also reports the projected binary-upload-plus-ID traffic so the
additional registry complexity can be judged against measured remaining
bytes.

CPU-vision eval and rollout records include cumulative and interval cache
metrics under `vision_encoder/cache/*` plus backend work and throughput under
`vision_encoder/backend/*`. They are written to the normal log and W&B run;
no separate metrics process is required. The same records include aggregated
count, total, mean, p50, p95, and maximum measurements for CPU-service round
trip, feature preparation, packed serialization, HTTP request construction,
response wait/read/decode, request/response bytes, and the original rollout
generation phases under `perf_detail/rollout/*`. Shared CPU feature work is
recorded once per prompt group rather than once per sampled response.
The historical metric key `precomputed_to_list_time` is retained for W&B
dashboard continuity even though it now measures packed BF16 serialization.

`http_response_wait` includes server queuing, feature reconstruction/H2D,
SGLang prefill, and decoding; it does not claim to separate those server-side
phases. Compare it with CPU backend, service round-trip, request-build, and
feature-conversion metrics before deciding whether the next experiment should
change CPU capacity or transport.

The corrected 2026-08-01 comparison identified transport as the current
boundary. Each raw feature was 524,312 bytes, but its JSON generation request
averaged 2.907 MB. Baseline evaluation sent 2.233 GB across 768 requests, and
a 64-sample rollout sent about 185 MB for only eight unique features. Actual
CPU encoding took 49-54 seconds for the 128 unique evaluation images; the
other 640 requests were cache hits.

Relax now uses SGLang GPU parallel sampling for a narrow eligible group:
CPU-precomputed Qwen3-VL or processor-backed GPU image-only Qwen3-VL,
multiple fresh pending samples, one shared prompt and media object, round-robin
routing (or any routing policy when exactly one SGLang engine exists),
stochastic inference, the built-in generator, and no partial rollout, routing
replay, or Slime middleware. GPU grouping runs the processor and raw-media
encoding once per prompt group while retaining tokenizer prompt IDs for
SGLang and processor-expanded IDs/features for Megatron. One
request carries `sampling_params.n=N_SAMPLES_PER_PROMPT`, and Relax maps the
ordered scalar responses back to the original samples. Request timing and
bytes are counted once, while each sample retains its own output tokens,
log-probabilities, reward, finish metadata, and Megatron visual inputs.
Unsupported groups use the established scalar path and log the reason once.
Deterministic mode stays scalar because SGLang does not provide distinct
per-branch seeds for `n>1`.

The 2026-08-03 live `n=8` gate and matched run first reduced each 64-sample
rollout from 64 SGLang requests to eight. Evaluation dispatch is now also
independent of `group_rm`: the 128 stochastic held-out prompts use one `n=4`
request each, while the deterministic controls remain scalar. That reduced the
nested-list evaluation from 768 requests and 2.233 GB to 384 requests and
1.117 GB without changing per-sample reward semantics.

Relax then replaced nested numeric features with contiguous BF16 bytes encoded
as base64 inside the same JSON envelope. The packed live parity gate matched
all eight GPU outputs; maximum A/B log-probability delta remained
`1.43e-6`. In the matched two-rollout run, evaluation traffic fell again to
268.9 MB (75.93% below grouped nested JSON, 87.96% below the original scalar
path). Each 64-sample rollout sent 5.60 MB, both rollouts and all four optimizer
updates passed, and final GPU memory returned to baseline. This stateless
representation is the current transport contract. A binary/ID registry remains
deferred until the remaining cross-request traffic is proven to be the next
bottleneck; do not scale CPU replicas yet. See
[the packed-transport report](../../training_reports/2026-08-04-qwen3-vl-packed-cpu-vision-transport.md),
[the parallel-sampling report](../../training_reports/2026-08-03-qwen3-vl-sglang-parallel-sampling.md)
and [the corrected performance report](../../training_reports/2026-08-01-qwen3-vl-gpu-cpu-vision-performance.md).

The 2026-08-06 matched steady-state experiment then disabled evaluation and
forced the CPU cache to evict every feature. The one-replica CPU producer still
completed each steady 64-sample rollout in 4.79 seconds and filled the
staleness window, while the training-cycle averaged 26.83 seconds and waited only
0.238 seconds for data. Relative to grouped GPU vision, skipping the GPU encoder reduced
actor compute by 13.68%, complete training-cycle time by 9.67%, and simultaneous
four-card peak VRAM by 738.03 MiB. The 5.60 MB packed request was 383.5 times
the GPU raw-image request, but transport did not pace this training run.
Defer a registry and CPU replica scaling until a larger workload actually
starves the actor. The GPU grouped performance run is qualified because its
64/64 generated tokens matched scalar GPU, but its strict branch-score gate
did not pass. See
[the steady-state overlap report](../../training_reports/2026-08-06-qwen3-vl-cpu-vision-steady-state-overlap.md).

Measure the three modes sequentially, from an idle four-GPU baseline, with the
same workload:

```bash
python -m examples.visual_xor.monitor_rocm_vram \
  --output benchmark_results/cpu_vision/gpu_vram.json \
  --label gpu -- \
  env HIP_VISIBLE_DEVICES=0,1,2,3 NUM_ROLLOUT=2 EVAL_INTERVAL=4 \
    SAVE_CHECKPOINTS=0 \
    bash scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_4gpus.sh

python -m examples.visual_xor.monitor_rocm_vram \
  --output benchmark_results/cpu_vision/cpu_vram.json \
  --label cpu -- \
  env HIP_VISIBLE_DEVICES=0,1,2,3 NUM_ROLLOUT=2 EVAL_INTERVAL=4 \
    SKIP_GPU_VISION_ENCODER=0 SAVE_CHECKPOINTS=0 \
    bash scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh

python -m examples.visual_xor.monitor_rocm_vram \
  --output benchmark_results/cpu_vision/cpu_skip_gpu_encoder_vram.json \
  --label cpu-skip-gpu-encoder -- \
  env HIP_VISIBLE_DEVICES=0,1,2,3 NUM_ROLLOUT=2 EVAL_INTERVAL=4 \
    SKIP_GPU_VISION_ENCODER=1 SAVE_CHECKPOINTS=0 \
    bash scripts/debug/qwen3_vl_visual_xor_refinement_fully_async_cpu_vision_4gpus.sh
```

The GPU launcher freezes its GPU visual tower and projector for
this comparison. Otherwise its gradients and optimizer state would make the
actor-side VRAM result incomparable with the frozen CPU modes. The monitor
records baseline, final, per-device peak, peak delta, and simultaneous total
peak; compare card 0-1 as actor ranks, card 2 as SGLang rollout, and card 3 as
actor-forward.

The first 2026-07-31 four-MI210 comparison completed all three modes. Omitting the
GPU visual weights reduced the simultaneous four-card peak by 155.73 MiB
relative to the otherwise identical CPU with GPU encoder kept mode. Its evaluation cache
served 768 requests with 128 encodes, 640 hits, an 83.33% hit rate, and no
evictions. Both CPU modes were about 1.53 times the GPU launcher wall time
on the one-core vision service. Both originally remained near chance because
SGLang's generic transformers wrapper did not consume its parsed precomputed
embedding field. After the DeepStack adapter fix, a GPU-encoder-skipped run scored
`0.69921875` held out and the deterministic eight-image SGLang gate matched
every token with maximum action-margin delta `1.043081283569336e-07`.
Megatron/SGLang sampled-token differences stayed below `5e-7`. See
[the GPU/CPU VRAM report](../../training_reports/2026-07-31-qwen3-vl-gpu-cpu-vision-vram.md)
for the historical per-device table and
[the live parity report](../../training_reports/2026-07-31-qwen3-vl-live-precomputed-deepstack-parity.md)
for the corrected semantic evidence.

SGLang receives the feature tensor through JSON, which is substantially
larger than the source PNG. Multiple CPU replicas also have independent
caches, so the same image can be encoded once per replica. Compare end-to-end
time, cache metrics, and the provided CPU scaling artifact before treating the
path as an optimization. See
[Frozen CPU Vision Encoder](../../docs/draft/run_frozen_vision_encoder_on_cpu.md) for
the exact tensor contract, parity and capacity commands, GPU-encoder skipping behavior,
and staged limitations.

## Run the harder discovery benchmark without checkpoints

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
GPU and one SGLang GPU. It clears inherited async/colocation settings, adds the
same output-format system prompt used during SFT, passes
`--multimodal-keys '{"image":"image"}'`, disables KL, and sets
`SAVE_CHECKPOINTS=0`. The XOR user prompt and reward remain unchanged. Relax
writes no Megatron checkpoint and no final HF export. W&B and the ordinary log
remain under `log/`.

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
