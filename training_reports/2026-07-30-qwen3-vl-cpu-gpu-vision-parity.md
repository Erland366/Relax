# Retrospective: Qwen3-VL CPU/GPU Vision Parity

- **Date:** 2026-07-30
- **Status:** Local Hugging Face gate and live GPU-resident Relax smoke passed
- **Scope:** Frozen Qwen3-VL visual tower, final projection, three DeepStack
  projections, and downstream language-model logits
- **Artifact:**
  `benchmark_results/cpu_vision/20260730_hf_parity/parity.json`

## Objective

Determine whether features produced by the frozen PyTorch CPU vision service
are equivalent to the native Hugging Face GPU visual path closely enough to
replace GPU vision computation. Separate:

1. structural or semantic integration errors;
2. ordinary cross-device BF16 numerical drift; and
3. downstream policy behavior changes.

This distinction matters because an exact tensor comparison can reject a
correct cross-device implementation, while a response-only comparison can
hide a wrongly ordered or unused DeepStack stream.

## Setup

### Environment

- Hardware: one AMD MI210 selected from the current four-GPU Slurm allocation
- Runtime interpreter:
  `/vast/users/qirong.ho/miniforge3/envs/relaxrl_rocm_after_fix/bin/python`
- Python: 3.12.13
- PyTorch: 2.9.1+rocm6.3
- ROCm reported by PyTorch: 6.3
- Transformers: 5.3.0
- GPU reference: Hugging Face Qwen3-VL in BF16 with SDPA
- CPU candidate: standalone frozen Qwen3-VL visual module in BF16
- Checkpoint:
  `/vast/users/qirong.ho/erland/Python_project/SFT_training/qwen3-vl-0.37b-visual-xor-refinement-sft`
- Dataset: eight unique images from
  `visual_xor_refinement_data/refinement_rl_eval.parquet`

Each image produced 64 merged visual tokens with text hidden size 1024. Across
eight images, every compared stream therefore had shape `[512, 1024]`.

### Compared representations

The diagnostic compared four separately named streams:

```text
vision_embeds
deepstack_visual_embeds_0
deepstack_visual_embeds_1
deepstack_visual_embeds_2
```

It also compared the full next-token vocabulary distribution after running:

```text
native GPU image forward
    versus
CPU precomputed features -> GPU language-only forward
```

Both sides consumed the same processor-produced `pixel_values` and
`image_grid_thw`. The experiment therefore measured vision execution parity,
not image-preprocessor parity.

## What we tried

### Attempt 1: Run the original eight-image parity command

Model loading and native visual execution succeeded, but the precomputed
language path failed before comparison:

```text
TypeError: Qwen3VLModel.get_rope_index() missing 1 required positional
argument: 'mm_token_type_ids'
```

Transformers 5.3 made `mm_token_type_ids` a required argument to
`Qwen3VLModel.get_rope_index`, even when its value is `None`. The parity helper
already passed the field to the alternative position-ID implementation but
not to `get_rope_index`.

The fix was to pass the field explicitly:

```python
mm_token_type_ids=model_inputs.get("mm_token_type_ids")
```

This was a real API compatibility error, not numerical drift.

### Attempt 2: Keep the original all-elements feature gate

After the API fix, all model execution completed. The first feature stream
then failed:

```text
ValueError: vision_embeds is not within rtol=0.01 and atol=0.01
```

The original gate used `torch.allclose`, so one non-close value rejected an
entire `[512, 1024]` stream.

### Attempt 3: Decompose the feature and response errors

A one-image probe compared the existing CPU eager-attention path and a
temporary CPU SDPA configuration against the same GPU SDPA reference. Both
produced:

- feature cosine near `0.999993`;
- maximum feature error `0.03125`;
- mean feature error around `0.002`;
- response KL around `1e-8`; and
- identical top tokens.

Changing only the CPU attention interface did not remove the small differences
or materially change downstream logits. This evidence did not support an
attention-routing or DeepStack semantic bug.

The full eight-image diagnostic then counted every non-close value rather than
collapsing the stream to one boolean.

### Attempt 4: Replace the brittle gate with layered criteria

The final feature gate requires all of the following:

```text
stream names and order are exact
shapes are exact
all values are finite
cosine similarity >= 0.999
at least 99.99% of values satisfy rtol=0.01, atol=0.01
maximum absolute error <= 0.05
```

Behavioral parity retains:

```text
mean full-vocabulary KL <= 0.001
maximum per-sample KL <= 0.01
every top token matches
absolute P(A) and P(B) deltas <= 0.01
```

A regression test allows sparse BF16-scale rounding differences, while a
separate test proves that a sparse large outlier still fails.

## Results

### Feature streams

| Stream | Shape | Cosine | Max absolute error | Mean absolute error | Close fraction |
|---|---:|---:|---:|---:|---:|
| Final `vision_embeds` | `[512,1024]` | 0.9999937643 | 0.03125 | 0.00219265 | 0.9999790192 |
| DeepStack 0 | `[512,1024]` | 0.9999948885 | 0.03125 | 0.00213711 | 0.9999885559 |
| DeepStack 1 | `[512,1024]` | 0.9999930732 | 0.03125 | 0.00210220 | 0.9999923706 |
| DeepStack 2 | `[512,1024]` | 0.9999935002 | 0.03125 | 0.00207216 | 0.9999828339 |

Non-close counts under `rtol=0.01`, `atol=0.01` were:

| Stream | Non-close values | Total values |
|---|---:|---:|
| Final | 11 | 524,288 |
| DeepStack 0 | 6 | 524,288 |
| DeepStack 1 | 4 | 524,288 |
| DeepStack 2 | 9 | 524,288 |
| **Total** | **30** | **2,097,152** |

Thus, the original `torch.allclose` failure represented approximately
0.00143% of all compared values. It was not evidence that the CPU path
produced a different visual representation.

### Downstream response behavior

| Metric | Observed | Gate |
|---|---:|---:|
| Mean full-vocabulary KL | `1.6458e-8` | `<= 1e-3` |
| Maximum per-sample KL | `2.1526e-8` | `<= 1e-2` |
| Top-token match rate | `1.0` | `1.0` |
| Maximum `P(A)` delta | `7.6924e-7` | `<= 1e-2` |
| Maximum `P(B)` delta | `4.6328e-7` | `<= 1e-2` |

Every sample retained the same top action token. The observed feature drift
was attenuated rather than amplified by the language model.

## Key findings

### 1. There was numerical misalignment, but no evidence of semantic misalignment

The CPU and GPU features were not bitwise identical, and exact elementwise
`allclose` was false for every aggregated stream. However:

- stream count, names, order, and shapes were correct;
- all DeepStack streams independently had extremely high cosine similarity;
- only 30 of about 2.1 million values missed the elementwise tolerance;
- the maximum error was `0.03125`, consistent with small BF16-scale
  cross-kernel differences; and
- full-vocabulary response behavior was effectively unchanged.

The correct conclusion is not “CPU and GPU are identical.” It is:

> The observed CPU/GPU differences are numerically measurable but
> behaviorally negligible for the tested checkpoint, images, and software
> stack.

### 2. Parity must be layered

The useful parity hierarchy is:

```text
structural parity
  stream identity/order/shape/grid/revision
          |
          v
representation parity
  cosine/error distribution/outliers
          |
          v
behavioral parity
  full logits/KL/top token/action probabilities
          |
          v
live-system parity
  SGLang/Megatron/policy version/sampled tokens
```

No single layer substitutes for the others:

- cosine alone can hide a localized large error;
- elementwise allclose alone is too brittle for BF16 across devices;
- top-token agreement alone can hide large probability changes;
- downstream logit agreement alone cannot prove that every DeepStack stream
  was correctly routed; and
- local Hugging Face parity cannot prove the SGLang/Megatron integration.

### 3. A threshold change requires evidence and a counter-test

The tolerance was not relaxed merely because the run failed. Before changing
it, the diagnostic measured:

- exact number and fraction of failing values;
- maximum and mean feature error;
- per-stream cosine;
- full-vocabulary KL;
- top-token agreement; and
- task-action probability deltas.

The new gate also added an independent maximum-error bound and a regression
test that rejects a sparse large outlier. This preserves sensitivity to real
corruption.

### 4. Versioned APIs are part of parity

The missing `mm_token_type_ids` argument prevented the language-only path from
running at all. A feature representation can be numerically correct while its
consumer adapter is incompatible with the installed Transformers version.
Parity validation must cover both tensor production and the exact consumer
API.

### 5. Interpreter identity is part of reproducibility

The interactive shell had an unrelated `engram-vit/.venv/bin` before the
activated Conda environment in `PATH`. A bare `python` therefore lacked
Transformers even after `conda activate`. The successful run used the exact
Relax interpreter path.

Future artifacts should record `sys.executable` in addition to package
versions. Environment names alone are insufficient when nested activation or
inherited `PATH` state exists.

## What failed, and what did not

| Observation | Classification | Reason |
|---|---|---|
| Missing `mm_token_type_ids` | Real compatibility failure | Installed Transformers API required the argument |
| Exact stream-wide `torch.allclose=False` | Overly brittle acceptance rule | 30 sparse BF16-scale deviations rejected 2.1 million otherwise close values |
| Fast image-processor warning | Not a parity failure in this experiment | Both CPU and GPU paths consumed the same processed tensor |
| Temporary ROCm kernel-cache warning | Not a parity failure | Execution and comparison completed; cache write was optional |
| Bare `python` imported the wrong environment | Reproducibility hazard | An unrelated virtual environment preceded Conda in `PATH` |
| CPU eager versus GPU SDPA | Expected cross-kernel variation to monitor | Changing the CPU interface did not materially improve feature or logit metrics |

## Decisions

1. Keep CPU features in BF16; the result does not justify doubling cache and
   transport size with FP32.
2. Keep the current CPU backend rather than changing attention solely to chase
   bitwise equivalence.
3. Preserve exact structural validation for stream names, order, shapes, grid,
   and DeepStack count.
4. Use close fraction plus maximum absolute error, not close fraction alone.
5. Preserve full-vocabulary response checks for the local Hugging Face gate.
6. Do not describe this artifact as SGLang/Megatron parity.
7. The live GPU-resident two-cycle Relax path has now succeeded. Keep
   GPU-weight omission as a separate measured gate rather than inferring it
   from resident-mode success.

## Open questions

1. Do SGLang and Megatron consume the same feature bundle and produce matching
   sampled-token log probabilities when policy version, prompt, and tokens are
   held fixed?
2. Does the live path preserve the same final and DeepStack routing after the
   GPU visual modules are omitted?
3. Are the current feature thresholds stable across more than eight images,
   multiple MI210s, repeated runs, and future PyTorch/Transformers versions?
4. Does explicitly selecting the checkpoint's slow image processor change the
   accepted SFT or live Relax input distribution, even though it does not
   affect this same-input CPU/GPU comparison?
5. Should parity failures write a versioned `passed=false` artifact before
   raising, so evidence is not lost on failed gates?
6. Should parity metadata include Python executable, Torch/ROCm/Transformers
   versions, processor implementation, attention implementation, and tensor
   dtypes?
7. How much larger is the live SGLang-versus-Megatron mismatch than the local
   Hugging Face CPU-versus-GPU mismatch?

## Next steps

- [x] Run the two-cycle, four-GPU Relax smoke with CPU vision enabled and GPU
      visual weights still resident. See
      `training_reports/2026-07-30-qwen3-vl-cpu-vision-two-cycle-smoke.md`.
- [ ] Hold `feature_id`, `vision_revision`, policy version, prompt, and sampled
      tokens fixed while comparing SGLang and Megatron diagnostics.
- [ ] Repeat the two-cycle smoke with GPU visual weights omitted only after the
      resident path passes.
- [ ] Extend parity artifacts with runtime/backend metadata and write a failed
      artifact before raising.
- [ ] Run a larger calibration set before treating the current feature
      thresholds as portable beyond this checkpoint and node class.
- [ ] Compare native GPU vision, CPU-resident vision, and CPU-omitted vision
      for VRAM and end-to-end throughput.

## Reusable knowledge proposal

This result belongs naturally in the existing `model-integration` skill rather
than a new one-off skill. A future update should add a “precomputed multimodal
parity ladder” to its validation checklist:

1. structural final/DeepStack contract;
2. cross-device representation distribution;
3. downstream full-logit behavior;
4. live rollout/training sampled-token behavior; and
5. omission-mode repeat.

Per the retrospective workflow, that skill should not be modified without
explicit user approval. A troubleshooting entry for the Transformers 5.3
`mm_token_type_ids` signature change and contaminated Python-path symptom is
also proposed but not written yet.
