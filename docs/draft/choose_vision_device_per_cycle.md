# Choose the vision-encoder device for each rollout cycle

## Scope

This design chooses whether a frozen Qwen3-VL vision encoder runs on the CPU or GPU for each rollout cycle. The
choice is computed before training from measured device times and a complete cycle-workload file. It is therefore
an **automatic device choice from a fixed plan**, not online adaptation.

The mechanism is separate from asynchronous policy-weight updates. Vision features are reusable because the vision
encoder and projection are frozen. Policy staleness does not apply to that frozen computation, although dataset
identity, processor revision, model revision, feature schema, and image grid must still match exactly.

## Runtime modes

- `--vision-encoder-device=gpu` uses GPU vision for a fixed-device run.
- `--vision-encoder-device=cpu` uses the CPU `VisionEncoder` service for a fixed-device run.
- `--vision-device-mode=automatic` loads `--vision-device-plan` and chooses CPU or GPU once per rollout cycle.

Automatic choice requires both implementations to remain available. Startup rejects it when the GPU vision encoder
is skipped, the vision encoder or projection is trainable, all dataset features are preloaded, partial rollout is
enabled, staleness is nonzero, rollout shuffling changes cycle identity, dynamic filtering is enabled, or the plan is
missing or invalid.

## Plan semantics

For cycle workload

$$
x = (1, N_{images}, N_{visual\ tokens}, B_{features}),
$$

the plan contains separate nonnegative linear time estimates for GPU and CPU vision:

$$
T_d(x) = \beta_{d,0} + \beta_{d,image}N_{images}
+ \beta_{d,token}N_{visual\ tokens} + \beta_{d,byte}B_{features}.
$$

CPU work can overlap other work in the cycle. Its predicted exposed time is

$$
T_{CPU,after\ overlap} = \max(0, T_{CPU} - T_{overlap}).
$$

The predicted device-time gap is

$$
G = \frac{T_{GPU} - T_{CPU,after\ overlap}}
{\max(T_{GPU}, T_{CPU,after\ overlap}, \epsilon)}.
$$

For minimum gap $\delta$:

- choose CPU when $G > \delta$;
- choose GPU when $G < -\delta$;
- otherwise keep the previous cycle's device.

The final rule is described directly as “keep the previous device within the minimum gap.” It is not reported
as statistical confidence.

## Public interfaces

The version-2 plan contains:

- `initial_device`;
- `minimum_gap`;
- `time_models.gpu` and `time_models.cpu` coefficient maps;
- one workload record for every rollout cycle;
- timing-model fit diagnostics and input provenance when produced by the plan builder.

The runtime emits one `VISION_DEVICE_CHOICE` record per cycle and the corresponding W&B metrics:

- selected `cpu` or `gpu` device;
- image, visual-token, feature-byte, and overlap values;
- predicted GPU, CPU, and CPU seconds after overlap;
- predicted device-time gap and minimum gap;
- the direct reason: `cpu_predicted_faster`, `gpu_predicted_faster`, or `keep_previous_device`.

The implementation rejects missing cycles. It does not silently reuse the final known choice or infer an unseen
workload.

## Build the plan

1. Collect JSONL timing rows for both devices with the fields `device`, `image_count`, `visual_tokens`,
   `feature_bytes`, and `seconds`.
2. Build a deterministic JSON cycle-workload map without consuming the training dataset.
3. Fit the plan:

```bash
python -m examples.visual_xor.build_vision_device_plan \
  --measurements /absolute/path/vision_timing_measurements.jsonl \
  --cycles /absolute/path/cycle_workloads.json \
  --minimum-gap 0.05 \
  --initial-device gpu \
  --output /absolute/path/vision_device_plan.json
```

4. Run the counterbalanced comparison:

```bash
VISION_DEVICE_PLAN=/absolute/path/vision_device_plan.json \
bash scripts/debug/qwen3_vl_compare_vision_devices.sh
```

The runner executes fixed GPU, fixed CPU, and automatic runs with matched seeds. It requires five repeats,
complete cycles 0–23, 64 accepted responses, 32 generation requests, valid-action rate `1.0`, and both device choices
within every automatic run.

## Evaluation names and claims

For each matched cycle, the analysis reports:

- `selected_device`;
- `faster_device`, derived from fixed CPU and fixed GPU observations;
- `device_time_gap` and `minimum_gap`;
- `over_minimum_gap`;
- whether the selected device matches the faster device; and
- `extra_time`, automatic mode's relative additional time versus the faster device.

Aggregate acceptance requires a match rate of at least `0.90` among cycles over the minimum gap, mean extra-time
fraction below `0.05`, and at least half of measured cycles over the minimum gap. These thresholds are engineering
criteria, not confidence statements.

The paper should distinguish three claims:

1. CPU vision can improve end-to-end training-cycle time under a fixed workload.
2. A full vision-device plan can select between two measured implementations without changing task validity.
3. Online workload observation and online plan updates are future work until implemented and evaluated.

Geo3K currently supports fixed `gpu`, `cpu`, and `cpu_skip_gpu_encoder` modes. Its `automatic` mode fails immediately
because real-task workload observation is not yet implemented. That rejection is part of the correctness contract.

Canonical terminology and replaced names are listed in [the naming guide](../naming.md).
