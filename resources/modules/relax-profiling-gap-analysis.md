# Relax Profiling Gap Analysis Runbook

This runbook is for learning how to profile Relax without blindly adding
profilers everywhere. The immediate goal is to identify where time is going in
the ROCm Megatron + SGLang training path; the longer-term goal is to make small
profiling code changes with clear intent.

It is not a profiler implementation. It tells you what to measure first, where
the relevant code lives, what the existing signals mean, and which gaps in the
framework are worth instrumenting.

## 1. Keep The Runtime Shape Fixed

Before profiling, choose one stable run shape and keep it fixed while comparing
results. Changing rollout batch size, response length, TP size, checkpoint
interval, or SGLang cache limits changes the workload.

Use the 0.5B mock for the first profiling experiments:

```bash
cd /vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron

MOCK_HIP_VISIBLE_DEVICES=0,1,2,3 \
MOCK_RAY_NUM_GPUS=4 \
MOCK_ACTOR_RESOURCE_GPUS=2 \
MOCK_ROLLOUT_RESOURCE_GPUS=2 \
MOCK_TENSOR_MODEL_PARALLEL_SIZE=2 \
MOCK_ENABLE_SEQUENCE_PARALLEL=1 \
MOCK_ROLLOUT_BATCH_SIZE=2 \
MOCK_N_SAMPLES_PER_PROMPT=4 \
MOCK_GLOBAL_BATCH_SIZE=4 \
MOCK_ROLLOUT_MAX_RESPONSE_LEN=128 \
NUM_ROLLOUT=60 \
SAVE_INTERVAL=20 \
CKPT_FORMAT=torch_dist \
NO_SAVE_OPTIM=0 \
WANDB_GROUP="profile-qwen3-mock-0.5b-$(date +%Y%m%d_%H%M%S)" \
./amd_qwen3_mock_2gpu_e2e.sh
```

Use Qwen3-4B only after the profiling method is already understood:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
RAY_NUM_GPUS=4 \
NUM_GPUS_PER_NODE=4 \
ACTOR_RESOURCE_GPUS=2 \
ROLLOUT_RESOURCE_GPUS=2 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
ENABLE_SEQUENCE_PARALLEL=1 \
GPU_LABEL=4gpu-tp2 \
NUM_ROLLOUT=60 \
SAVE_INTERVAL=20 \
CKPT_FORMAT=torch_dist \
NO_SAVE_OPTIM=0 \
WANDB_GROUP="profile-qwen3-4b-$(date +%Y%m%d_%H%M%S)" \
./amd_qwen3_4b_2gpu_e2e.sh
```

Do not use CPU optimizer offload. Do not set `NO_SAVE_OPTIM=1` for checkpoint
validation runs.

## 2. Understand The Process Topology

Relax is not one process. A profiler attached to the wrong process answers the
wrong question.

| Layer | Process / actor | Main files | What to measure |
| --- | --- | --- | --- |
| Launcher | shell + Ray job driver | `amd_qwen3_4b_2gpu_e2e.sh` | Startup cost, env, Ray job command |
| Controller | Ray driver process | `relax/core/controller.py` | Service creation order, global loop |
| Metrics | Ray Serve replica | `relax/utils/metrics/service.py` | W&B/TensorBoard flush and timeline dump |
| Rollout service | Ray Serve replica | `relax/components/rollout.py` | Step scheduling and staleness waits |
| Rollout manager | Ray actor | `relax/distributed/ray/rollout.py`, `relax/engine/rollout/sglang_rollout.py` | Data sampling, SGLang generation, reward, transfer queue |
| SGLang engines | SGLang server workers | `relax/backends/sglang/sglang_engine.py` | Prefill/decode kernels, KV cache, weight update HTTP |
| Actor service | Ray Serve replica | `relax/components/actor.py` | Step loop and async training scheduling |
| Megatron actor ranks | Ray actors | `relax/backends/megatron/actor.py`, `relax/backends/megatron/model.py` | Data wait, forward/backward, optimizer, checkpoint, weight sync |

The critical synchronous ROCm path is:

```text
Rollout starts step N
  -> SGLang generates samples
  -> reward/filter converts samples into train_N
  -> Actor starts training step N
  -> Megatron actor waits for train_N from transfer queue
  -> forward/backward
  -> optimizer.step
  -> loss/metric logging
  -> optional checkpoint
  -> update weights into rollout engines
  -> next rollout step can use fresh weights
```

## 3. First Pass: Use Existing Signals, No Heavy Profiler

Start with logs and W&B metrics. This tells you which layer deserves a real
trace. A heavy profiler before this step usually creates more noise than
signal.

Useful log markers:

```bash
LOG=log/<run-log>.log

grep -E \
  "Start rollout|Finish rollout|Actor training step|Actor training completed|train_one_step|perf [0-9]+:|saving checkpoint|successfully saved checkpoint|update_weights|Reported .* metrics" \
  "$LOG"
```

Important W&B / metrics keys:

| Metric | Meaning |
| --- | --- |
| `perf/rollout_time` | End-to-end rollout generation time for a rollout step |
| `perf/tokens_per_gpu_per_sec` | SGLang rollout throughput, including framework overhead |
| `perf/longest_sample_tokens_per_sec` | Worst sample throughput; useful when one long response dominates |
| `perf/non_generation_time/*` | Time outside token generation when sample metadata records it |
| `perf/train_wait_time` | Actor time waiting outside the measured train region |
| `perf/train_time` | Actor train-side region after `train_wait` ends |
| `perf/wait_time_ratio` | Fraction of actor step time spent waiting |
| `perf/actor_train_time` | Megatron train compute region around `train()` |
| `perf/actor_train_tok_per_s` | Actor-side training token throughput |
| `train/entropy_loss`, `train/ppo_kl`, `train/loss` | Training sanity metrics |
| `rollout/raw_reward`, `rollout/acc/*`, `rollout/truncated_ratio` | Rollout/reward sanity metrics |

Interpretation:

| Symptom | First hypothesis | Next profiler |
| --- | --- | --- |
| `perf/wait_time_ratio` is high | Actor is waiting for rollout or transfer queue | Timeline/log timestamp pass |
| `perf/rollout_time` is high | SGLang decode, reward, data sampling, or transfer dominates | SGLang profile, rollout timers |
| `perf/actor_train_time` is high | Megatron forward/backward/optimizer dominates | PyTorch training profiler |
| Long gap after `Actor training completed` | Checkpoint, update_weights, eval trigger, or metrics flush | Add Timer scopes after training |
| Long gap around checkpoint boundary | `torch_dist` save is slow | Checkpoint-specific timers/logs |
| W&B only has system charts | Metrics service is not flushing app metrics | Metrics service logs, not GPU profiler |

## 4. Build A Per-Step Timeline From Logs

Before adding code, create a simple timeline table from one run:

```text
step
rollout_start_ts
rollout_finish_ts
actor_train_start_ts
forward_backward_start_ts
forward_backward_end_ts
optimizer_start_ts
optimizer_end_ts
actor_train_complete_ts
checkpoint_start_ts
checkpoint_end_ts
```

You can build it manually from logs at first. The point is to learn the shape:

```bash
grep -E \
  "Start rollout [0-9]+|Finish rollout [0-9]+|Actor training step [0-9]+|Actor training completed step [0-9]+|train_one_step rollout=|saving checkpoint at iteration|successfully saved checkpoint" \
  "$LOG"
```

For the current ROCm branch, `relax/backends/megatron/model.py` already logs
these train boundaries:

```text
train_one_step rollout=N step=0: starting forward_backward
train_one_step rollout=N step=0: finished forward_backward
train_one_step rollout=N step=0: starting optimizer.step
train_one_step rollout=N step=0: finished optimizer.step
train_one_step rollout=N step=0: scheduler step completed
train_one_step rollout=N step=0: reducing losses
train_one_step rollout=N step=0: loss reduction completed
```

If the wall-clock gap is visible in this table, add the next profiler only to
that region. Do not turn on every profiler at once.

## 5. Existing Profiling Tools

### 5.1 Framework Timeline Trace

Relax has a lightweight timeline path built around `Timer`:

- Timer implementation: `relax/utils/timer.py`
- Metrics adapter: `relax/utils/metrics/metrics_service_adapter.py`
- Trace writer: `relax/utils/metrics/timeline_trace.py`
- CLI flag: `--timeline-dump-dir`

When enabled, `Timer` events become Chrome trace events and are dumped as:

```text
<timeline_dump_dir>/timeline_step_<step>.json
```

View with:

```text
chrome://tracing
https://ui.perfetto.dev/
```

Current gap: the AMD launcher does not expose `--timeline-dump-dir` through an
environment variable yet. The first small profiling code task should be a
launcher passthrough, for example:

```bash
RELAX_TIMELINE_DUMP_DIR="${RELAX_TIMELINE_DUMP_DIR:-}"
if [ -n "${RELAX_TIMELINE_DUMP_DIR}" ]; then
    DEBUG_ARGS+=(--timeline-dump-dir "${RELAX_TIMELINE_DUMP_DIR}")
fi
```

Then run:

```bash
RELAX_TIMELINE_DUMP_DIR=/tmp/relax-timeline \
NUM_ROLLOUT=5 \
SAVE_INTERVAL=20 \
CKPT_FORMAT=torch_dist \
NO_SAVE_OPTIM=0 \
./amd_qwen3_mock_2gpu_e2e.sh
```

Use this profiler to answer framework questions:

- Is actor waiting for rollout?
- Is rollout waiting for data source?
- Is checkpoint/update_weights hidden after metrics logging?
- Are timers attributed to the right step?

Important caveat: `Timer` records are process-local and are shipped to the
metrics service only when that process logs metrics. In `train_actor`,
`log_perf_data()` happens before `save_model()` and `update_weights()`. That
means existing `save_model` or `update_weights` timers can be delayed or
mis-attributed unless you explicitly log or flush timeline events after those
regions. This is a real framework observability gap worth fixing.

### 5.2 Training PyTorch Profiler

Training profiler code lives in `relax/utils/profile_utils.py`. It wraps:

- `train_overall`
- `train_actor`
- `train_log_probs`

Relevant flags:

```bash
--use-pytorch-profiler
--profile-target train_overall
--profile-step-start 2
--profile-step-end 4
--profile-with-stack
--profile-with-memory
--profile-with-flops
--tb-experiment-name my-profiling-run
```

Output:

```text
traces/<tb_experiment_name>/train_trace/
```

Step semantics:

- `--profile-step-start` and `--profile-step-end` are inclusive.
- They are offsets from the current training launch, not absolute rollout IDs.
- After resume, the profiler counter starts from the new process run.

Use this only after logs show actor compute is the likely bottleneck. For ROCm
Qwen3-4B, start with a narrow window such as steps 2-3. Stack and memory
recording add overhead.

Current gap: the AMD launcher also does not expose these flags through a clean
environment variable. Add a launcher passthrough before using this repeatedly.

### 5.3 SGLang Profiler

SGLang profiler orchestration lives in `relax/utils/profile_utils.py` and is
called from `relax/engine/rollout/sglang_rollout.py` around rollout generation.

Relevant flags:

```bash
--sglang-profile
--sglang-profile-step-start 2
--sglang-profile-step-end 4
--sglang-profile-num-steps 3
--sglang-profile-output-dir /tmp/sglang-profile
--sglang-profile-with-stack
--sglang-profile-record-shapes
```

Output:

```text
traces/<tb_experiment_name>/sglang_trace/rollout_<rollout_id>/
```

Step semantics:

- SGLang step filters use absolute rollout IDs.
- `--sglang-profile-steps 3 10 50` takes precedence over start/end.

Use this when `perf/rollout_time` or SGLang server logs show decode/prefill is
the gap.

### 5.4 Memory Profiling

Relevant flags:

```bash
--record-memory-history
--memory-snapshot-num-steps 3
--memory-snapshot-dir /tmp/relax-memory
--memory-snapshot-path snapshot.pickle
--memory-recorder torch
```

Output:

```text
traces/<tb_experiment_name>/memory_snapshot/
```

Use memory snapshots for OOM, memory leaks, or allocator spikes. On ROCm, the
torch memory snapshot API may be unavailable depending on the installed
PyTorch; the code logs a warning and skips if the backend does not expose the
needed memory API. Treat that warning as a profiling-surface limitation, not as
a training failure.

## 6. Where To Add New Timers First

Start with framework-level timers, not kernel-level traces. Use `Timer` for
phase gaps and `torch.profiler.record_function()` only inside an already
identified hot compute region.

Good first Timer scopes:

| Question | Add timer around | File |
| --- | --- | --- |
| Is checkpoint save the post-train gap? | `self.save_model(...)` call site and inside `save_model()` | `relax/backends/megatron/actor.py` |
| Is weight sync the post-train gap? | `self.update_weights()` and `self.weight_updater.update_weights()` | `relax/backends/megatron/actor.py` |
| Is actor blocked on data? | `_get_data_from_transfer_queue(...)` and `all_consumed(...)` polling | `relax/backends/megatron/actor.py` |
| Is rollout blocked on sampling? | `data_source.get_samples.remote(...)` | `relax/engine/rollout/sglang_rollout.py` |
| Is rollout blocked on generation? | `state.submit_generate_tasks(...)` through completed `asyncio.wait(...)` | `relax/engine/rollout/sglang_rollout.py` |
| Is transfer queue slow? | `transfer_batch_to_data_system(...)` | `relax/utils/utils.py` |
| Is metrics flush slow? | `tracking_utils.log(...)`, `tracking_utils.flush_metrics(...)` | call sites in actor/rollout |

Timer naming rule:

```python
with timer("actor_update_weights"):
    self.update_weights()
```

For very high-frequency inner loops, use `keep=False` so the event appears in
the timeline but does not pollute W&B perf metrics:

```python
with timer("rollout_wait_first_completed", keep=False):
    done, remaining = await asyncio.wait(...)
```

If you add timers after the main `log_perf_data()` call, also add an explicit
metrics/timeline flush path for those post-train timers. Otherwise they may be
reported on the next step or never appear on the final step.

## 7. A Non-Blind Profiling Ladder

Follow this ladder. Move to the next level only when the previous one gives a
specific hypothesis.

### Level 0: Sanity

Run 5-10 steps and confirm:

```text
rollout starts and finishes
actor training starts and completes
W&B has application metrics
checkpoint settings are real if checkpoint is part of the test
```

### Level 1: Always-On Metrics

Run 30-60 steps without heavy profiler. Compare:

```text
perf/rollout_time
perf/train_wait_time
perf/train_time
perf/actor_train_time
perf/wait_time_ratio
perf/tokens_per_gpu_per_sec
```

Expected output: one sentence hypothesis, for example:

```text
Actor compute is only 8s, rollout is 22s, and wait ratio is low; focus on SGLang/reward path.
```

### Level 2: Log Timestamp Timeline

Extract start/end markers for 3 representative steps:

- one early non-checkpoint step;
- one checkpoint step, such as 19 when `SAVE_INTERVAL=20`;
- one later steady-state step.

Expected output: a table showing which phase owns the wall-clock gap.

### Level 3: Framework Timeline

Enable `--timeline-dump-dir` after adding launcher passthrough. Add missing
Timer scopes only around the suspected region.

Expected output: a Perfetto/Chrome trace showing the framework phase gap.

### Level 4: Targeted Torch / SGLang Profile

Use PyTorch profiler only for actor compute. Use SGLang profiler only for
rollout generation. Do not enable both unless you already know you need a
correlation trace.

Expected output:

- top kernels/operators by self time;
- CPU gaps between GPU kernels;
- communication time, if visible;
- memory allocation spikes, if memory profiling is enabled.

### Level 5: Implement The Fix And Re-Benchmark

Save a simple before/after result. At minimum record:

```text
git commit
run command
model target
GPU topology
NUM_ROLLOUT
SAVE_INTERVAL
mean rollout time
mean actor_train time
mean step time
tokens/sec
checkpoint time if relevant
W&B run id
log path
```

Do not compare a 2-GPU mock run against a 4-GPU Qwen3-4B run. The baseline and
candidate must use the same target and config.

## 8. What Gaps Are Already Suspect

These are not proven bottlenecks. They are good first places to investigate.

1. Post-train timer attribution
   - `log_perf_data()` runs before checkpoint and weight sync.
   - Current W&B perf metrics may not attribute `save_model` and
     `update_weights` to the step where they happen.
   - Fixing this would make framework profiling much easier.

2. Launcher profiler passthrough
   - The core CLI supports `--timeline-dump-dir`, `--use-pytorch-profiler`,
     and `--sglang-profile`.
   - The AMD launcher does not expose them through environment variables yet.
   - A clean passthrough is the first low-risk profiling-enablement change.

3. Rollout subphase visibility
   - Rollout has `perf/rollout_time` and token throughput, but the split
     between data sampling, SGLang generation, reward/filter, and transfer
     queue is not fully visible in W&B.
   - Add Timer scopes in `generate_rollout_async()` before jumping to kernel
     traces.

4. Weight sync visibility
   - Logs show SGLang pause/flush/update/continue requests.
   - W&B does not yet give a clean per-step `perf/update_weights_time` tied to
     the same rollout step.

5. Checkpoint visibility
   - Checkpoint logs are clear enough for correctness.
   - Per-rank checkpoint timing and bucket timing should be improved only if
     checkpoint steps dominate wall time.

## 9. First Coding Tasks When You Return

Do these in order.

1. Add launcher passthrough flags.
   - Add env vars such as `RELAX_TIMELINE_DUMP_DIR`,
     `RELAX_USE_PYTORCH_PROFILER`, `RELAX_SGLANG_PROFILE`, and
     `RELAX_PROFILE_STEP_START/END`.
   - Append corresponding CLI flags to `DEBUG_ARGS`, `PERF_ARGS`, or
     `SGLANG_ARGS` in `amd_qwen3_4b_2gpu_e2e.sh`.
   - Keep defaults off.

2. Fix post-train timeline flushing.
   - Decide where timers after `log_perf_data()` should be reported.
   - Make `save_model` and `update_weights` metrics/timeline events land on
     the rollout step that executed them.

3. Add rollout subphase timers.
   - Start in `generate_rollout_async()`.
   - Separate sample fetch, generation wait, reward/filter, and transfer queue.

4. Run the 0.5B mock 4-GPU baseline for 60 steps.
   - Keep `SAVE_INTERVAL=20` so step 19 includes checkpoint behavior.
   - Record W&B run id, log path, save dir, and latest checkpoint.

5. Only then profile Qwen3-4B.
   - Use the same profiling window and names.
   - Compare phase ratios, not just total wall time.

## 10. Quick Reference

Open traces:

```bash
tensorboard --logdir traces
```

or upload trace JSON/GZ files to:

```text
https://ui.perfetto.dev/
chrome://tracing
```

Check current Ray job:

```bash
python3 -m ray.scripts.scripts job list --address=http://127.0.0.1:8265
```

Tail logs:

```bash
tail -f log/amd-*.log
```

Check GPU process placement:

```bash
rocm-smi --showpid
```

Use `py-spy` when the gap looks CPU/Ray-side rather than GPU-side:

```bash
py-spy dump --pid <pid>
```

Use the debug-hang skill if Ray looks alive but the actor or rollout stops
making step progress.
