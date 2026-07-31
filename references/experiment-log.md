# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-07-27 - Capture Ray stale-live and Slurm/tmux isolation lessons

**Type:** Retrospective
**General description:** The approved battlefield follow-up turned two
operational failures into reusable triage guidance without adding another
active result skill.

### Details

- Extended `ray-stale-live-state-triage` so distributed actor health requires
  every expected rank plus fresh completed-step progress. A surviving DP rank
  is no longer treated as proof that training is alive.
- Documented the causal chain from actor failure through an uncleared
  `train_<n>` partition to `MAX_STALENESS` rollout backpressure.
- Added global ROCm GPU ownership checks for cases where device-wide memory is
  much larger than the failing rank's PyTorch allocation.
- Added a troubleshooting entry explaining why the default tmux server can
  cross Slurm allocations and how to isolate each job with
  `tmux -L "slurm-${SLURM_JOB_ID}"`.
- Kept the active skill set lean by leaving visual-XOR benchmark results in the
  consolidated training report instead of creating a new result skill.

### Key Points

- Ray `RUNNING`, health traffic, and one surviving actor rank are all
  insufficient health signals.
- Shell and tmux-server Slurm cgroups must match before a GPU workload starts.

### Links

- Report:
  `training_reports/2026-07-26-relax-synthetic-rl-battlefield.md`
- Skill: `ray-stale-live-state-triage`
- Troubleshooting: `references/troubleshooting.md`

## 2026-07-26 - Retrospective on the synthetic text and visual RL battlefield

**Type:** Retrospective
**General description:** The ROCm Relax stack now has validated text multitask
and image-conditioned learning baselines in both synchronous and fully
asynchronous execution, so the next research phase can focus on async stability
rather than basic reward or training plumbing.

### Details

- The tied, random-initialized joint EOS/two-action-bandit SFT preserved both
  intended choices, and a 200-cycle fully-async run solved both objectives:
  immediate EOS reached 100% and bandit action A reached 100%.
- The random-initialized 371.4M Qwen3-VL refinement SFT passed held-out,
  permuted-image, constant-image, and counterfactual gates at a 68.16%
  held-out XOR baseline.
- A synchronous visual refinement run improved held-out reward from 0.6582 to
  0.9492 by step 231 while keeping permuted and constant controls near chance.
- A clean DP2 fully-async run completed 250 actor cycles and 500 optimizer
  steps. It passed the full visual convergence gate near step 147, then
  regressed to 0.8652 held-out reward by the last recorded evaluation. This
  makes policy-version lag, update geometry, and best-state selection the next
  controlled questions.
- The earlier DP2 step-0 RCCL OOM was accompanied by abnormal GPU-wide
  occupancy and became a stale-live Ray job after actor death. A later clean
  run proved the same topology fits and trains.
- Different scheduler/request IDs did not prevent a GPU collision because one
  default tmux server was rooted in an older Slurm cgroup. Future jobs must use
  a unique tmux server label such as `tmux -L "slurm-${SLURM_JOB_ID}"`.
- Standalone SGLang and vLLM startup showed that model loading takes only a few
  seconds; most of the original 6:40 Relax startup is orchestration, process
  construction, warmup, Megatron initialization, and weight wiring.

### Key Points

- Basic text reward routing, multimodal reward routing, actor optimization,
  SGLang weight synchronization, and fully-async DP2 learning are now proven.
- Held-out combination metrics and anti-shortcut controls—not raw rollout mean
  reward—define success for visual-XOR.
- Keep checkpoints disabled for short plumbing smokes, but future async
  optimization studies need one bounded eval-best artifact because the final
  policy can be worse than the peak policy.
- Proposed follow-ups are a staleness/update/LR ablation, fatal actor failure
  propagation, and Slurm/tmux/GPU preflight checks.

### Links

- Report:
  `training_reports/2026-07-26-relax-synthetic-rl-battlefield.md`

## 2026-06-21 - Observation on mini colocate memory-saver diagnostic

**Type:** Observation
**General description:** The Qwen3-Mock-0.5B mini path reproduces the colocated
SGLang HIP prefill failure quickly and is now the preferred debug target for
this issue.

### Details

- Qwen3-Mock-0.5B sync colocate baseline run `2usrz00j` reproduced the same
  SGLang first-prefill HIP named-symbol failure with
  `enable_memory_saver=True`, so the colocate failure is not specific to 0.6B
  or 4B model size.
- Added `RELAX_SGLANG_DISABLE_MEMORY_SAVER=1` as an opt-in diagnostic that is
  forwarded through the AMD launcher into Ray runtime env. With this override,
  Qwen3-Mock-0.5B foreground validation run `kse80ff1` passed the old immediate
  failure point far enough to load SGLang weights and allocate KV cache with
  `enable_memory_saver=False`, but the 5-minute validation cutoff interrupted
  before first prefill/training could be observed.
- Follow-up no-memory-saver mini colocate runs reached rollout metrics. Run
  `tyhcen6i` completed rollout 0 with `rollout/reward/mean=-64.0`,
  `rollout/response_len/mean=64.0`, and `truncated_ratio=1.0`, then failed in
  actor step 0 with ROCm `HIP error: named symbol not found` during the first
  Megatron log-prob forward.
- The same mini model and reward succeeded in a non-colocated A/B control:
  W&B run `oc9gqu3i` / Ray job `raysubmit_3iJmqWyawsa16HLq` completed one
  rollout, one actor training step, weight update, and checkpoint save. This
  rules out the Qwen3-Mock-0.5B asset as the primary cause of the colocated
  actor crash.
- Added actor memory-saver paused-state bookkeeping and made sync
  `update_weights()` return offloaded actors to `sleep()` before rollout KV
  resumes. This fixed the earlier `Cannot resume allocation that is not paused`
  state error and verified GPU memory is released before rollout, but did not
  fix the colocated actor HIP named-symbol failure.
- Mini colocate run `frx2vwtb` with the sleep-after-weight-update patch, and
  run `6dq4eeav` with both the patch and `AMD_SERIALIZE_KERNEL=3`, both still
  completed rollout and then failed in actor log-prob forward at the same
  ROCm `HIP error: named symbol not found` boundary. `AMD_SERIALIZE_KERNEL=3`
  did not move the reported stack closer to the real failing kernel on this
  stack.

## 2026-06-20 - Retrospective on completion-length reward sync and colocate probes

**Type:** Retrospective
**General description:** The toy completion-length reward path is useful for
reward/W&B plumbing, but the current blocking failures are now separated into a
colocated SGLang ROCm prefill crash and a non-colocated 4B actor optimizer OOM.

### What we tried

- Added a simple completion-length reward where the reward is
  `-len(completion_tokens)` and the optimal response is immediate EOS.
- Ran short sync and fully-async probes with small prompt batches, online W&B,
  `ROLLOUT_MAX_RESPONSE_LEN=64`, and higher rollout temperature to test whether
  reward improves away from the generation cap.
- Repeated sync probes across Qwen3-0.6B and Qwen3-4B, and across colocated
  versus non-colocated rollout/actor placement.
- Added `USE_COLLOCATE=1` support to the AMD Qwen3 launcher so the same
  environment can be rerun with colocated placement.

### Key findings

- The early Qwen3-0.6B sync run completed, but rollout reward stayed flat at
  `-64.0`, response length stayed at 64, and advantages/gradients collapsed to
  zero. That run proves plumbing more than learning quality.
- Qwen3-4B sync colocate loaded the real `Qwen/Qwen3-4B` weights in both
  Megatron and SGLang, then failed during first SGLang prefill with
  `torch.AcceleratorError: HIP error: named symbol not found`.
- Qwen3-4B sync non-colocate reached rollout and actor training, proving the
  4B SGLang path itself can prefill/generate on this stack. It then OOMed at
  the first actor `optimizer.step()` while allocating Adam state on MI210.
- Qwen3-0.6B sync colocate reproduced the same SGLang HIP
  `named symbol not found` failure after actor sleep/offload and SGLang server
  readiness. This makes the colocated ROCm runtime path the stronger suspect,
  not the 4B model weights.

### What failed

- Fully-async execution remains noisy for this toy setup because queue/storage
  pressure can dominate the learning signal.
- Sync learning with the earlier 0.6B configuration did not improve reward
  because all samples hit the max response length, producing no useful
  within-prompt reward variance.
- Colocated sync startup currently fails before rollout metrics on both 0.6B
  and 4B with the same HIP named-symbol signature.
- Non-colocated 4B separates that failure from model loading, but hits a
  separate actor optimizer memory ceiling at Adam state initialization.

### Open questions

- Whether the colocated HIP failure is triggered by SGLang memory saver,
  actor sleep/offload, Ray placement/process state, or a specific SGLang ROCm
  prefill kernel.
- Whether lower SGLang pressure plus `AMD_SERIALIZE_KERNEL=3` exposes a more
  precise failing kernel boundary for the colocated path.
- Whether the non-colocated 4B optimizer OOM should be addressed with a larger
  actor topology, shorter sequence settings, or the already validated TP2 GPU
  optimizer path rather than CPU optimizer offload.

### Reusable lessons captured

- Updated `skills/completion-length-reward-smoke-test/SKILL.md` with the
  Qwen3-4B colocate, Qwen3-4B non-colocate, and Qwen3-0.6B colocate probe
  outcomes.
- Added opt-in propagation for `AMD_SERIALIZE_KERNEL` from the AMD launcher to
  Ray runtime env and SGLang rollout engines so ROCm kernel serialization can
  be used in the next colocate diagnostic without affecting normal runs.
- Follow-up Qwen3-0.6B sync colocate run `jidwd60n` with reduced SGLang
  pressure and `AMD_SERIALIZE_KERNEL=3` reproduced the same first-prefill
  failure on both SGLang engines. The SGLang server reached ready state, then
  `schedule_batch.py:1510` raised `torch.AcceleratorError: HIP error: named
  symbol not found`; logs also warned that `AMD_SERIALIZE_KERNEL=3` was parsed
  as an invalid boolean-style env value.
- Candidate troubleshooting entry: colocated SGLang on ROCm can fail during
  first prefill with `HIP error: named symbol not found` even when the same
  model works non-colocated.

## 2026-05-31 - Retrospective on Qwen3-0.5B four-GPU ROCm overnight regression

**Type:** Retrospective
**General description:** The 0.5B mock Qwen3 path is now the standing fast
ROCm e2e regression for Relax/SGLang/Megatron integration, with real W&B
application metrics and optimizer-inclusive `torch_dist` checkpoints.

### What we tried

- Generated a Qwen3-compatible mock checkpoint at the requested mini-model
  scale instead of keeping the earlier 1M toy asset.
- Validated a two-GPU smoke path with one actor GPU, one rollout GPU,
  `TENSOR_MODEL_PARALLEL_SIZE=1`, `SAVE_INTERVAL=1`,
  `CKPT_FORMAT=torch_dist`, and `NO_SAVE_OPTIM=0`.
- Validated a four-GPU foreground path with two Megatron actor GPUs, two SGLang
  rollout GPUs, actor TP2, and sequence parallel.
- Launched the four-GPU overnight production shape from tmux with:
  `NUM_ROLLOUT=1000`, `SAVE_INTERVAL=20`, `CKPT_FORMAT=torch_dist`,
  `NO_SAVE_OPTIM=0`, `MOCK_ROLLOUT_BATCH_SIZE=2`,
  `MOCK_N_SAMPLES_PER_PROMPT=4`, `MOCK_GLOBAL_BATCH_SIZE=4`, and
  `MOCK_ROLLOUT_MAX_RESPONSE_LEN=128`.

### Key findings

- The standing mock asset is
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-Mock-0.5B`
  with `533,126,144` parameters and the real Qwen3 tokenizer.
- The four-GPU overnight run is tmux session `tmux-28`, Ray job
  `raysubmit_ixWfXAnuy6cjSAwc`, W&B run `jr6tid47`, and save directory
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-Mock-0.5B_mcore_4gpu-20260531_161442`.
- Both SGLang engines loaded the mock model with `mem usage=1.04 GB` and
  allocated `#tokens: 65536`, matching the smoke caps
  `SGLANG_MEM_FRACTION_STATIC=0.2`, `SGLANG_MAX_TOTAL_TOKENS=65536`, and
  `SGLANG_MAX_RUNNING_REQUESTS=128`.
- The production run crossed multiple real checkpoint boundaries: iterations
  19 and 39 were saved in `torch_dist` format, and the live tmux pane later
  showed training step 48 still running.
- Checkpoint metadata for `iter_0000019` and `iter_0000039` contained
  optimizer moment keys and fp32 parameter keys, and
  `dataset/global_dataset_state_dict_19.pt` plus
  `dataset/global_dataset_state_dict_39.pt` were present. This proves
  checkpointing stayed real; it was not a weights-only save and did not disable
  optimizer state.
- W&B application metrics were present in the primary run after the metrics
  service fixes. The system-only W&B failure mode is not acceptable evidence of
  a healthy training run.

### What failed

- The `Qwen3-Mock-1M` asset was too small to be the standing target. It can
  remain historical context, but the standing mock should be the 0.5B asset.
- Treating startup or step 0 as "overnight health" is too weak. With
  `SAVE_INTERVAL=20`, the first production checkpoint is zero-based iteration
  19, so the run is not checkpoint-proven until that boundary is crossed.
- Disabling checkpointing, disabling optimizer save, or moving optimizer state
  to CPU would invalidate this regression. The validated path keeps
  `CKPT_FORMAT=torch_dist` and `NO_SAVE_OPTIM=0`.
- The reward and loss values are not quality evidence. The mock model is
  random, so invalid generated answers, reward `-1.0`, collapsed advantages,
  and `train/loss=0.0` are expected.

### Open questions

- Let the overnight run continue when possible, then audit the latest
  checkpoint after manual stop or completion. The latest observed durable
  checkpoint during this retrospective is iteration 39.
- After a later checkpoint is available, run one short explicit resume from the
  latest save directory to verify continuation still starts at `latest + 1`
  without a step reset.

### Reusable lessons captured

- Added `skills/qwen3-mock-0-5b-rocm-e2e/SKILL.md` for the 0.5B mock ROCm
  e2e regression path.
- Updated `skills/registry.json` so future retrospective/advisor workflows
  discover the new result skill.
- Updated `references/troubleshooting.md` with the checkpoint-boundary pitfall
  for long mock runs.

## 2026-05-31 - Validate mock Qwen3-0.5B AMD e2e smoke path

**Type:** Observation
**General description:** The generated Qwen3-compatible smoke model was resized
from the rejected 1M scale to a 0.5B-class checkpoint, close to the
`Erland/mini-glm-moe` test-model size, while preserving the fast ROCm e2e path.

### Details

The mock path uses:

- `scripts/tools/create_mock_qwen3.py`
- `scripts/models/qwen3-mock.sh`
- `amd_qwen3_mock_2gpu_e2e.sh`
- model asset
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-Mock-0.5B`
- two logical Ray GPUs with one actor GPU and one rollout GPU
- `TENSOR_MODEL_PARALLEL_SIZE=1` and sequence parallel disabled
- `CKPT_FORMAT=torch_dist`, `NO_SAVE_OPTIM=0`, and `SAVE_INTERVAL=1` for the
  validation run

The generated model keeps Qwen3/HF compatibility and the real Qwen3 tokenizer,
but uses a 0.5B-class dense architecture:

```text
num_parameters=533,126,144
hidden_size=1024
intermediate_size=3072
num_hidden_layers=24
num_attention_heads=16
num_key_value_heads=8
head_dim=128
vocab_size=151936
```

The earlier `Qwen3-Mock-1M` / `Erland/mini-qwen3-1m` asset is historical only.
It proved the code path, but it is too small to be the standing mock model.

The 0.5B mock wrapper still caps SGLang's smoke-test memory use with
`SGLANG_MEM_FRACTION_STATIC=0.2`, `SGLANG_MAX_TOTAL_TOKENS=65536`, and
`SGLANG_MAX_RUNNING_REQUESTS=128`. In the successful run, SGLang loaded the
533M model with `mem usage=1.04 GB` and logged
`KV Cache is allocated. #tokens: 65536`.

The successful tmux run was:

```bash
NUM_ROLLOUT=2 SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 \
WANDB_GROUP="qwen3-mock-0.5b-tmux-20260531_154301" \
./amd_qwen3_mock_2gpu_e2e.sh
```

Evidence:

- tmux session: `tmux-27`
- Ray job: `raysubmit_i2PLzbjh1F48f5am`
- W&B run: `hd2ddkjo`
- log: `log/amd-qwen3-mock-0.5b-2gpu-20260531_154301.log`
- save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-Mock-0.5B_mcore_2gpu-20260531_154301`
- Hugging Face repo: `https://huggingface.co/Erland/mini-qwen3-0.5b`
- Hugging Face commit: recorded after upload

The job succeeded and completed both rollout/training steps:

```text
> number of parameters on (tensor, pipeline) model parallel rank (0, 0): 533126144
Actor training completed step 0/2
saving checkpoint at iteration       0 ... in torch_dist format
successfully saved checkpoint from iteration       0 ...
Actor training completed step 1/2
saving checkpoint at iteration       1 ... in torch_dist format
successfully saved checkpoint from iteration       1 ...
Job 'raysubmit_i2PLzbjh1F48f5am' succeeded
```

Checkpoint verification:

- `latest_checkpointed_iteration.txt` contains `1`.
- `iter_0000000/` and `iter_0000001/` each contain `.metadata`,
  two `.distcp` shards, `common.pt`, and `metadata.json`.
- Both `.metadata` files contain optimizer state keys, including
  `optimizer.state.exp_avg` and `optimizer.state.exp_avg_sq`.

### Key Points

- This is the current Qwen3 mock regression path for the ROCm Relax stack.
- The loss and rewards are not meaningful because the model is random;
  generated answers are invalid, rewards are all `-1.0`, advantages collapse to
  zero, and `train/loss` is `0.0`. That is expected for this smoke path.
- The run still proves the important infrastructure boundaries: HF asset load,
  SGLang transformers rollout, Megatron Qwen3Bridge import, distributed weight
  update, optimizer step, W&B application metrics, and optimizer-inclusive
  `torch_dist` checkpoint save.

## 2026-05-31 - Add and validate tiny Qwen3-0.6B AMD launcher

**Type:** Observation
**General description:** The Qwen3-0.6B AMD path now has a dedicated two-GPU
launcher that reuses the ROCm Megatron/SGLang fixes while keeping real
`torch_dist` checkpointing and optimizer-state save enabled.

### Details

The tiny launcher uses:

- `amd_qwen3_0_6b_2gpu_e2e.sh`
- `scripts/models/qwen3-0.6B.sh`
- `Qwen/Qwen3-0.6B` materialized under
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B`
- two logical Ray GPUs with one actor GPU and one rollout GPU
- `TENSOR_MODEL_PARALLEL_SIZE=1` and sequence parallel disabled
- `CKPT_FORMAT=torch_dist`, `NO_SAVE_OPTIM=0`, and production
  `SAVE_INTERVAL=20`

The first smoke gate proved the topology needed to override conda's inherited
`HIP_VISIBLE_DEVICES`. The wrapper now passes
`RELAX_HIP_VISIBLE_DEVICES_OVERRIDE=0,1` so the base launcher reapplies the
two-GPU topology after conda activation and `.env` loading.

The production tiny run is Ray job `raysubmit_qrTv36H5f7qiRdnG` in `tmux-25`,
with W&B run `sb6k87bn` and log
`log/amd-qwen3-0.6b-2gpu-20260531_090253.log`. It reached real training,
completed step 20, and continued to step 21. The first checkpoint boundary
completed successfully:

```text
saving checkpoint at iteration      19 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      19 ...
Actor training completed step 19/200
```

Checkpoint directory:

```text
/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B_mcore_2gpu-20260531_090253/iter_0000019
```

`latest_checkpointed_iteration.txt` contains `19`, and `.metadata` contains
optimizer state keys including `optimizer.state.exp_avg` and
`optimizer.state.exp_avg_sq`, confirming this is not a weights-only save.

### Key Points

- Tiny Qwen3 is training with W&B application metrics and real optimizer
  checkpoint state.
- The run is still active; this entry validates startup, training, and the
  first checkpoint boundary, not final 200-step completion.
- The base launcher is now model-parameterized so Qwen3-4B defaults are
  preserved while the tiny wrapper can select Qwen3-0.6B explicitly.

## 2026-05-31 - Retrospective on the no-offload TP2 ROCm e2e path

**Type:** Retrospective
**General description:** The viable AMD Qwen3-4B path is the four-GPU TP2
Megatron actor with GPU optimizer state, W&B application metrics, and real
`torch_dist` checkpoint save/resume; CPU optimizer offload and checkpoint
shortcuts are not acceptable for this objective.

### What we tried

- rejected CPU optimizer offload after the single-GPU actor OOM and the
  historical HybridDeviceOptimizer instability made offload a misleading fix;
- moved the AMD launcher to a four-GPU topology with two actor GPUs and two
  rollout GPUs:
  `HIP_VISIBLE_DEVICES=0,1,2,3 RAY_NUM_GPUS=4 ACTOR_RESOURCE_GPUS=2 ROLLOUT_RESOURCE_GPUS=2 TENSOR_MODEL_PARALLEL_SIZE=2`;
- enabled sequence parallel for the TP2 actor and patched the TE-less Megatron
  `WrappedTorchNorm` path on HIP so torch RMSNorm/LayerNorm can be used on
  sequence-parallel shards;
- kept checkpointing required with `CKPT_FORMAT=torch_dist`, `SAVE_INTERVAL=20`,
  and `NO_SAVE_OPTIM=0`;
- fixed W&B reporting so MetricsService joins the primary run and namespaced
  `train/step` / `rollout/step` metrics are not hidden in a system-only run;
- explicitly resumed by setting both `LOAD_DIR` and `SAVE_DIR` to the same
  checkpoint directory, keeping `SCHEDULER_RESUME_POLICY=strict` because the
  scheduler-driving settings did not change.

### Key findings

- The original long run reached step 100 and saved complete `torch_dist`
  checkpoints through iteration 99. Iterations 19, 39, 59, 79, and 99 all
  contained `.metadata`, four `.distcp` shards, `common.pt`, `metadata.json`,
  and optimizer state entries while `NO_SAVE_OPTIM=0`.
- The resume run loaded checkpoint iteration 99, loaded streaming dataset state
  at `epoch=0, position=404`, initialized the actor at step 100, saved newer
  checkpoints at iterations 119 and 139, and continued through completed step
  154 before manual stop at step 155.
- W&B application metrics were present after the metrics fixes. The resumed run
  reported values including `train/step=154`, `rollout/step=154`,
  `train/loss=0`, `train/entropy_loss=0.18966230750083923`, and
  `perf/step_time=157.73920893669128`.
- The latest durable checkpoint at the time of this retrospective is iteration
  139 with `dataset/global_dataset_state_dict_139.pt` present. That proves
  checkpoint save after resume, but not final 200-step completion.
- A fresh production resume from checkpoint 139 was launched as Ray job
  `raysubmit_npbaWGLVyyJMxhD8` in `tmux-24`, with W&B run `bkzsnt9k`. It loaded
  iteration 139, completed step 141, started step 142, emitted W&B application
  metrics for step 141, and was then manually stopped when the direction moved
  to a smaller Qwen3 bring-up.

### What failed

- CPU optimizer offload is the wrong path for this objective. It avoids one
  memory boundary but repeatedly introduces ROCm HDO crashes and violates the
  current "no cheating" constraint.
- Treating checkpointing as optional was wrong. The validation only matters if
  `torch_dist` checkpointing and optimizer-state save remain enabled.
- Treating a healthy partial run as completion was too weak. Stopping after step
  154 proves resume and continued training, but it does not prove the final
  checkpoint boundary for `NUM_ROLLOUT=200`.
- W&B "system only" views were misleading until MetricsService joined the
  primary run and namespaced step metrics were flushed promptly.

### Open questions

- The Qwen3-4B run still has not produced the final 200-step save boundary,
  likely iteration 199 with `SAVE_INTERVAL=20` and zero-based Megatron
  iteration numbering, because the active objective moved to tiny Qwen3 first.
- After the final checkpoint is present, a fresh resume from the latest
  checkpoint should be tested quickly to prove the final saved state can be
  used without a loss spike or step reset.

### Reusable lessons captured

- Added `skills/rocm-megatron-tp2-checkpoint-resume/SKILL.md` for the current
  no-CPU-offload TP2 GPU optimizer path.
- Updated `skills/registry.json` so future agents discover the new result
  skill.
- Updated the existing ROCm bring-up skill so its current AMD Qwen3 path no
  longer recommends CPU optimizer offload.

## 2026-05-30 - Reject ROCm CPU optimizer offload and validate TP2 GPU optimizer

**Type:** Observation
**General description:** The current AMD Qwen3-4B path no longer treats CPU
optimizer offload as an acceptable workaround; fitting the actor uses tensor
parallel GPU optimizer state instead.

### Details

The four-GPU overnight attempt first proved that a single MI210 actor rank with
GPU Adam cannot fit the first lazy Adam state allocation:

```text
torch.OutOfMemoryError ... state["exp_avg_sq"] = torch.zeros_like
```

CPU optimizer offload was explicitly rejected as a workaround. The AMD launcher
therefore removed `--optimizer-cpu-offload`,
`--use-torch-optimizer-for-cpu-offload`, precision-aware CPU-offload flags, and
the pinning overrides. `relax/backends/megatron/optimizer_utils.py` now only
normalizes unsupported single-DP-rank optimizer flags and raises immediately if
any ROCm Megatron training role requests `optimizer_cpu_offload=True`.

The active fit path is:

- four visible GPUs;
- two actor GPUs and two rollout GPUs;
- `--tensor-model-parallel-size 2`;
- `--sequence-parallel`;
- GPU Adam optimizer state;
- `NO_SAVE_OPTIM=0`, so checkpoint optimizer state remains enabled.

TP2 exposed a TE-less Megatron assertion:

```text
AssertionError: sequence parallel not supported by torch LayerNorm
```

Relax now patches Megatron's `WrappedTorchNorm` at runtime on HIP so Torch
RMSNorm/LayerNorm can be created on sequence-parallel shards and still mark
norm parameters with `sequence_parallel=True`.

Validation status from the active tmux run `tmux-22`:

```text
optimizer_cpu_offload ........................... False
Patched Megatron WrappedTorchNorm for ROCm sequence-parallel torch norms
Actor training step 0/200
train_one_step rollout=0 step=0: finished forward_backward
train_one_step rollout=0 step=0: finished optimizer.step (update_successful=True, ...)
train_one_step rollout=0 step=0: scheduler step completed
train_one_step rollout=0 step=0: loss reduction completed
step 0: {'train/loss': ..., 'train/step': 0}
```

By `2026-05-31 00:47:14 UTC`, the same run had completed actor step 20 and
moved to rollout/training step 21. The first save boundary completed at Megatron
iteration 19 because the training loop uses a zero-based iteration index with
`SAVE_INTERVAL=20`:

```text
saving checkpoint at iteration      19 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 1: buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      19 ...
Actor training completed step 19/200
Actor training completed step 20/200
```

The checkpoint directory contains `.metadata`, four `.distcp` shards,
`common.pt`, `metadata.json`, and `latest_checkpointed_iteration.txt`.
`latest_checkpointed_iteration.txt` contains `19`, and `.metadata` inspection
found optimizer entries while the launch kept `NO_SAVE_OPTIM=0`. This confirms
the checkpoint is not the weights-only path.

By `2026-05-31 01:47:28 UTC`, the same run had completed actor step 41, started
step 42, and passed the second save boundary:

```text
saving checkpoint at iteration      39 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 1: buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      39 ...
Actor training completed step 40/200
Actor training completed step 41/200
```

The save directory contains both `iter_0000019` and `iter_0000039`.
`latest_checkpointed_iteration.txt` contains `39`. The `iter_0000039/.metadata`
file contains optimizer state keys such as
`optimizer.state.exp_avg.embedding.word_embeddings.weight` and
`optimizer.state.exp_avg_sq.embedding.word_embeddings.weight`, confirming again
that optimizer checkpoint state is enabled. W&B API inspection for run
`gult6g9a` showed application metrics, not only system metrics:

```text
train/step=40
rollout/step=41
perf/step_time=349.7924120426178
```

By `2026-05-31 02:41:04 UTC`, the same run had completed the third save
boundary at iteration 59 and continued through step 61:

```text
saving checkpoint at iteration      59 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 1: buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      59 ...
Actor training completed step 59/200
Actor training completed step 60/200
Actor training completed step 61/200
```

The save directory now contains `iter_0000019`, `iter_0000039`, and
`iter_0000059`, and `latest_checkpointed_iteration.txt` contains `59`.
Inspection of `iter_0000059/.metadata` again found optimizer state entries
including `optimizer.state.exp_avg.*` and `optimizer.state.exp_avg_sq.*`. W&B
API inspection showed the run still active with application metrics through:

```text
train/step=61
rollout/step=61
perf/step_time rows=62
```

By `2026-05-31 03:35:13 UTC`, the same run had completed the fourth save
boundary at iteration 79 and continued through step 80 into step 81:

```text
saving checkpoint at iteration      79 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 1: buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      79 ...
Actor training completed step 79/200
Actor training completed step 80/200
Actor training step 81/200
```

The save directory now contains `iter_0000019`, `iter_0000039`,
`iter_0000059`, and `iter_0000079`; `latest_checkpointed_iteration.txt`
contains `79`. Inspection of `iter_0000079/.metadata` found 30 optimizer
metadata entries, including `optimizer.state.exp_avg.*`,
`optimizer.state.exp_avg_sq.*`, and `optimizer.state.fp32_param.*`, plus RNG
state entries. W&B API inspection showed the run still active with application
metrics through:

```text
train/step=80
rollout/step=80
perf/step_time rows=81
```

By `2026-05-31 04:31:46 UTC`, the same run had crossed the old step-99
checkpoint failure window. Iteration 99 saved successfully and the actor
continued to step 100:

```text
saving checkpoint at iteration      99 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 1: buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration      99 ...
Actor training completed step 99/200
Actor training step 100/200
```

The save directory now contains a complete `iter_0000099` checkpoint with
`.metadata`, four `.distcp` shards, `common.pt`, and `metadata.json`;
`latest_checkpointed_iteration.txt` contains `99`. Inspection of
`iter_0000099/.metadata` found optimizer state entries, fp32 optimizer
parameter entries, and RNG entries. W&B API inspection showed the run still
active with application metrics through:

```text
train/step=99
rollout/step=99
perf/step_time rows=100
```

The host TensorBackuper path is separate from optimizer offload. It keeps actor
and ref weight snapshots for the colocated actor/ref workflow. The ROCm guard
only disables pinned host memory for those snapshots; it does not move optimizer
state or Adam updates to CPU. That helper now lives in
`relax/backends/megatron/weight_backup_utils.py` so `optimizer_utils.py` stays
optimizer-only.

## 2026-05-30 - Fix W&B namespaced metric flush after actor crash

**Type:** Observation
**General description:** The 4-GPU overnight run produced rollout metrics, but
they stayed buffered in MetricsService because the Megatron actor died before
the end-of-step flush.

### Details

The W&B run `s9z00ddu` initially appeared to contain only system metrics. The
log showed that application metrics were not missing:

```text
POST /metrics/log_metrics_batch 200
rollout 0: {...}
Actor training failed at step 0: ActorDiedError
```

The missing boundary was `/metrics/report_step`. Preserving namespaced step
metrics such as `rollout/step` and `train/step` fixed the earlier W&B x-axis
problem, but those metrics were still buffered until the actor called
`flush_metrics()` after completing the training step. In this run the actor
died before that flush, so W&B did not receive the buffered rollout metrics.

The metrics adapter now reports immediately for namespaced step keys ending in
`/step`. That keeps `train/step` and `rollout/step` in the payload for W&B
custom axes, while avoiding a system-only W&B run when a later actor crash
prevents the normal end-of-step flush.

When a reported payload already contains a namespaced step metric, MetricsService
logs it to W&B without forcing the global W&B `step=` argument. This avoids
dropping later same-step batches while still letting W&B charts use
`train/step` or `rollout/step` as their x-axis.

For the interrupted run, the buffered step-0 metrics were manually flushed:

```text
POST /metrics/report_step {"step": 0}
Reported 38 metrics for step 0
```

## 2026-05-30 - Launch 4-GPU overnight AMD Qwen3-4B run

**Type:** Plan
**General description:** Start a longer MI210 run that uses all four local GPUs
while preserving the validated single-rank Megatron actor/checkpoint path.

### Details

The AMD launcher now derives its Ray and Relax resource topology from
environment variables instead of hardcoding a two-GPU run:

- `HIP_VISIBLE_DEVICES` controls the visible GPU list and defaults to `0,1`.
- `RAY_NUM_GPUS` defaults to the number of visible GPUs.
- `NUM_GPUS_PER_NODE` defaults to `RAY_NUM_GPUS`.
- `ACTOR_RESOURCE_GPUS` defaults to `1`.
- `ROLLOUT_RESOURCE_GPUS` defaults to `RAY_NUM_GPUS - ACTOR_RESOURCE_GPUS`.
- `RESOURCE_JSON` defaults to `{"actor": [1, ACTOR_RESOURCE_GPUS], "rollout": [1, ROLLOUT_RESOURCE_GPUS]}`.

The overnight run uses four visible MI210 GPUs with one GPU reserved for the
single-rank Megatron actor and three GPUs reserved for rollout engines. This
uses all four GPUs without changing the actor path that validated ROCm
`torch_dist` checkpoint save/resume with optimizer state enabled.

Launch metadata:

- tmux session: `tmux-17`
- Ray job: `raysubmit_PEf7qZPy4MwWgBpt`
- W&B run: `https://wandb.ai/erlandpg/relax-amd/runs/s9z00ddu`
- Save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_4gpu-overnight-20260530_205124`
- Log:
  `log/amd-qwen3-4b-4gpu-20260530_205125.log`
- Args:
  `HIP_VISIBLE_DEVICES=0,1,2,3 RAY_NUM_GPUS=4 NUM_GPUS_PER_NODE=4 ACTOR_RESOURCE_GPUS=1 ROLLOUT_RESOURCE_GPUS=3 SAVE_INTERVAL=20 NUM_ROLLOUT=200`

Checkpointing remains enabled. The launch uses `CKPT_FORMAT=torch_dist`, omits
`NO_SAVE_OPTIM`, and therefore keeps optimizer checkpoint save enabled through
the launcher default `NO_SAVE_OPTIM=0`. `SAVE_INTERVAL=20` means the first
overnight checkpoint is expected at iteration 20, not every step.

## 2026-05-30 - Fix W&B MetricsService run split

**Type:** Observation
**General description:** The post-resume validation run uploaded system/runtime
data to the primary W&B run, but user-visible train/rollout charts were hidden
in a separate MetricsService run.

### Details

The primary W&B link for the checkpoint validation was run `5fw7cmi1`. The log
showed training metrics did reach `MetricsService`:

```text
POST /metrics/log_metrics_batch 200
step 2: {'train/loss': ..., 'train/step': 2}
Reported 71 metrics for step 2
```

The same log showed `MetricsService` initialized a new run `kbriia47` named
`metrics-service`, while the primary process, rollout manager, and Megatron
actor resumed run `5fw7cmi1`. That split explains why the primary run appeared
to contain only system metrics.

`MetricsService._init_wandb()` now uses the shared secondary W&B initializer
whenever `config.wandb_run_id` is present. That keeps aggregated metrics on the
primary run id and avoids hiding train/rollout metrics in a side run.
MetricsService also finishes W&B during replica shutdown so queued metrics are
flushed before Ray Serve tears down the process.

## 2026-05-30 - Validate post-resume ROCm `torch_dist` train and checkpoint

**Type:** Validation
**General description:** A resumed MI210 run loaded a `torch_dist` checkpoint
with optimizer state, trained one additional rollout step, and saved a new
`torch_dist` checkpoint with optimizer state still enabled.

### Details

The first explicit resume validation used `NUM_ROLLOUT=2`, matching the source
checkpoint. That proved optimizer restore and step restoration, but because the
checkpoint already ended at iteration `1`, the actor started at step `2` and
finished immediately. A stronger validation intentionally extended the run to
`NUM_ROLLOUT=3` so the actor had to train and checkpoint step `2`.

Changing `NUM_ROLLOUT` also changes Megatron's optimizer scheduler horizon.
The first `NUM_ROLLOUT=3` attempt correctly failed loud:

```text
OptimizerParamScheduler: class input value 48 and checkpointvalue 32 for total number of iterations do not match
```

The AMD launcher now exposes `SCHEDULER_RESUME_POLICY`:

- `strict` keeps Megatron's default mismatch check;
- `override` appends `--override-opt-param-scheduler` for deliberate
  continuation with a new schedule horizon;
- `checkpoint` appends `--use-checkpoint-opt-param-scheduler` to keep the old
  checkpoint scheduler values.

Validation run:

- tmux session: `tmux-16`
- Ray job: `raysubmit_tD6jkVCbk6t7cZ21`
- W&B run: `https://wandb.ai/erlandpg/relax-amd/runs/5fw7cmi1`
- Load directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-live-torch-dist-lazyfix-20260530_112514`
- Save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-resume-train-override-20260530_135915`
- Args:
  `LOAD_DIR=<source> SAVE_DIR=<new> SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=3 SCHEDULER_RESUME_POLICY=override`
- Log:
  `validation_logs/live_torch_dist_resume_train_override_tmux_20260530_135915.log`

The run passed the full post-resume train/save path:

```text
Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore
loading distributed checkpoint from ... at iteration 1
checkpoint version 3.0
Actor initialized with starting step 2
Actor training step 2/3
saving checkpoint at iteration       2 ... in torch_dist format
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration       2
Actor training completed step 2/3
All training steps finished
Job 'raysubmit_tD6jkVCbk6t7cZ21' succeeded
TMUX_STATUS=0
```

The new save directory contains `iter_0000002/.metadata`, `__0_0.distcp`,
`__0_1.distcp`, `common.pt`, `metadata.json`, and
`latest_checkpointed_iteration.txt` with value `2`.

### Key Points

- The validated path now covers save, explicit load, optimizer restore,
  post-resume training, weight update, and a new `torch_dist` checkpoint save.
- Optimizer state was not disabled: the run used `NO_SAVE_OPTIM=0`.
- `SCHEDULER_RESUME_POLICY=override` is only needed when intentionally changing
  the scheduler horizon, such as extending a short checkpoint smoke from
  `NUM_ROLLOUT=2` to `NUM_ROLLOUT=3`.

## 2026-05-30 - Validate explicit ROCm `torch_dist` resume with optimizer state

**Type:** Validation
**General description:** An explicit `LOAD_DIR` resume from the live
`torch_dist` checkpoint completed on MI210 with Megatron optimizer state
enabled.

### Details

After the live save validation, the explicit resume path exposed two
ROCm/Megatron optimizer-restore failures:

1. Megatron wrapped the ROCm `HybridDeviceOptimizer` in `FP32Optimizer` with
   `init_state_fn=None`. During checkpoint load,
   `FP32Optimizer.sharded_state_dict(is_loading=True)` called that missing
   initializer and failed with `TypeError: 'NoneType' object is not callable`.
2. After installing an Adam-state initializer, the actor passed that boundary
   but died after `checkpoint version 3.0`. The failing phase was optimizer
   `load_state_dict`: PyTorch maps checkpoint optimizer state to the public HDO
   param groups, which are the live GPU model params, instead of the HDO inner
   CPU-offload params.

The Relax ROCm optimizer hook now installs both pieces for the single-rank HIP
actor CPU-offload path:

- a fail-loud Adam state initializer for `HybridDeviceOptimizer`, so Megatron
  can build the optimizer sharded state dict during load;
- an `FP32Optimizer.load_state_dict` override that loads checkpoint optimizer
  state onto HDO inner params, synchronizes sub-optimizer state, and restores
  the public HDO param groups.

Validation run:

- tmux session: `tmux-15`
- Ray job: `raysubmit_KiRTCAx7v8RCU9L6`
- W&B run: `https://wandb.ai/erlandpg/relax-amd/runs/aghbilif`
- Checkpoint directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-live-torch-dist-lazyfix-20260530_112514`
- Args:
  `LOAD_DIR=<checkpoint> SAVE_DIR=<checkpoint> SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=2`
- Log:
  `validation_logs/live_torch_dist_explicit_load_hdoload_tmux_20260530_132101.log`

The validation used the real optimizer-restore path and completed:

```text
Installed HybridDeviceOptimizer Adam-state initializer and load patch for ROCm checkpoint restore
loading distributed checkpoint from .../Qwen3-4B_mcore_2gpu-live-torch-dist-lazyfix-20260530_112514 at iteration 1
Initialized 290 HybridDeviceOptimizer Adam states for ROCm checkpoint restore
checkpoint version 3.0
Actor initialized with starting step 2
All training steps finished
Job 'raysubmit_KiRTCAx7v8RCU9L6' succeeded
```

Because the checkpoint's `latest_checkpointed_iteration.txt` points to
iteration `1` and the validation used `NUM_ROLLOUT=2`, starting at step `2`
is the expected resume behavior. No optimizer-save shortcut was used:
`NO_SAVE_OPTIM=0`, and the source checkpoint still contains optimizer state in
DCP metadata.

### Key Points

- `torch_dist` resume is now validated for the MI210 single-rank actor path
  with optimizer checkpoint state enabled.
- This is not a skip-optimizer workaround. The fix keeps optimizer state load
  enabled and patches the placement of restored HDO state.
- The old `common.pt` PyTorch 2.6 failure, the missing HDO `init_state_fn`
  failure, and the post-`checkpoint version 3.0` HDO load death are separate
  restore boundaries; all three are now covered by targeted Relax-side hooks.

## 2026-05-30 - Validate live ROCm `torch_dist` checkpoint save and explicit load knob

**Type:** Validation
**General description:** A live non-cached AMD Qwen3-4B two-GPU run completed
two rollout/train/checkpoint cycles with Megatron `torch_dist` checkpointing
and optimizer state enabled.

### Details

The previous cached-rollout smoke proved that the writer could finish one
checkpoint, but it did not prove that the full Relax/SGLang/Megatron path could
generate fresh rollouts, train, and checkpoint repeatedly. The follow-up live
run kept checkpointing required and kept optimizer state enabled:

- tmux session: `tmux-13`
- Ray job: `raysubmit_FeKYagrKwzrfrcPU`
- W&B run: `https://wandb.ai/erlandpg/relax-amd/runs/7yi7a38m`
- Save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-live-torch-dist-lazyfix-20260530_112514`
- Args: `SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=2`
- Log:
  `validation_logs/live_torch_dist_lazyfix_tmux_20260530_112514.log`

The live run completed both actor steps and both checkpoint saves:

```text
rollout 0: {... response_lengths: 768.0 ...}
step 0: {... 'train/step': 0}
saving checkpoint at iteration       0 ... in torch_dist format
ROCm lazy checkpoint prepare: write_items=1123, buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration       0
Actor training completed step 0/2
rollout 1: {... response_lengths: 768.0 ...}
step 1: {... 'train/step': 1}
saving checkpoint at iteration       1 ... in torch_dist format
ROCm lazy checkpoint prepare: write_items=1123, buckets=2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration       1
Actor training completed step 1/2
All training steps finished
Job 'raysubmit_FeKYagrKwzrfrcPU' succeeded
```

Each saved iteration contains DCP metadata, two `.distcp` shards, `common.pt`,
and `metadata.json`. `latest_checkpointed_iteration.txt` points to iteration
`1`. DCP metadata inspection reported 175 state-dict entries and 1123 storage
entries for both `iter_0000000` and `iter_0000001`, including
`embedding.word_embeddings.weight` and optimizer state such as
`optimizer.state.exp_avg.embedding.word_embeddings.weight`.

This validation also exposed a resume contract issue. Setting `SAVE_DIR` to an
existing checkpoint directory only changes the save target; it does not pass
Megatron `--load`. A wrong resume attempt with only `SAVE_DIR` initialized the
actor at step 0, so it was stopped before it could overwrite the validated
checkpoint. The AMD launcher now has an explicit optional `LOAD_DIR` variable:

```bash
LOAD_DIR=/path/to/checkpoint SAVE_DIR=/path/to/checkpoint \
  SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=0 NUM_ROLLOUT=2 \
  bash ./amd_qwen3_4b_2gpu_e2e.sh
```

The first explicit `LOAD_DIR` run then reached Megatron checkpoint loading and
exposed a separate PyTorch 2.6 load issue. Megatron's common checkpoint
strategy loaded `common.pt` with `torch.load(load_path, map_location="cpu")`;
on PyTorch 2.6 that defaults to `weights_only=True`, but this `common.pt`
contains trusted Megatron metadata with OmegaConf `DictConfig` objects. The
ROCm checkpoint hook now patches Megatron's `TorchCommonLoadStrategy` on HIP so
the local trusted `common.pt` load uses `weights_only=False`.

### Key Points

- The validated checkpoint path is no longer the temporary legacy `torch`
  workaround and no longer only a cached-rollout smoke.
- `torch_dist` checkpoint save works on a live two-rollout run with optimizer
  state enabled.
- The ROCm writer must keep lazy DCP `WriteItem` planning and resolve tensors
  one at a time during bucket writes; pre-resolving all tensors can kill the
  worker before Python gets a useful traceback.
- Resume is explicit: use `LOAD_DIR` to pass `--load`. `SAVE_DIR` alone is not
  a resume signal.
- On PyTorch 2.6, trusted Megatron `common.pt` loads need
  `weights_only=False`; otherwise resume fails before the actor can restore its
  step.

## 2026-05-30 - Fix ROCm Megatron `torch_dist` checkpoint save

**Type:** Validation
**General description:** The AMD Qwen3-4B two-GPU debug-train smoke run
completed with Megatron `torch_dist` checkpointing and optimizer state enabled.

### Details

The previous `torch_dist` failure died after `common.pt` and before any DCP
writer logs. The fix keeps checkpointing enabled and keeps the sharded
`torch_dist` format, but changes the ROCm checkpoint hook to match the current
Megatron/PyTorch DCP expectations:

- disable the stale `dist_ckpt_save_pre_mcore_014` override on HIP when
  `torch_dist` runs on PyTorch >= 2.6;
- patch both Megatron writer aliases, including the cached
  `strategies.torch.FileSystemWriterAsync` alias;
- avoid PyTorch's legacy `ShardedTensor` construction for old MCore
  `prepend_axis_num` / `flattened_range` tensors by using DCP checkpointable
  sharded tensors;
- force `flatten_sharded_tensors=False` for Megatron DCP planners on HIP,
  matching the newer upstream Megatron save path;
- write checkpoint tensors through a thread-local result queue and blocking
  per-tensor CPU staging instead of Megatron's original async fork/preload path.

Validation run:

- tmux session: `tmux-10`
- Ray job: `raysubmit_fnyW8C56FnBpgbhs`
- W&B group: `qwen3-4b-mi210-2gpu-torch-dist-fix4-20260530_101423`
- Save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-torch-dist-fix4-20260530_101423`
- Args: `SAVE_INTERVAL=1 CKPT_FORMAT=torch_dist NUM_ROLLOUT=1`
- Debug data:
  `Qwen3-4B_mcore_2gpu-torch-dist-fix3-20260530_094730/debug_rollout/0.pt`

The run completed actor step 0/1 and checkpoint finalization:

```text
HIP/ROCm detected: setting flatten_sharded_tensors=False for Megatron torch_dist checkpoint planners
HIP/ROCm detected: using thread-local checkpoint results queue
ROCm streaming checkpoint write rank 0 bucket 1/2
ROCm streaming checkpoint write rank 0 bucket 2/2
ROCm streaming checkpoint write finished on rank 0: buckets=2
successfully saved checkpoint from iteration       0
All training steps finished
Job 'raysubmit_fnyW8C56FnBpgbhs' succeeded
```

The resulting checkpoint contains:

```text
iter_0000000/.metadata 165366 bytes
iter_0000000/__0_0.distcp 12059698420 bytes
iter_0000000/__0_1.distcp 12076849495 bytes
iter_0000000/common.pt 41058 bytes
iter_0000000/metadata.json 119 bytes
latest_checkpointed_iteration.txt 1 bytes
```

`FileSystemReader.read_metadata()` also loaded the DCP metadata successfully
with the ROCm Megatron checkout on `PYTHONPATH`, reporting 175 state-dict
entries and 1123 storage entries, including model embeddings and optimizer
state.

### Key Points

- `CKPT_FORMAT=torch_dist` is now the default checkpoint format for
  `amd_qwen3_4b_2gpu_e2e.sh`.
- Checkpointing and optimizer state remain enabled by default:
  `SAVE_INTERVAL=100`, `CKPT_FORMAT=torch_dist`, `NO_SAVE_OPTIM=0`.
- Legacy `CKPT_FORMAT=torch` remains only a fallback/debug override, not the
  validated default path.

## 2026-05-30 - Validate ROCm Megatron smoke run with checkpointing

**Type:** Validation
**General description:** The AMD Qwen3-4B two-GPU smoke run completed with
Megatron checkpointing enabled by using the legacy Megatron `torch` checkpoint
format before the later `torch_dist` fix was available.

### Details

Checkpointing is required for this launcher. The practical fix is to keep
`--save` and `--save-interval` enabled, but set `--ckpt-format torch` on ROCm.
For the single-rank MI210 actor, `torch` selects Megatron's legacy checkpoint
path (`use_dist_ckpt=False`) and avoids the `torch_dist` sharded checkpoint
writer that had been killing the actor after writing only `common.pt`.

At that point the launcher temporarily defaulted to:

```bash
SAVE_INTERVAL=100
CKPT_FORMAT=torch
NO_SAVE_OPTIM=0
```

Phase-1 foreground validation used `SAVE_INTERVAL=1 CKPT_FORMAT=torch
NUM_ROLLOUT=1`. The generated Ray entrypoint included `--save`,
`--save-interval 1`, and `--ckpt-format torch`. It reached service startup
before the external 300-second validation timeout. That timeout sent SIGTERM to
the validation Ray cluster before the actor reached checkpoint save, so the
stale foreground driver and Ray state were cleaned up before the tmux run.

Phase-2 tmux validation then ran the checkpoint-enabled path:

- tmux session: `tmux-6`
- Ray job: `raysubmit_tqwQYHbbhAfD7RDi`
- Log: `log/amd-qwen3-4b-2gpu-20260530_080242.log`
- W&B group: `qwen3-4b-mi210-2gpu-torch-ckpt-20260530_080241`
- Save directory:
  `/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-4B_mcore_2gpu-torch-ckpt-20260530_080241`

The job completed rollout generation, actor training step 0, checkpoint save,
weight update, and graceful shutdown:

```text
saving checkpoint at iteration       0 to .../Qwen3-4B_mcore_2gpu-torch-ckpt-20260530_080241 in torch format
successfully saved checkpoint from iteration       0 to .../Qwen3-4B_mcore_2gpu-torch-ckpt-20260530_080241 [ t 1/1, p 1/1 ]
Actor training completed step 0/1
All training steps finished
Main func successfully
Job 'raysubmit_tqwQYHbbhAfD7RDi' succeeded
```

The resulting checkpoint contains:

```text
iter_0000000/mp_rank_00/model_optim_rng.pt 24135297502 bytes
latest_checkpointed_iteration.txt 1 bytes
```

### Key Points

- The Relax + SGLang + Megatron ROCm smoke path now works for the
  checkpoint-required MI210 validation.
- This entry records the temporary legacy `torch` workaround.
- The later 2026-05-30 `torch_dist` validation supersedes this workaround as
  the default path.

## 2026-05-29 - Patch ROCm Megatron checkpoint writer aliases

**Type:** Runtime fix
**General description:** The late W&B MI210 run reached the first scheduled
checkpoint and then died because the active Megatron torch-dist save strategy
could still hold the original async filesystem writer alias.

### Details

The run that reached step 99/200 failed immediately after:

```text
saving checkpoint at iteration      99 to .../Qwen3-4B_mcore_2gpu in torch_dist format
Overwriting old incomplete / corrupted checkpoint...
```

The checkpoint directory contained `iter_0000099/common.pt` but not the sharded
checkpoint metadata/files, which points to failure during the torch-dist
sharded save rather than during training, rollout, or optimizer step.

The Relax ROCm patch previously replaced
`megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync`.
However, Megatron's `strategies.torch` module also imports
`FileSystemWriterAsync` into its own module namespace and uses that cached alias
inside `TorchDistSaveShardedStrategy.async_save()`. If that module was imported
first, the runtime save path could bypass the ROCm-safe writer.

The fix adds `patch_rocm_checkpoint_writer()`, which patches both the source
`filesystem_async` module and the cached `strategies.torch` alias before
Megatron model/optimizer setup. The AMD launcher also now supports
environment-overridden `SAVE_DIR`, `SAVE_INTERVAL`, and `NO_SAVE_OPTIM`, so a
fresh short validation can force checkpoint save around rollout 3 instead of
waiting until rollout 99.

### Validation

- Focused tests passed in the `relaxrl_rocm` environment:
  `tests/utils/test_rocm_checkpoint_writer.py`,
  `tests/utils/test_megatron_model.py`, and
  `tests/backends/megatron/test_checkpoint_warmup.py`.
- The required `.venv` path could not run those tests because it does not have
  `pytest` or `pip`; the training environment is the conda `relaxrl_rocm`
  environment used by the launcher.
- Forced `torch_dist` validation did not pass. `SAVE_INTERVAL=4` reached
  `saving checkpoint at iteration 3`, and
  `SAVE_INTERVAL=2 CKPT_FORMAT=torch_dist NO_SAVE_OPTIM=1` reached
  `saving checkpoint at iteration 1`; both runs wrote only `common.pt` before
  `MegatronTrainRayActor` died with Ray `SYSTEM_ERROR` / EOF. Excluding
  optimizer state ruled out optimizer checkpoint state as the primary trigger.
- The AMD launcher temporarily checkpointed by default with `CKPT_FORMAT=torch`
  as a workaround. A later 2026-05-30 validation fixed and restored
  `CKPT_FORMAT=torch_dist` as the default path.

## 2026-05-29 - Preserve W&B custom step metrics in MetricsService

**Type:** Fix
**General description:** The live MI210 run was advancing in the logs, but W&B
could show `train/step` stuck at zero because the MetricsService adapter
stripped namespaced step metrics before reporting to W&B.

### Details

The active tmux run showed repeated actor progress, for example:

```text
step 34: {..., 'train/step': 34}
Reported 66 metrics for step 34
```

The Megatron train path correctly emits `train/step` in the log dictionary.
However, when `use_metrics_service=True`, `MetricsServiceAdapter.log()` removed
the configured `step_key` from the payload before forwarding the metrics batch.
That behavior is useful for a generic helper key such as `"step"`, but it is
wrong for W&B custom step metrics such as `train/step`, `rollout/step`, and
`eval/step`. W&B defines `train/*` against `train/step`, so the named step
metric must be present in the reported payload.

The adapter now preserves step keys ending in `/step` and only strips plain
helper step keys. The focused regression test asserts that generic `"step"` is
still removed while `train/step` remains in the metrics batch.

### Key Points

- The running job was not stuck at train step zero; this was a metrics transport
  issue.
- The fix applies to the next run or a restarted MetricsService, not to an
  already-imported running worker.

## 2026-05-29 - Retrospective on post-sync ROCm SGLang and Megatron bring-up

**Type:** Retrospective
**General description:** After merging the newer upstream `origin/main`, the
AMD Qwen3-4B two-GPU run failed through several ROCm-specific SGLang and
Megatron boundaries before reaching repeated real actor training steps again.

### What We Tried

- Ran the two-phase validation flow for `amd_qwen3_4b_2gpu_e2e.sh` from the
  `relaxrl_rocm` environment.
- Kept the ROCm Megatron checkout ahead of stale Megatron paths in the launcher.
- Added `--sglang-disable-overlap-schedule` after the SGLang overlap scheduler
  failed in a CUDA-named future-token JIT helper on ROCm.
- Patched Relax's SGLang bootstrap on HIP so SGLang uses existing safe fallback
  paths for KV-cache store and clamp-position instead of CUDA-centric JIT
  kernels.
- Patched Megatron HF checkpoint warmup so a single-process distributed actor
  can be the warmup leader even when Ray assigns it `LOCAL_RANK=1`.

### Key Findings

- The post-sync failures were not one bug. The run advanced through a sequence:
  overlap scheduler JIT failure, KV-cache store JIT failure, Megatron warmup
  wait, then clamp-position JIT failure.
- The `LOCAL_RANK=1` warmup hang was a placement/configuration mismatch, not a
  dead Megatron process: SGLang owned GPU 0 and the one-rank actor legitimately
  ran on GPU 1.
- After the fixes, the production tmux run reached real rollout, Megatron train
  metrics, weight sync, and repeated actor completion. At the time of this
  retrospective it had completed step 18 and entered step 19 of 200.
- The visible `waiting for data system to catch up` messages are normal for
  this run shape when they clear after each actor step. They are not the stale
  `train_<n>` wedge unless the actor disappears or the partition stops
  draining.

### What Failed

- SGLang still exposes CUDA-named JIT helpers on ROCm even when higher-level
  launcher flags look ROCm-safe.
- Megatron's checkpoint page-cache warmup assumed `LOCAL_RANK=0` exists for the
  local actor group, which is false in this one-rank actor / one-GPU rollout
  split.
- The full script is not a quick smoke test: it is configured for 200 steps and
  is currently taking roughly 80 seconds per step after startup.

### Open Questions

- Whether this post-sync run can pass the old long-run checkpoint boundary near
  step 99.
- Whether SGLang has additional CUDA-centric JIT helpers that only appear under
  later sampling or scheduling paths.
- Whether the 200-step script should get a shorter explicit smoke-test variant
  so future sync validation can prove "Relax runs on Megatron" without waiting
  multiple hours.

### Key Points

- The practical fix is to keep the ROCm launcher conservative and install
  Relax-side HIP runtime patches before SGLang scheduler code imports the JIT
  helpers.
- Treat single-rank actor placement independently from physical GPU local rank.
  Distributed world size is the correct leader signal for the HF checkpoint
  warmup edge case.
- Current validation status is "working and still running", not "finished".

## 2026-05-29 - Route SGLang clamp-position to torch fallback on ROCm

**Type:** Runtime fix
**General description:** The first post-warmup tmux run reached rollout
generation and actor step 0, then exposed another SGLang HIP-incompatible JIT
helper during decode.

### Details

The production run after the HF checkpoint warmup fix confirmed the actor no
longer waited for a nonexistent local rank 0:

```text
[local_rank=1, dist_world_size=1] Warming HF checkpoint page cache
Actor training step 0/200
```

The next failure moved back to SGLang decode. During rollout generation, the
scheduler crashed in `clamp_position_cuda`:

```text
Runtime check failed at .../jit_kernel/csrc/elementwise/clamp_position.cuh:46:
CUDA error: no ROCm-capable device is detected
```

SGLang's `forward_batch_info.py` already provides `_clamp_position_native`,
which computes the same `torch.clamp(seq_lens - 1, min=0).to(torch.int64)`
fallback. Relax now patches `forward_batch_info.clamp_position` to that native
fallback on `torch.version.hip`, alongside the existing KV-cache store JIT
disablement.

## 2026-05-29 - Let single-process Megatron actors warm HF checkpoints on ROCm

**Type:** Runtime fix
**General description:** The post-sync AMD run got past SGLang bring-up and
then exposed a Megatron actor initialization wait caused by using physical
`LOCAL_RANK` as the only HF checkpoint warmup leader signal.

### Details

After disabling the SGLang overlap scheduler and JIT KV-cache store path, the
production tmux run brought up SGLang, registered rollout, and started the
Megatron actor. The actor stayed alive but stopped progressing after:

```text
[local_rank=1] waiting for local_rank=0 to warm HF checkpoint page cache
```

Ray actor state showed the `MegatronTrainRayActor` was still alive, and SGLang
continued to answer health checks. The process was distributed rank 0/world
size 1, but Ray had placed it on the second GPU while SGLang owned GPU 0, so
the environment exposed `LOCAL_RANK=1`. No Megatron process with
`LOCAL_RANK=0` existed in that single-rank actor path.

The fix updates `relax/backends/megatron/checkpoint.py` so a distributed
world-size-1 process warms the HF checkpoint page cache even when
`LOCAL_RANK != 0`. The existing `/dev/shm` marker and advisory lock remain the
cross-job guard against duplicate checkpoint reads.

## 2026-05-29 - Disable SGLang JIT KV-cache store on ROCm

**Type:** Runtime fix
**General description:** The post-sync AMD launcher now gets past the overlap
scheduler JIT failure, then exposes and avoids the next SGLang HIP-incompatible
KV-cache store kernel path.

### Details

The first validation after syncing with newer `origin/main` reached SGLang
weight loading and Uvicorn startup, but the scheduler crashed in
`resolve_future_token_ids_cuda` with `CUDA error: no ROCm-capable device is
detected`. Adding `--sglang-disable-overlap-schedule` moved the run onto
SGLang's normal scheduler path.

The next foreground run confirmed that the flag took effect, but then the
normal scheduler crashed while storing KV cache:

```text
Runtime check failed at .../jit_kernel/csrc/elementwise/kvcache.cuh:196:
CUDA error: no ROCm-capable device is detected
```

The local SGLang memory pool already has a safe fallback: if
`can_use_store_cache(row_bytes)` returns `False`, it writes KV cache with direct
tensor assignment. Because the external SGLang checkout is outside this
workspace, the fix lives in Relax's SGLang process bootstrap: on
`torch.version.hip`, `relax/backends/sglang/sglang_engine.py` patches
`sglang.srt.mem_cache.memory_pool.can_use_store_cache` to return `False`
inside both the SGLang server process and scheduler subprocess.

### Links

- Runtime file: `relax/backends/sglang/sglang_engine.py`
- Test: `tests/backends/sglang/test_sglang_engine.py`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Failed log before KV-cache fallback:
  `log/amd-qwen3-4b-2gpu-20260529_165722.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-24 - Retrospective on W&B e2e ROCm Megatron run

**Type:** Retrospective
**General description:** The W&B e2e run confirms the ROCm Megatron path now
survives many repeated rollout, training, optimizer, and weight-sync cycles
under tmux, then exposes a later actor-death boundary around checkpoint save.

### What We Tried

- Ran the AMD Qwen3-4B two-GPU e2e launcher from the isolated
  `relaxrl_rocm` environment.
- Used the ROCm Megatron checkout at
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`.
- Enabled W&B online logging with project `relax-amd` and group
  `qwen3-4b-mi210-2gpu-20260424_134200`.
- Followed the two-phase run discipline: foreground validation first, then
  full run in tmux.

### Key Findings

- The foreground validation reached rollout generation, actor training,
  weight sync, and active decoding before the external timeout fired.
- The tmux run completed repeated actor steps through step 98, completed
  Megatron train/loss reduction for rollout 99, emitted `saving checkpoint at
  iteration 99`, and then the `MegatronTrainRayActor` died before the actor
  service could complete step 99.
- After the actor died, Ray still reported the job as `RUNNING`, SGLang served
  health checks, and rollout repeatedly waited on `Current partitions:
  ['train_99']`; this is the same stale-live shape previously seen with
  `train_0`, but at a much later boundary.
- The decisive stability fixes were not W&B-related; they were the ROCm
  Megatron path selection, import isolation, ROCm-safe fusion/provider
  overrides, and the bf16 CPU-offload optimizer path.
- The system is now e2e-functional for many steps, but not full-run stable.
  The current reward/training signal is also weak: recent batches show
  invalid/truncated generations, reward `0.0`, advantage `0.0`, loss `0.0`,
  and grad norm `0.0`.

### What Failed Previously

- Inherited non-ROCm Megatron paths contaminated the ROCm fork.
- Rollout-side Ray/SGLang processes imported Megatron too early.
- CUDA/TE fused paths remained reachable despite ROCm launcher flags.
- ROCm Megatron JIT helpers entered TorchInductor/Triton or TorchScript paths
  that failed on this stack.
- The first stable optimizer path required preventing Megatron wrappers from
  rebuilding HDO around fp32 CPU-offloaded main params.

### Open Questions

- Whether the step-99 death is tied to checkpoint saving, resource/GCS
  pressure, W&B/metric flushing, or a late memory/process failure.
- Why current generations are nonsensical and training metrics remain zero.
- Whether checkpoint save at iteration 99 should be disabled, delayed, or
  converted to a more ROCm/Ray-safe path before retrying 200 steps.

### Links

- W&B e2e log: `log/amd-qwen3-4b-2gpu-20260424_134200.log`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- ROCm Megatron checkout:
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`
- Related skills: `.codex/skills/rocm-relax-bringup/SKILL.md`,
  `.codex/skills/megatron-hybrid-device-optimizer-rocm/SKILL.md`,
  `.codex/skills/ray-rollout-import-isolation/SKILL.md`,
  `.codex/skills/megatron-bridge-rocm-overrides/SKILL.md`

## 2026-04-24 - Retrospective on ROCm Megatron MI210 optimizer bring-up

**Type:** Retrospective
**General description:** The ROCm Megatron branch moved from repeated actor
deaths at the first optimizer step to repeated successful MI210 training steps
under the foreground validation gate.

### What We Tried

- Used the dedicated `relaxrl_rocm` environment rather than the earlier Relax
  environment.
- Pointed the launcher at
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`.
- Kept the run on the 2-GPU MI210 split: SGLang rollout on one GPU, Megatron
  actor on one GPU.
- Repeated foreground validation with:

```bash
HIP_VISIBLE_DEVICES=0,1 timeout 600s bash ./amd_qwen3_4b_2gpu_e2e.sh
```

- Patched ROCm Megatron JIT helpers away from `torch.compile` / TorchScript on
  HIP.
- Patched the single-rank ROCm CPU-offload path so `HybridDeviceOptimizer`
  keeps CPU copies in bf16, uses unpinned synchronous transfers, and steps one
  CPU optimizer per parameter.
- Patched Megatron optimizer helpers to tolerate bf16 live params with fp32
  `main_grad` and bf16 CUDA grads during clipping.
- Patched Relax-side optimizer setup so the single-rank ROCm actor does not
  reintroduce Megatron fp32 main-param wrapping around the CPU-offload fallback.
- Skipped single-rank TP all-gather and transfer-queue object broadcasts that
  previously hit ROCm SIGBUS paths.

### Key Findings

- The active failure was not Ray, SGLang, reward execution, or generic rollout
  startup. Once those were healthy, the hard boundary was Megatron optimizer
  internals on ROCm.
- ROCm Megatron's default `jit_fuser` behavior is unsafe on this stack because
  `torch.compile` can hit the `KernelMetadata.cluster_dims` TorchInductor /
  Triton failure, while TorchScript also fails on helper methods.
- Relax launcher flags can look correct while downstream Megatron wrappers have
  already rebuilt the optimizer around fp32 main params.
- Keeping HDO CPU-offloaded params in bf16 is the decisive memory/stability
  fix for this MI210 path.
- The latest validation completed actor training steps 0 through 5 and timed
  out only at the external 10-minute gate while rollout 6 was running.

### What Failed

- A no-offload branch got past the previous CPU-offload boundary but exposed
  the ROCm TorchInductor/Triton `KernelMetadata.cluster_dims` failure.
- Keeping CPU offload but allowing fp32 CPU param updates made large CPU AdamW
  steps too slow/fragile and could kill the actor without a Python traceback.
- Bypassing some fp32 wrapping then exposed dtype contract failures:
  fp32 `main_grad` assignment into bf16 params and fp32-only clip-grad
  assertions.
- The HDO per-parameter warning logs were useful for diagnosis but too verbose
  for steady-state validation, so they were later demoted to debug.

### Open Questions

- Whether the full 200-step run completes under tmux without another later
  boundary.
- Whether a less conservative CPU optimizer grouping can be reintroduced after
  a stable long run.
- Whether the ROCm Megatron patches should be upstreamed, carried as a local
  patchset, or isolated behind runtime guards in the launcher.

### Proposed Skill Updates

- Update `.codex/skills/megatron-hybrid-device-optimizer-rocm/SKILL.md` with
  the successful bf16 CPU-offload wrapper recipe and validation evidence.
- Update `.codex/skills/rocm-relax-bringup/SKILL.md` so its latest validated
  state is "multiple training steps completed" rather than "current boundary is
  optimizer step".

### Links

- Validation log: `log/amd-qwen3-4b-2gpu-20260424_094405.log`
- Troubleshooting: `references/troubleshooting.md`
- ROCm Megatron checkout:
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`

## 2026-04-24 - ROCm Megatron CPU-offload path passes repeated MI210 optimizer steps

**Type:** Observation
**General description:** The ROCm Megatron branch now reaches real rollout and
actor training on the 2-GPU MI210 validation, including repeated successful
CPU-offloaded optimizer steps.

### Details

The latest foreground validation used the isolated `relaxrl_rocm` environment,
the ROCm Megatron checkout, and the local two-GPU AMD launcher:

```bash
HIP_VISIBLE_DEVICES=0,1 timeout 600s bash ./amd_qwen3_4b_2gpu_e2e.sh
```

This run moved beyond the earlier step-0 crash chain:

- SGLang rollout startup completed on one MI210.
- The actor initialized from
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`.
- The actor-to-rollout weight update completed.
- Rollout generation produced `train_0`.
- The actor completed forward/backward and `optimizer.step()` for multiple
  rollouts.

The relevant log is `log/amd-qwen3-4b-2gpu-20260424_094405.log`. During the
run, the actor emitted:

```text
train_one_step rollout=0 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
train_one_step rollout=1 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
train_one_step rollout=2 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
train_one_step rollout=3 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
train_one_step rollout=4 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
train_one_step rollout=5 step=0: finished optimizer.step (update_successful=True, grad_norm=0.0, num_zeros_in_grad=None)
```

The foreground command exited with code `124` only because the external
10-minute validation timeout fired while rollout 6 was decoding. A follow-up
cleanup found no active Ray processes and ROCm SMI showed all GPU memory freed.

The key fixes were split across the local Relax fork and the local ROCm
Megatron checkout:

- keep ROCm Megatron JIT helpers eager on HIP instead of routing them through
  TorchInductor/Triton or TorchScript;
- keep the single-rank ROCm CPU-offload path on unpinned, non-overlap,
  one-parameter CPU optimizers;
- prevent Megatron mixed-precision optimizer config from reintroducing fp32
  CPU-offloaded main params on this path;
- cast `FP32Optimizer.prepare_grads()` assignments back to the live parameter
  dtype when Megatron has bf16 params with fp32 `main_grad`;
- allow bf16 CUDA grads in `clip_grad_by_total_norm_fp32()`;
- skip redundant single-rank TP all-gather and transfer-queue object broadcast
  calls that previously produced ROCm SIGBUS failures.

### Key Points

- The current boundary is no longer "actor dies at first real optimizer step".
- The safe path still depends on the local ROCm Megatron patches; it is not an
  upstream-clean validation.
- The HDO per-parameter trace logs were useful while diagnosing the step
  boundary but have been demoted to debug level for future runs.
- A full production run should still use the two-phase workflow: foreground
  validation first, then tmux for the long run.

### Links

- Validation log: `log/amd-qwen3-4b-2gpu-20260424_094405.log`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Relax-side optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Relax train boundary logging: `relax/backends/megatron/model.py`
- ROCm Megatron patches:
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/jit.py`,
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`,
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/optimizer.py`,
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/optimizer/clip_grads.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-24 — Keep ROCm Megatron JIT helpers off TorchInductor

**Type:** Observation
**General description:** The ROCm Megatron fork reached step-0 log-prob computation, then failed because Megatron's JIT helper decorator selected `torch.compile` on HIP.

### Details

The newest long run in the ROCm Megatron sibling workspace was no longer an
optimizer-offload failure. The launcher had already removed CPU optimizer
offload, and the run reached real actor training with:

- rollout and SGLang startup complete
- `optimizer_cpu_offload=False`
- `train_0` produced for the actor
- actor-side log-prob computation starting on step 0

The first concrete failure was:

```text
torch._inductor.exc.InductorError: AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'
```

The stack showed the failing compiled helper was
`ROCm-Megatron-LM/megatron/core/fusions/fused_cross_entropy.py`:
`calculate_logits_max(...)`. A tiny one-GPU reproduction confirmed the same
failure by importing that helper from the ROCm Megatron checkout and calling it
on a bf16 HIP tensor.

The root cause was `ROCm-Megatron-LM/megatron/core/jit.py`: on PyTorch >= 2.2
it promoted `jit_fuser` from `torch.jit.script` to `torch.compile`, including
on HIP. That sent small Megatron fused helpers through the ROCm
TorchInductor/Triton launcher path that expects `KernelMetadata.cluster_dims`.

This pass patches the local ROCm Megatron checkout so HIP uses an eager no-op
`jit_fuser` instead of either `torch.compile` or TorchScript. The first version
kept `torch.jit.script` on HIP, which fixed the tiny `calculate_logits_max`
reproduction but failed during a real import when TorchScript compiled
`megatron.core.transformer.torch_norm.L2Norm` and could not resolve
`self.eps`.

The first foreground validation after that patch exposed a separate launcher
contamination issue before it reached training: the Ray runtime env still
contained `/vast/users/qirong.ho/erland/Python_project/Megatron-LM` and the log
showed imports from the non-ROCm checkout. The ROCm sibling launcher had set
`MEGATRON_DIR`, but it did not prepend that directory to `PYTHONPATH`; instead
it appended the inherited shell `PYTHONPATH`. The launcher now constructs
`PYTHONPATH` explicitly as SGLang, `${MEGATRON_DIR}`, then the Relax fork root.

### Key Points

- The current boundary is compiler/runtime compatibility in the ROCm Megatron
  fork, not Ray, SGLang, or the old CPU-offload optimizer path.
- The eager `jit_fuser` patch is local to
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/jit.py`.
- The ROCm launcher must not inherit a stale non-ROCm `Megatron-LM` path.
- The next meaningful validation is a clean foreground run long enough to get
  past the old step-0 `calculate_logits_max` failure.

### Links

- Local Megatron patch: `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM/megatron/core/jit.py`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Latest failing log: `log/amd-qwen3-4b-2gpu-20260421_174147.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — Single-rank Megatron optimizer normalization on MI210

**Type:** Observation
**General description:** The saved replay batch and new step-boundary logs showed the remaining MI210 actor crash is inside `optimizer.step()`, so this pass removes the distributed-optimizer wrapper from the single-rank actor path.

### Details

The saved debug capture from `log/debug_capture_20260418_153613` moved the
current failure from a broad “step 0 actor death” into a precise boundary:

- rollout generation completed
- `train_0` transferred successfully
- actor-side rollout statistics were logged
- `train_one_step ... finished forward_backward` was emitted
- `train_one_step ... starting optimizer.step` was emitted
- the worker then died before `finished optimizer.step`

That ruled out the forward/backward path and pointed directly at the optimizer
stack. Inspecting the Relax-side Megatron setup showed that we still force
distributed-optimizer features globally, while this actor path is only using a
single data-parallel rank.

This pass adds a small normalization step in
`relax/backends/megatron/model.py` before `get_model(...)` and optimizer
construction. When `data_parallel_size == 1` (or `world_size == 1` as a
fallback), Relax now disables:

- `use_distributed_optimizer`
- `use_precision_aware_optimizer`
- `overlap_param_gather`
- `overlap_param_gather_with_optimizer_step`

The goal is not to change the multi-rank path. It is to keep the MI210
single-rank actor on the simpler non-distributed optimizer path while still
using CPU optimizer offload for memory pressure.

Focused test coverage was added for the normalization helper so the single-rank
and multi-rank cases stay explicit.

### Key Points

- The current hypothesis is no longer “some random ROCm actor death”; it is
  specifically a distributed-optimizer / optimizer-step boundary on the
  single-rank actor path.
- This pass intentionally changes only the single-rank optimizer config and
  leaves the rest of the rollout / actor / SGLang pipeline untouched.
- The next meaningful validation is a replay or foreground run that proves the
  actor can get past `optimizer.step()` at step 0.

### Links

- Train setup: `relax/backends/megatron/model.py`
- Replay capture: `log/debug_capture_20260418_153613`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — HDO remap and single-rank offload-overlap disable stabilize replay

**Type:** Observation
**General description:** After removing distributed-optimizer features from the single-rank MI210 actor path, the next failures became explicit CPU-offload bookkeeping issues, and fixing those moved both replay and full-stack validation past the old step-0 actor death.

### Details

The first replay pass after single-rank optimizer normalization did not hit the
old opaque EOF crash first. Instead, it failed with a concrete Python error
inside the local Megatron CPU-offload optimizer:

- `HybridDeviceOptimizer` raised a `KeyError` during param-group synchronization
- the failure came after Megatron had already wrapped main params
- that pointed to stale CPU-offload bookkeeping rather than another random Ray
  or ROCm worker death

This pass fixed that in two layers:

1. `relax/backends/megatron/model.py` now refreshes wrapped
   `HybridDeviceOptimizer` instances immediately after
   `get_megatron_optimizer(...)`, re-running the sub-optimizer initialization
   and param-group sync against the final wrapped/main-param view.
2. `relax/backends/megatron/optimizer_utils.py` now also disables
   `overlap_cpu_optimizer_d2h_h2d` when the actor is effectively single-rank
   (`data_parallel_size == 1` or `world_size == 1`), since that overlap path
   adds ROCm-native complexity without a distributed upside on the current
   MI210 actor layout.

I also added deeper optimizer-step logging in the local Megatron checkout to
separate “native crash before Python returns” from “foreground timeout killed a
healthy worker”.

The important validation outcomes changed materially:

- train-only replay with the saved step-0 batch now survives the full
  5-minute foreground validation window and exits only because `timeout`
  terminates the Ray cluster
- the python-core worker log for the replay boundary now shows shutdown via
  `SIGTERM` from the timeout, not the earlier spontaneous actor death
- the full-stack 5-minute validation also survives the window from a clean
  cluster and reaches SGLang distributed init plus `Load weight begin` / shard
  loading on the transformers backend

That means the old “first real train step on MI210 always dies immediately”
claim is no longer true under the current replay and foreground-validation
conditions. The next correct step is a long tmux production run from the new
state.

### Key Points

- The saved replay path remains the highest-signal way to debug train-step
  crashes, because it turns opaque worker deaths into concrete optimizer
  boundaries.
- HDO param-group refresh after Megatron main-param wrapping is required on
  this CPU-offload path.
- Single-rank MI210 actor runs should keep CPU offload but avoid
  `overlap_cpu_optimizer_d2h_h2d` until stability is proven with replay.

### Links

- Runtime files: `relax/backends/megatron/model.py`, `relax/backends/megatron/optimizer_utils.py`
- Local Megatron: `Megatron-LM/megatron/core/optimizer/optimizer.py`, `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Replay capture: `log/debug_capture_20260418_153613`
- Full-stack validation log: `log/amd-qwen3-4b-2gpu-20260418_163048.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — Serial startup order changed to rollout-first on MI210

**Type:** Observation
**General description:** The next surviving boundary was not another train-step kernel failure; it was actor lifecycle during rollout startup, so this pass changed non-colocated serial service creation to start rollout before actor.

### Details

The previous `tmux-5` failure looked, at first glance, like another
`set_rollout_manager()` crash. The worker logs showed a different story:

- SGLang actually finished weight loading and reached ready state
- the Megatron actor worker had already died earlier at `2026-04-18 17:04:06 UTC`
- the later `set_rollout_manager()` RPC only exposed that already-dead actor
- the actor had been created first and then sat resident and idle while rollout
  spent minutes inside SGLang startup

That made the failure a startup-order problem, not a direct
`set_rollout_manager()` implementation bug.

This pass added `order_service_creation(...)` in
`relax/core/controller.py` and used it during `register_all_serve()`:

- for non-colocated serial startup (`fully_async=False`, `colocate=False`)
  `rollout` is now created before `actor`
- colocated and fully-async paths keep their existing order
- focused controller tests now lock in both the rollout-first serial order and
  the unchanged colocated/async behavior

Focused validation:

- `python -m pytest -q tests/core/test_controller_data_source_config.py tests/core/test_service_runtime_env.py tests/utils/test_megatron_model.py tests/distributed/ray/test_train_actor.py tests/entrypoints/test_train.py`
- result: `24 passed`

The required clean foreground validation then materially improved:

- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
- result: exit code `124`
- rollout came up first and SGLang finished startup
- actor was created afterward, survived initialization, and
  `set_rollout_manager()` succeeded
- initial weight sync completed
- the run entered real work:
  - `Actor training step 0/200`
  - `Start rollout 0/200`
  - rollout generation and decode activity were active before timeout

This is the first foreground validation that definitively cleared the old
idle-actor-before-rollout death boundary.

### Key Points

- The old `set_rollout_manager()` error had become a delayed symptom, not the
  real root cause.
- On MI210, startup order matters: rollout-first is materially more stable than
  actor-first on the non-colocated serial path.
- After a foreground timeout, Ray still needs explicit cleanup before the tmux
  production rerun.

### Links

- Controller: `relax/core/controller.py`
- Tests: `tests/core/test_controller_data_source_config.py`
- Validation log: `log/amd-qwen3-4b-2gpu-20260418_181910.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — `dapo` reward moved off the Ray reward-worker pool

**Type:** Observation
**General description:** The first-train-step MI210 stall was narrowed to the rollout reward path, so this pass stopped routing `dapo` through `RewardWorker` actors and revalidated from a clean cluster.

### Details

The overnight run from `2026-04-17` had already moved past startup and into
real work:

- actor step 0 started
- rollout generation started
- SGLang served real `/generate` traffic
- the Megatron actor logged
  `start to get rollout_id: 0 data from transfer queue for train with mcore.`

Then the run stalled. The useful worker-side signals were:

- `RolloutManager` logged only
  `RewardExecutor: created 16 RewardWorker actors (max_concurrency=64)`
- there was then a long silent gap before the Megatron actor died with
  `ActorDiedError` / EOF
- no `Prepared rollout batch ...` or `Batch ... transferred successfully`
  logs appeared before the crash

That pointed to the reward-to-transfer boundary rather than another startup
bug. The active launcher uses `--rm-type dapo`, and `dapo` is a small local
regex/string reward defined in `relax.engine.rewards.math_dapo_utils`. It did
not need the generic Ray `RewardWorker` path that was originally built for
heavier or thread-unsafe reward functions.

This pass changed the reward dispatching rule:

- `dapo` now runs locally via `asyncio.to_thread` inside
  `relax/engine/rewards/__init__.py`
- the Ray `RewardWorker` pool is still used for the heavier reward types
- regression tests now assert that `dapo` does not initialize reward-worker
  actors in either single-sample or batched reward execution

Focused validation:

## 2026-04-21 — Disable CPU optimizer offload to separate fit issues from HDO crashes

**Type:** Observation
**General description:** The MI210 run now dies at a narrow `HybridDeviceOptimizer` boundary, so this pass removes CPU optimizer offload from the AMD launcher to see whether offload itself is the bottleneck.

### Details

The latest long run no longer failed in startup, rollout, or actor
initialization. It got to real step-0 training and then consistently died at:

- `HybridDeviceOptimizer step: cpu sub-optimizer 51 grad sync done`
- `HybridDeviceOptimizer step: cpu sub-optimizer 51 step begin`

with the actor disappearing and rollout wedging forever on `train_0`.

That is narrow enough that the right next branch is not another Ray or rollout
change. It is to remove the CPU-offload path entirely from the AMD launcher and
re-run the required clean foreground gate.

This pass therefore drops these launcher flags:

- `--optimizer-cpu-offload`
- `--optimizer-offload-fraction 1.0`
- `--overlap-cpu-optimizer-d2h-h2d`
- `--use-precision-aware-optimizer`

The purpose is diagnostic first:

- if the run now fails with HIP OOM, then offload was still required for model
  fit and the real work stays in the offload path
- if it gets past the old `cpu sub-optimizer 51 step begin` boundary, then the
  offload implementation was the real blocker

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-21 — Create a sibling Relax workspace for the ROCm Megatron fork

**Type:** Observation
**General description:** The baseline AMD workspace is preserved unchanged while a new sibling copy is configured to use `ROCm/Megatron-LM` through a launcher-local path override.

### Details

The baseline `Relax/` workspace is now a valuable reference because it contains
all current AMD bring-up patches and the latest no-offload launcher branch.
Trying the ROCm Megatron fork in place would have mixed backend-fork effects
with ongoing local workspace drift.

This pass therefore creates a sibling experiment workspace:

- baseline workspace: `/vast/users/qirong.ho/erland/Python_project/Relax`
- fork workspace: `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron`
- ROCm Megatron checkout:
  `/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM`

The sibling launcher now uses:

- `MEGATRON_DIR="${MEGATRON_DIR:-/vast/users/qirong.ho/erland/Python_project/ROCm-Megatron-LM}"`

and routes both `PYTHONPATH` and `MEGATRON` through that variable, so the fork
experiment can be run without touching the original baseline folder or the
shared NVIDIA checkout.

### Links

- Fork workspace: `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron`
- Fork launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Troubleshooting: `references/troubleshooting.md`

- `python -m pytest -q tests/engine/rewards/test_reward_worker.py`
- result: `37 passed`

Foreground validation:

- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
- result: exit code `124`
- meaning: the patched run stayed alive through the full 5-minute validation
  window and did not reproduce an immediate failure

Because a timed foreground timeout can leave stale Ray processes behind on this
stack, the next step is still to stop Ray explicitly and restart the full run
in tmux from a clean cluster.

### Key Points

- The current fix is about reducing control-plane load and removing an
  unnecessary actor boundary on the exact MI210 path under test.
- This pass does not yet prove full end-to-end stability; it proves the patched
  run remains healthy through the required 5-minute foreground window.
- The next meaningful checkpoint is a fresh tmux run that reaches and passes
  the old step-0 reward/transfer boundary.

### Links

- Runtime file: `relax/engine/rewards/__init__.py`
- Tests: `tests/engine/rewards/test_reward_worker.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Rollout replica init args still serialized Megatron enums

**Type:** Observation
**General description:** After moving the Megatron blocker to rollout-only module import, the next MI210 failure became a Ray Serve deserialization issue in the rollout replica init payload.

### Details

This pass closed two earlier boundaries first:

- controller-side data-source config sanitization stopped the old
  `RolloutDataSourceWithBuffer` contamination during actor creation
- service-scoped runtime env propagation moved the rollout-side SGLang
  isolation to process startup without poisoning the actor replica

The next clean 5-minute foreground validation then produced a much cleaner
rollout-side failure:

- `ServeReplica:rollout:Rollout` logged
  `Removed Megatron-LM from sglang_engine module import PYTHONPATH and sys.path`
- it also logged
  `Installed Megatron import blocker in sglang_engine module import`
- the replica then died before `Rollout.__init__` could really start
- the traceback pointed to
  `cloudpickle.loads(serialized_init_args)` inside Ray Serve replica creation
- the concrete blocked import was
  `megatron.core.transformer.enums`

That means the blocker is now firing at the right lifecycle stage, and the
remaining problem is that the rollout deployment payload still carries a
Megatron-owned object. The most obvious example is the parsed
`attention_backend`, which shows up in the args dump as `AttnBackend.auto`.
Because Ray Serve serializes the full rollout init args, deserializing that enum
requires importing `megatron.core.transformer.enums`, which the rollout-side
transformers isolation correctly blocks.

The code fix in this pass is to sanitize rollout service config objects before
binding the Serve deployment:

- `relax/core/service.py` now has `build_service_config()`
- for the rollout role, enum-valued config entries are converted to their plain
  `.value`
- non-rollout services still receive the original config object

Focused validation stayed green after the patch with:

- `tests/core/test_service_runtime_env.py`
- `tests/utils/test_runtime_env.py`
- `tests/core/test_controller_data_source_config.py`
- `tests/backends/sglang/test_sglang_engine.py`
- `tests/distributed/ray/test_rollout.py`
- `tests/engine/rollout/test_data_source.py`

for `37 passed`.

The next step is another clean foreground AMD validation from a stopped Ray
cluster to confirm the rollout replica no longer dies in
`cloudpickle.loads(serialized_init_args)`.

### Key Points

- The rollout-side module-import blocker is now useful because it exposes the
  real serialized payload boundary instead of letting Megatron leak silently.
- This is the rollout-service analogue of the earlier data-source actor
  serialization bug.
- The remaining work is to prove the enum sanitization closes the new boundary
  in a fresh run.

### Links

- Runtime files: `relax/core/service.py`, `relax/backends/sglang/sglang_engine.py`
- Tests: `tests/core/test_service_runtime_env.py`, `tests/backends/sglang/test_sglang_engine.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Lazy DCS package exports stop Megatron import leakage

**Type:** Observation
**General description:** The next AMD startup boundary was caused by checkpoint-service package init side effects, so this pass converted those exports to lazy imports and added a direct regression test.

### Details

After the `SGLangEngine.__init__` blocker regression was fixed, the remaining
startup contamination was still visible in control-plane services such as
`DCSCoordinator`, `HealthStatus`, and `RolloutManager`. A plain interpreter
reproduction showed the issue without Ray:

- `import relax.distributed.checkpoint_service.coordinator.service`
- Python first executed `relax.distributed.checkpoint_service.__init__`
- that `__init__` eagerly imported the client and backend exports
- `relax.distributed.checkpoint_service.backends.device_direct` imported
  `from megatron.core import mpu`

So the control-plane import path was contaminated before any rollout engine or
training actor work began.

This pass changed the two package init files:

- `relax/distributed/checkpoint_service/__init__.py`
- `relax/distributed/checkpoint_service/backends/__init__.py`

Both now expose the same public symbols through lazy `__getattr__` lookups
instead of eager import-time re-exports. The behavior change is intentionally
narrow: importing the coordinator service should stay lightweight, while code
that actually requests `CheckpointEngineClient` or `DeviceDirectBackend` still
resolves those symbols on demand.

The regression target added in this pass is:

- `tests/distributed/checkpoint_service/test_imports.py::test_importing_coordinator_service_does_not_import_megatron_or_sglang`

The next validation step is the standard clean 5-minute AMD foreground run from
a fresh Ray cluster.

### Key Points

- The contamination boundary is now reproducible without Ray, which makes it
  much cheaper to debug.
- Package `__init__` re-exports were enough to import `megatron.core` on
  control-plane startup paths.
- This is a true control-plane cleanup, not another SGLang-specific workaround.

### Links

- Runtime files: `relax/distributed/checkpoint_service/__init__.py`, `relax/distributed/checkpoint_service/backends/__init__.py`
- Tests: `tests/distributed/checkpoint_service/test_imports.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Controller/registry startup contamination narrowed to advantages imports

**Type:** Observation
**General description:** After the DCS package fix, the next control-plane Megatron leak came from `relax.components.advantages`, which made `relax.core.registry` and `relax.core.controller` heavy at import time.

### Details

The clean AMD validation after the DCS lazy-export patch still showed Megatron
warnings from `HealthStatus`, `Rollout`, and `RolloutManager`. A plain
interpreter reproduction isolated the new boundary:

- `import relax.utils.health_system` stayed clean
- `import relax.components.rollout` stayed clean
- `import relax.core.registry` imported Megatron immediately
- `import relax.core.controller` did the same because it imports
  `relax.core.registry`

The concrete source was `relax.components.advantages`. That module is imported
eagerly by `relax.core.registry`, but it had two Megatron-specific imports at
file scope:

- `from megatron.core import mpu`
- `from relax.backends.megatron.loss import apply_opd_kl_to_advantages`

This pass moved those imports into the exact branches that need them inside
`Advantages.compute_advantages_and_returns()`:

- the PPO path imports `mpu` on demand
- the OPD path imports `apply_opd_kl_to_advantages` on demand

The new regression targets are:

- `tests/components/test_advantages_imports.py::test_importing_advantages_module_does_not_import_megatron_or_sglang`
- `tests/core/test_registry_imports.py::test_importing_core_registry_does_not_import_megatron_or_sglang`

The next validation step is to rerun the targeted import suite and then restart
the clean 5-minute AMD foreground run from a fresh Ray cluster.

### Key Points

- The DCS package leak was real, but not the only control-plane contamination.
- `relax.core.registry` must remain lightweight because many workers touch it
  indirectly.
- `advantages.py` was the next highest-level Megatron import surface.

### Links

- Runtime files: `relax/components/advantages.py`, `relax/core/registry.py`
- Tests: `tests/components/test_advantages_imports.py`, `tests/core/test_registry_imports.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Rollout-side isolation moved ahead of rollout-function loading

**Type:** Observation
**General description:** The next AMD startup leak was inside the rollout-side control plane itself, so this pass moved the transformers-mode Megatron isolation earlier into the `Rollout` and `RolloutManager` processes.

### Details

After the controller/registry cleanup, the clean 5-minute AMD validation still
showed Megatron warnings from:

- `ServeReplica:rollout:Rollout`
- `RolloutManager`

At the same time, the import reproductions for these modules were already clean:

- `import relax.core.registry`
- `import relax.utils.health_system`
- `import relax.engine.rollout.data_source`

That pointed to runtime work inside the rollout processes rather than simple
module-scope imports. The key rollout-side edge is that
`RolloutManager.__init__` loads the configured rollout function via
`load_function(self.args.rollout_function_path)`, and on this setup that path is
`relax.engine.rollout.sglang_rollout.generate_rollout`. That happens before any
`SGLangEngine` actor exists, so the previous engine-only isolation was too late.

This pass made the rollout-side isolation explicit:

- `relax/components/rollout.py` now installs transformers-mode Megatron
  isolation at the start of `Rollout.__init__`
- `relax/distributed/ray/rollout.py` now installs the same isolation at the
  start of `RolloutManager.__init__`

The rollout regression suite was extended to cover those helper paths.

The next validation step is another clean 5-minute AMD foreground run from a
stopped Ray cluster, because the in-flight run was started before this patch.

### Key Points

- The engine actor was not the earliest rollout-side import surface.
- `load_function(...sglang_rollout.generate_rollout)` is an important startup
  boundary for transformers-mode isolation.
- Rollout-side isolation now covers the Serve replica, the manager, and the
  engine actor rather than only the final engine actor.

### Links

- Runtime files: `relax/components/rollout.py`, `relax/distributed/ray/rollout.py`
- Tests: `tests/distributed/ray/test_rollout.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — SGLang Actor-Creation Isolation Regression

**Type:** Observation
**General description:** The latest MI210 pass found that module-import-time Megatron blocking was too early for Ray actor creation, then narrowed the fix to import-time path pruning plus later actor/process blocking.

### Details

This pass started from the remaining rollout-side contamination problem and
tried to move the SGLang worker isolation earlier by propagating a dedicated
runtime env flag into the rollout engine actors. The first version installed a
module-import-time `MetaPathFinder` blocker for `megatron` inside
`relax/backends/sglang/sglang_engine.py`.

The regression was clear in the next clean foreground validation:

- `RolloutManager` reached rollout-engine creation
- `SGLangEngine` died during actor creation, before it could answer
  `_get_current_node_ip_and_free_port()`
- Ray reported `ActorDiedError` from `ray::SGLangEngine.__init__()`
- the nested root cause was `Blocked import of megatron for SGLang transformers backend`

That means the module-import-time blocker was running before Ray had finished
computing actor creation task inputs. The fix in this pass narrowed the early
isolation to path/finder pruning only:

- keep pruning Megatron from `PYTHONPATH`, `sys.path`, editable path hooks, and
  cached modules at module import time
- delay the stronger `megatron` import blocker until `SGLangEngine.__init__`
  and the spawned SGLang server/scheduler subprocess path

The focused regression suite stayed green with:

- `tests/backends/sglang/test_sglang_engine.py`
- `tests/distributed/ray/test_rollout.py`

for `13 passed`.

The follow-up clean 5-minute foreground validation is in progress from this
state. The important fact already established is that the earlier blocked-import
regression is understood and the fix direction is no longer guesswork.

### Key Points

- Import-time path pruning is safe in the Ray actor worker; import-time hard
  blocking is not.
- The previous SGLang isolation regression was introduced by us and was made
  explicit with a clean Ray stack and worker-side logs.
- The remaining broader contamination in `DCSCoordinator`, `HealthStatus`, and
  `RolloutManager` still exists, but it is now separate from the specific
  `SGLangEngine.__init__` blocked-import failure.

### Links

- Runtime files: `relax/backends/sglang/sglang_engine.py`, `relax/distributed/ray/rollout.py`
- Tests: `tests/backends/sglang/test_sglang_engine.py`, `tests/distributed/ray/test_rollout.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Clean-Cluster Revalidation After Import-Surface Fixes

**Type:** Observation
**General description:** This pass reduced the rollout startup import surface, proved the 5-minute validation still passes on a clean cluster, and isolated a separate operational problem where timed foreground validation leaves a stale Ray job behind.

### Details

This pass changed two code paths that were importing SGLang/Megatron too early:

- `relax/distributed/ray/rollout.py` no longer imports `sglang.srt.constants` at module scope; it now mirrors those simple constant values locally.
- `relax/backends/sglang/sglang_engine.py` no longer imports the checkpoint-service client at module scope; DCS client creation is now lazy.

The targeted regression coverage passed:

- `tests/distributed/ray/test_rollout.py`
- `tests/backends/sglang/test_sglang_engine.py`

with `12 passed`.

Two clean import checks also passed:

- importing `relax.distributed.ray.rollout` no longer imports `sglang`
- importing `relax.backends.sglang.sglang_engine` no longer imports `sglang` or `megatron`

Operationally, the first 5-minute validation after these fixes exposed another issue: the shell-side timeout ended the foreground command, but the Ray job tree continued running. That stale job later polluted a tmux run with mixed worker populations. After explicitly stopping Ray and killing leftover Relax processes, a fresh clean-cluster foreground validation again survived the full 5-minute window with exit code `124`.

What remains true after the clean rerun:

- rollout-side startup still logs Megatron warnings before the later `Installed Megatron import blocker in SGLangEngine process` marker
- the clean 5-minute validation still reaches actor init, rollout deployment, RolloutManager creation, W&B run creation, and SGLang launch
- the clean production rerun is now live in `tmux-4`

### Key Points

- The rollout startup import surface is materially smaller than before; importing the rollout stack itself is no longer enough to pull in SGLang/Megatron.
- A timed-out phase-1 validation must be followed by explicit Ray/process cleanup on this machine.
- The remaining blocker is still the later actor lifecycle boundary, not the earlier rollout module import surface.

### Links

- Runtime files: `relax/distributed/ray/rollout.py`, `relax/backends/sglang/sglang_engine.py`
- Tests: `tests/distributed/ray/test_rollout.py`, `tests/backends/sglang/test_sglang_engine.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-16 — AMD MI210 Overnight Validation

**Type:** Retrospective
**General description:** The overnight MI210 run validated that the Ray control-plane overload mitigation was real, but it also showed that the remaining actor death during rollout-manager hookup is still nondeterministic and not fully fixed.

### Details

This pass used the same AMD Qwen3-4B 2-GPU GRPO launcher on MI210 (`gfx90a`) with:

- single-GPU actor and single-GPU rollout layout
- `lr=1e-6`
- `global_batch_size=16`
- `rollout_batch_size=2`
- `n_samples_per_prompt=8`
- `rollout_max_response_len=1024`
- CPU optimizer offload enabled
- `--qkv-format bshd`
- SGLang `transformers` backend on Triton attention
- Ray head capped to 16 CPUs with internal task events disabled on Relax-managed actors

There were still no standalone `training_reports/` artifacts, so this retrospective is based on the launcher configuration, the foreground validation run, and the overnight `tmux-6` log.

What changed relative to the previous pass:

- The foreground validation remained alive for the full `timeout 540s` window.
- The previous Ray/GCS failure did not recur overnight, which strongly suggests the lower head CPU count plus reduced task-event traffic actually fixed that control-plane boundary.
- W&B authentication and run creation were healthy. The overnight run created and synced run `3xuisxfn`, so the remaining logging gap is downstream of startup rather than a basic W&B setup failure.

What still failed:

- The overnight run still died during `Actor.set_rollout_manager`, inside `RayTrainGroup.set_rollout_manager`, when the underlying `MegatronTrainRayActor` worker exited with Ray `SYSTEM_ERROR` / EOF.
- That means the actor-hookup boundary was not deterministically fixed. The previous foreground success was real, but it was not sufficient evidence that the underlying worker crash was gone.
- Because the run never reached a stable training loop, it still did not produce meaningful training metrics even though W&B run creation succeeded.

The most important lesson from this pass is methodological: on this ROCm stack, a healthy 9-minute foreground validation is enough to clear some boundaries, but it is not enough to claim the `set_rollout_manager` crash is resolved. That specific worker death is flaky and needs native-process-level evidence rather than only controller-side logs.

### Key Points

- Ray control-plane overload is no longer the leading issue on this setup.
- The remaining blocker is again the `MegatronTrainRayActor` worker dying during rollout-manager hookup.
- The failure is intermittent: the same code path can survive foreground validation and still die in a longer run.
- W&B setup is basically working; the lack of useful metrics is because the run does not reach stable training.

### Links

- Overnight log: `log/amd-qwen3-4b-2gpu-20260416_153631.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-15 — AMD MI210 Relax E2E Bring-Up

**Type:** Observation
**General description:** The current MI210 pass removed another brittle offload wake-up transition, reduced Ray control-plane load, and the foreground validation finally stayed alive through both the old actor and GCS failure windows.

### Details

The next debugging pass started from the remaining `MegatronTrainRayActor` death during late startup. The earlier sync-path fix had already removed the fully-async rollout-manager wiring from non-`fully_async` runs, but the offloaded Megatron actor was still going to sleep at the end of `_init()`. That meant the first real rollout-manager RPC immediately had to wake the actor again during bootstrap. The actor path now tracks `_is_sleeping`, only calls `wake_up()` when the actor is actually asleep, and leaves the sync offloaded actor resident after `_init()` until rollout-manager hookup completes.

Once that was fixed, the run advanced farther and exposed a different late-stage failure in Ray itself: the driver lost contact with GCS during rollout startup. Ray logs showed `Deadline Exceeded` while Serve tried to fetch resource usage, large control-plane queueing delays, many idle Python workers, and finally `Failed to connect to GCS within 60 seconds`. The AMD launcher and Relax Ray wrappers were then tightened to reduce control-plane load:

- internal Relax actors/managers now disable Ray task events where they are not needed
- the Ray head is capped to a smaller CPU count appropriate for this 2-GPU job
- GCS reconnect timeouts are increased so short stalls do not hard-kill the job

The follow-up foreground validation used the same MI210 launcher with a 540-second timeout. It again cleared actor initialization, rollout-manager hookup, and rollout-side SGLang launch, and this time it stayed alive through the former GCS failure window. The shell exited only because `timeout 540s` expired while SGLang weight loading was still progressing. After that healthy validation, the long run was restarted in tmux session `tmux-6`.

### Key Points

- The remaining actor crash on the sync offload path was caused by an unnecessary sleep/wake transition during bootstrap.
- The later GCS timeout was a Ray control-plane overload issue, not a new model/runtime regression.
- The current patch set survived both former failure windows in a 9-minute foreground validation and was promoted to a tmux run.

### Links

- Actor lifecycle patch: `relax/backends/megatron/actor.py`
- Ray hookup patch: `relax/distributed/ray/train_actor.py`
- Ray load-reduction patches: `relax/distributed/ray/*.py`, `relax/core/controller.py`
- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Troubleshooting: `references/troubleshooting.md`

**Type:** Observation
**General description:** The latest MI210 pass removed another startup-only cross-actor step from the sync training path and the foreground validation finally stayed alive beyond the previous actor-death window.

### Details

After the SGLang transformers import issues were fixed, the next production failure moved later: the run brought up actor, rollout, RolloutManager, and the rollout-side SGLang server, then died when the controller called `Actor.set_rollout_manager`. The nested failure boundary was `MegatronTrainRayActor.set_rollout_manager`, but the actor worker still exited with a raw Ray `SYSTEM_ERROR` / EOF rather than a surfaced Python exception.

Tracing the code path showed that `TrainRayActor.set_rollout_manager()` always performed two rollout-manager round-trips intended for the fully-async DCS path: pushing `train_parallel_config` to the rollout manager and retrieving the distributed weight-sync lock. In the non-`fully_async` training path, those values are not consumed later. The method now returns early for sync runs after storing the rollout-manager handle, and keeps the extra setup only for fully-async mode.

That change is covered by a targeted regression test. More importantly, the next foreground validation used a longer timeout and stayed alive past the previous actor-death point. It reached rollout-side SGLang launch and remained healthy until the shell-side `timeout 540s` expired, which is the strongest validation so far that the old sync `set_rollout_manager` crash boundary moved.

### Key Points

- The previous late startup failure was in sync-path rollout-manager wiring, not in actor model initialization or rollout bring-up themselves.
- The sync path no longer performs fully-async rollout-manager setup.
- The MI210 foreground validation now survives beyond the old `set_rollout_manager` death point.

### Links

- Relax patch: `relax/distributed/ray/train_actor.py`
- Regression test: `tests/distributed/ray/test_train_actor.py`
- Troubleshooting: `references/troubleshooting.md`

**Type:** Observation
**General description:** The current MI210 pass fixed the next rollout-side SGLang creation failure by making the generic transformers model import independent of optional Quark `aiter` kernels.

### Details

Once the rollout-side SGLang subprocess stopped discovering the local `Megatron-LM` checkout, the next failure became explicit: the SGLang scheduler died during `SGLangEngine.init()` with `ValueError: Model architectures ['TransformersForCausalLM'] are not supported for now`. The root cause was not the model itself. The generic transformers backend class never registered because importing `sglang.srt.models.transformers` pulled in `sglang.srt.layers.moe.ep_moe.layer`, which hard-imported the Quark MXFP4 MoE scheme. That scheme imports `aiter`, which is not present on this machine.

The local SGLang checkout was patched so `ep_moe.layer` treats `QuarkW4A4MXFp4MoE` as optional. When the optional import fails, it now logs that Quark MXFP4 MoE support is unavailable and uses an empty type tuple for the Quark-specific checks instead of aborting module import. A direct Python import with the local SGLang checkout on `PYTHONPATH` now succeeds for `sglang.srt.models.transformers`.

After that patch, the full AMD launcher was re-run in foreground. It no longer failed at the earlier `TransformersForCausalLM` boundary. The run progressed through actor deployment, rollout deployment, RolloutManager creation, SGLang server launch, and returned control while the Ray job was still in `RUNNING` state.

### Key Points

- The rollout-side failure was caused by optional Quark/`aiter` import coupling, not by a true lack of transformers-model support for Qwen3.
- Making the Quark MXFP4 MoE import optional allowed the generic SGLang transformers backend to load on this MI210 ROCm machine.
- The foreground validation now reaches a live Ray job instead of failing during rollout creation.

### Links

- Runtime patch: `relax/backends/sglang/sglang_engine.py`
- Local SGLang patch: `../sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
- Troubleshooting: `references/troubleshooting.md`

**Type:** Observation
**General description:** The current MI210 actor-death boundary moved again after isolating the rollout-side SGLang subprocess from the local `Megatron-LM` checkout.

### Details

The next debugging pass focused on the generic Ray `ActorDiedError` that appeared while rollout initialization was still in progress. Ray core logs showed the `MegatronTrainRayActor` died before the later `set_rollout_manager` call, and the worker stderr did not contain a surfaced Python exception. The strongest correlation in the failing run was that the SGLang child, despite `--sglang-model-impl transformers`, was still logging imports from the local `Megatron-LM` checkout and entering Megatron-FSDP-related code.

To keep the actor-side local Megatron import path intact without letting the rollout-side SGLang child discover it, Relax now strips `Megatron-LM` entries from `PYTHONPATH` only while spawning the SGLang server process for the `transformers` backend. The AMD launcher also pins SGLang to the transformers implementation and disables CUDA graph capture on that path.

The follow-up foreground validation used the same `relaxrl` environment and MI210 2-GPU launcher with a 420-second timeout. Unlike the previous run, it stayed alive through the old actor-death window, created a live W&B run, launched the SGLang HTTP server, and logged `Removed Megatron-LM from PYTHONPATH for SGLang transformers backend` before the validation timeout ended the process. That does not yet prove a full successful training loop, but it does show that the previous rollout-startup crash boundary moved forward again.

### Key Points

- The previous generic Ray actor death was not actually caused by `set_rollout_manager`; the actor had already died earlier.
- The rollout-side SGLang child was accidentally discovering the local `Megatron-LM` checkout through inherited `PYTHONPATH`.
- Isolating the child `PYTHONPATH` moved the run past the earlier actor-death window in foreground validation.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Runtime patch: `relax/backends/sglang/sglang_engine.py`
- Troubleshooting: `references/troubleshooting.md`

**Type:** Retrospective
**General description:** The latest MI210 debugging pass converted the remaining failures from startup deadlocks and actor OOMs into a partially healthy end-to-end bring-up, but it still did not produce a completed train loop or useful W&B metrics.

### Details

This pass used the AMD Qwen3-4B 2-GPU launcher on MI210 (`gfx90a`) with GRPO, W&B online mode, and a single-GPU actor plus single-GPU rollout layout. There were no standalone `training_reports/` artifacts, so the retrospective is based on the launcher config and the live run logs.

The main fixes and observations were:
- Local proxy handling was incomplete. Internal requests to node-local SGLang endpoints were being sent through the cluster proxy until the launcher/runtime env started appending localhost plus resolved node addresses to `no_proxy` and `NO_PROXY`.
- ROCm TorchInductor failures in the PPO helper path were reduced by avoiding `torch.compile(dynamic=True)` for the small HIP-sensitive helper functions.
- The actor no longer died on lazy Adam state allocation after the AMD launcher switched to Megatron CPU optimizer offload with the matching precision-aware flags.
- Enabling CPU optimizer offload exposed a local Megatron bug where `is_te_min_version()` crashed if Transformer Engine was absent; the local Megatron checkout was patched so TE absence is treated as an unsupported capability instead of a type error.
- After those fixes, actor initialization completed successfully again, rollout deployment proceeded further, and the latest validation run remained alive long enough to show that the previous OOM and TE-version boundaries were cleared.

### Key Points

- The current AMD recipe progressed farther than the previous attempts and removed several concrete blockers: proxy interception, PPO HIP compiler crashes in helper functions, actor-side Adam OOM, and the CPU-offload TE capability crash.
- The run still did not yield a clean finished training loop or useful W&B training logs, so the stack should be treated as "advanced bring-up, not yet stable end-to-end training."

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Latest validation log: `log/amd-qwen3-4b-2gpu-20260415_171113.log`
- Troubleshooting: `references/troubleshooting.md`

**Type:** Observation
**General description:** The next MI210 pass cleared the proxy stall, the PPO ROCm compiler crash in the hot helpers, and the actor-side optimizer memory failure by switching the AMD launcher to CPU optimizer offload and patching a local Megatron TE-version guard.

### Details

The concrete sequence for the follow-up pass was:
- Local SGLang health checks were being intercepted by the cluster proxy because the node hostname/IP was not in `no_proxy`; Relax runtime env propagation and the AMD launcher now append localhost plus resolved local addresses to `no_proxy` and `NO_PROXY`.
- After that, the run moved into actor training and exposed ROCm TorchInductor failures in small compiled PPO helpers. The HIP path now avoids `torch.compile(dynamic=True)` for those helpers.
- The next blocker was not compilation but memory: the first Adam state allocation OOMed on the single-GPU actor rank. The AMD launcher now uses Megatron CPU optimizer offload with precision-aware optimizer support.
- Enabling CPU optimizer offload exposed a separate bug in the local Megatron checkout: `is_te_min_version()` crashed when Transformer Engine was absent. That capability check was patched so TE absence returns `False` instead of raising.
- With those fixes in place, the actor initialized successfully again, the rollout service came back up, and the validation run progressed beyond the previous OOM/TE crash boundary.

### Key Points

- The current AMD recipe is no longer blocked by proxy interception, lazy Adam-state allocation OOM, or the TE-version crash in the offload path.
- The latest foreground validation cleared all of those earlier failures and remained healthy through actor initialization and rollout bring-up.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Troubleshooting: `references/troubleshooting.md`
- Logs: `log/amd-qwen3-4b-2gpu-20260415_171113.log`

**Type:** Retrospective
**General description:** Brought the Relax Qwen3-4B GRPO stack onto an MI210 ROCm machine and iterated through repeated overnight startup failures until the remaining blocker was isolated to a later Megatron/TorchInductor runtime path.

### Details

We set up a dedicated `relaxrl` conda environment, copied the nearby `.env` so W&B credentials were available, and validated the full 2-GPU `actor + rollout` path with Ray Serve, SGLang, and Megatron. Early failures were dominated by ROCm-incompatible SGLang and weight-conversion paths; later failures moved into actual training execution after rollout generation and reward computation were already working.

The concrete progression was:
- SGLang import and JIT/kernel bring-up failed on ROCm due to CUDA-only assumptions. Local ROCm-compatible patches in the checked-out `sglang` tree plus a locally built `sgl-kernel` for `gfx90a` were required.
- Relax weight conversion then failed on Qwen layernorm parameter aliases during actor-to-rollout weight sync. Extending the Qwen2 converter to accept the current Megatron names fixed that stage.
- Packed-sequence training then failed with plain `DotProductAttention`, so the AMD launcher was switched away from THD packed-sequence flow to `bshd`.
- After that, Megatron failed on NVIDIA fused masked softmax imports. Disabling masked-softmax fusion in the launcher was not sufficient because the Megatron-Bridge provider override path did not propagate `masked_softmax_fusion`; that path was patched.
- The next observed blocker, after the run reached real rollout generation and reward execution, was a TorchInductor/Triton runtime failure on ROCm: `KernelMetadata` missing `cluster_dims`.

### Key Points

- The system now consistently reaches: Ray startup, DCSCoordinator, Actor ready, Rollout ready, SGLang weight load, CUDA graph capture, rollout generation, reward execution, and initial actor training step.
- The current remaining blocker is no longer infrastructure bring-up. It is a later-stage training runtime issue in the Megatron/TorchInductor path on this ROCm stack.
- Overnight retries confirm the failure class moved from install/startup incompatibilities to training-time execution incompatibilities.

### Links

- Logs: `log/amd-qwen3-4b-overnight-attempt-*.log`
- Status ledger: `results.tsv`

## 2026-04-17 — Retrospective on AMD rollout import-surface cleanup

**Type:** Retrospective
**General description:** This retrospective covers the latest MI210 debugging passes, where the main progress came from turning several rollout-side and control-plane Megatron import leaks into direct code fixes with regression tests.

### Details

This retrospective is based on the current AMD bring-up notes and the latest
clean foreground validation log:

- `references/troubleshooting.md`
- `log/amd-qwen3-4b-2gpu-20260417_103842.log`

There were no `training_reports/` entries for this pass, so the source of truth
is the experiment log plus the runtime logs.

What we changed in this stretch:

- converted `relax.distributed.checkpoint_service` package exports to lazy
  `__getattr__` lookups so coordinator/control-plane imports do not eagerly
  pull in `device_direct.py` and `megatron.core`
- moved Megatron-specific imports in `relax.components.advantages` into the
  PPO/OPD branches that actually need them, so `relax.core.registry` and
  `relax.core.controller` stay lightweight at import time
- moved transformers-mode Megatron isolation earlier into both
  `Rollout.__init__` and `RolloutManager.__init__`, rather than relying only on
  the later `SGLangEngine` path

Regression coverage added or extended in this stretch:

- `tests/distributed/checkpoint_service/test_imports.py`
- `tests/components/test_advantages_imports.py`
- `tests/core/test_registry_imports.py`
- `tests/distributed/ray/test_rollout.py`
- `tests/backends/sglang/test_sglang_engine.py`

The focused validation suite was green, and the clean 5-minute AMD validation
advanced farther than before:

- actor initialization succeeded
- rollout deployed
- rollout-side isolation logged explicitly in both `Rollout` and
  `RolloutManager`
- `SGLangEngine` reached `Load weight begin` and started shard loading with the
  transformers backend

What remains unresolved:

- `ServeReplica:rollout:Rollout` still emitted Megatron/TE warnings before its
  isolation log line
- `RolloutDataSourceWithBuffer` still showed Megatron-related warnings in prior
  runs on this branch of investigation
- the newest clean validation is no longer running, and the captured artifacts
  still do not show a final explicit crash line after `Load weight begin`

### Key Points

- Several high-level import leaks were real and are now fixed in code rather
  than being treated as vague startup noise.
- The remaining AMD blocker is narrower than before: there is still at least
  one rollout/bootstrap contamination path left, but the stack now gets into
  real SGLang weight loading on the clean path.
- The next debugging pass should focus on the last shared worker/bootstrap
  contamination path instead of running more blind overnight retries.

### Links

- Runtime files: `relax/distributed/checkpoint_service/__init__.py`, `relax/distributed/checkpoint_service/backends/__init__.py`, `relax/components/advantages.py`, `relax/components/rollout.py`, `relax/distributed/ray/rollout.py`
- Tests: `tests/distributed/checkpoint_service/test_imports.py`, `tests/components/test_advantages_imports.py`, `tests/core/test_registry_imports.py`, `tests/distributed/ray/test_rollout.py`, `tests/backends/sglang/test_sglang_engine.py`
- Validation log: `log/amd-qwen3-4b-2gpu-20260417_103842.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — Step-0 crash instrumentation and train-only replay path

**Type:** Observation
**General description:** The remaining MI210 failure is now firmly in the first Megatron train step. This pass added pre-train batch persistence, train-step boundary logs, and env-gated dump/replay controls in the AMD launcher so the next debug run can isolate the native crash without rebuilding rollout every time.

### Details

The latest tmux run still died with `MegatronTrainRayActor` EOF/SystemError during `async_train(step=0)`, after `train_0` had already been transferred and rollout metrics had been logged. The key problem for debugging was that the existing debug train dump happened only after `train(...)` returned, so a step-0 native crash lost the exact batch that triggered it.

The concrete changes in this pass were:

- add actor-side boundary logs for:
  - data iterator creation
  - `compute_advantages_and_returns`
  - rollout-stat logging
  - entry into `train(...)`
  - completion of `train(...)`
- add model-side boundary logs for:
  - start/end of `forward_backward_func(...)`
  - start/end of `optimizer.step()`
  - scheduler step
  - final loss reduction
- save debug train data before entering the native Megatron train call, so a crashing batch is still preserved
- add env-gated AMD launcher flags for:
  - `RELAX_SAVE_DEBUG_ROLLOUT_DATA`
  - `RELAX_SAVE_DEBUG_TRAIN_DATA`
  - `RELAX_LOAD_DEBUG_ROLLOUT_DATA`
  - `RELAX_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE`
  - `RELAX_DUMP_DETAILS`

This keeps a single launcher path while making it possible to capture one good rollout batch and then replay the actor in train-only mode via `--load-debug-rollout-data`.

### Key Points

- The debugging target is now the exact first train-step subphase, not rollout startup.
- The repo can now preserve the failing train input even if the worker dies before returning from `train(...)`.
- The next high-signal run is a capture/replay cycle, not another blind overnight e2e attempt.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Actor: `relax/backends/megatron/actor.py`
- Train step: `relax/backends/megatron/model.py`
- Troubleshooting: `references/troubleshooting.md`

<!-- New entries go above this line -->
## 2026-04-17 — Durable SGLang import isolation for rollout on MI210

**Type:** Observation
**General description:** The previous SGLang import-path fix was too weak because it only filtered `Megatron-LM` around `p.start()`. This pass tightened the rollout-side isolation so the SGLang child imports its server code under a filtered path for the full server lifetime, and the 5-minute MI210 validation cleared without reproducing the old `Megatron-FSDP` marker in the SGLang logs.

### Details

The failing `tmux-9` run made the problem visible. Although the rollout path logged that it had removed `Megatron-LM` from `PYTHONPATH` and `sys.path`, the later SGLang startup still printed:
- `Using Megatron-FSDP without Transformer Engine.`
- `Detected Megatron Core, using Megatron-FSDP with Megatron.`

That showed the earlier fix only covered the `multiprocessing.Process(...).start()` boundary; later SGLang imports were still happening after the broader import path had been restored. The concrete changes in this pass were:
- move the `sglang.srt.entrypoints.http_server` import behind top-level wrapper functions that apply the `Megatron-LM` filter inside the spawned SGLang child before importing the server module
- keep that filter active for the entire `launch_server(...)` call rather than restoring it immediately after `p.start()`
- add a second guard in the rollout actor runtime env so rollout-side `PYTHONPATH` is already filtered when `--sglang-model-impl transformers` is used
- add focused regression coverage for the filtered `PYTHONPATH` helper and the rollout-engine runtime-env builder

Validation for this pass:
- `python -m pytest -q tests/backends/sglang/test_sglang_engine.py` passed (`4 passed`)
- `python -m pytest -q tests/distributed/ray/test_rollout.py` passed (`2 passed`)
- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh` passed the required foreground validation window
- during that validation, the active launcher log no longer contained `Using Megatron-FSDP` or `Detected Megatron Core` for the SGLang startup path

### Key Points

- The rollout-side import leak was real, but the temporary `p.start()`-only filter was insufficient.
- The current fix is stronger because it narrows the import path both before the child is spawned and inside the child while the SGLang server is importing and starting.
- A new long run was started in `tmux-1` after the strengthened foreground validation succeeded.

### Links

- SGLang launch code: `relax/backends/sglang/sglang_engine.py`
- Rollout runtime env: `relax/distributed/ray/rollout.py`
- Tests: `tests/backends/sglang/test_sglang_engine.py`, `tests/distributed/ray/test_rollout.py`
- Validation log: `log/amd-qwen3-4b-2gpu-20260417_063102.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-17 — Ray keepalive mitigation for long SGLang startup

**Type:** Observation
**General description:** Reframed the remaining MI210 failure from a misleading `MegatronTrainRayActor.set_rollout_manager` death into a Ray keepalive/control-plane stall during rollout-side SGLang startup, then added launch-time Ray tuning to reduce that stall and revalidated with a healthy 5-minute foreground run.

### Details

The previous overnight log showed the controller eventually failing while calling `Actor.set_rollout_manager`, but the worker-side Ray logs told a different story. The first hard failure was earlier, during rollout replica startup: `SGLangEngine.init()` stalled long enough for Ray to emit `keepalive watchdog timeout` on actor RPCs, and only afterward did the controller observe collateral actor deaths. The raylet and worker state dumps showed sustained `NodeManagerService.grpc_server.ReportWorkerBacklog` pressure, which pointed to a control-plane problem rather than a direct Python exception in the Megatron actor.

The concrete changes for this pass were:
- export `RAY_task_events_report_interval_ms=0` in the AMD launcher to suppress periodic Ray task-event reporting that was not useful for this run
- widen Ray gRPC client keepalive settings in the launcher with `RAY_grpc_client_keepalive_time_ms=600000` and `RAY_grpc_client_keepalive_timeout_ms=300000`
- propagate those Ray env vars through `post_process_env()` so submitted Ray jobs, Serve replicas, and child workers inherit the same tuning
- add focused test coverage for the new runtime-env propagation path

Validation on this pass was stronger than the failing overnight trace:
- `python -m pytest -q tests/utils/test_runtime_env.py` passed (`3 passed`)
- `bash -n amd_qwen3_4b_2gpu_e2e.sh` passed
- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh` stayed alive through actor initialization, rollout deployment, RolloutManager startup, W&B reconnect, and SGLang launch; it exited only because the shell timeout fired

### Key Points

- The apparent `set_rollout_manager` crash boundary was not the primary cause; the earlier failure surface was Ray keepalive loss during long SGLang startup.
- The new Ray launch/runtime tuning did not fully prove end-to-end success yet, but it was enough to clear the previous 5-minute validation boundary without reproducing the old watchdog failure.
- A new long run was started in `tmux-7` after the foreground validation succeeded.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Runtime env helper: `relax/utils/utils.py`
- Validation log: `log/amd-qwen3-4b-2gpu-20260417_041407.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-18 — Conservative AMD train-step launcher for the first Megatron backward

**Type:** Observation
**General description:** The MI210 run now survives reward execution and transfer-queue handoff into real actor training, but the Megatron worker still dies before the first `train/...` metric. I narrowed that boundary to the first actor-side train kernel path and tightened the AMD launcher accordingly.

### Details

The latest failed run no longer died at rollout bring-up or in the reward-worker pool. The concrete progression was:

- rollout generation succeeded
- `train_0` was transferred successfully
- the Megatron actor logged `start to get rollout_id: 0 data from transfer queue for train with mcore`
- rollout statistics were logged from `relax.backends.megatron.data`
- then `MegatronTrainRayActor` died with Ray `ActorDiedError` / EOF and no Python traceback

That makes the next failure surface much narrower: the first real Megatron backward / optimizer path on ROCm. Since there was still no actionable Python exception, I made the AMD launcher more conservative instead of guessing at deeper internals:

- switched actor training from `--attention-backend auto` to `--attention-backend unfused`
- added `--no-bias-dropout-fusion`
- removed `--sequence-parallel` from this single-GPU actor path
- made `--micro-batch-size 1` explicit
- reduced the train-step load from `global_batch_size=16` to `global_batch_size=8` with `--num-steps-per-rollout 2`
- reduced `--rollout-max-response-len` from `1024` to `768`

I first tried `--use-dynamic-batch-size`, but that failed immediately during argument validation because Relax does not support dynamic batching together with the required `--qkv-format bshd` path. I then replaced that invalid knob with the supported smaller-step configuration above.

These changes are specifically aimed at the first-step native failure window on MI210 rather than at startup or rollout behavior.

### Key Points

- The current blocker is no longer reward dispatch or rollout transfer.
- The actor reaches real training and then dies before logging its first train-step metrics.
- The AMD launcher now prefers the safest available Megatron attention/microbatch path for this boundary.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Failing run log: `log/amd-qwen3-4b-2gpu-20260418_042431.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-19 — Safer single-rank ROCm CPU-offload optimizer path

**Type:** Observation
**General description:** The step-0 MI210 crash was narrowed to the CPU-offload optimizer path after forward/backward completed. I changed the single-rank ROCm actor path to use safer host-buffer and CPU-optimizer settings, then revalidated with focused tests plus a clean 5-minute AMD foreground run.

### Details

The strongest worker-side evidence from the failing run was:

- `train_one_step rollout=0 step=0: finished forward_backward`
- `train_one_step rollout=0 step=0: starting optimizer.step`
- then the `MegatronTrainRayActor` worker exited with Ray EOF / `ActorDiedError`

That ruled out rollout generation, reward execution, and the Megatron forward/backward path for this boundary. The remaining suspicious component was Megatron's `HybridDeviceOptimizer` on a single-rank ROCm actor, especially because the launcher still required CPU optimizer offload to fit the actor on MI210.

The code changes in this pass were:

- add a single-rank ROCm CPU-offload safety override in `relax/backends/megatron/optimizer_utils.py`
- for that path, force:
  - `use_torch_optimizer_for_cpu_offload=True`
  - `pin_cpu_grads=False`
  - `pin_cpu_params=False`
- add a torch `AdamW` wrapper that strips Megatron-only kwargs (`bias_correction`, `fused`)
- patch Megatron's CPU-offload builder at runtime so `CPUAdam` resolves to that safe wrapper for the single-rank ROCm actor path
- wire the new safety override and patch into `setup_model_and_optimizer()` before constructing the optimizer

Validation in this pass:

- `python -m pytest -q tests/utils/test_megatron_model.py tests/distributed/ray/test_train_actor.py`
  - result: `17 passed`
- `bash -n amd_qwen3_4b_2gpu_e2e.sh`
  - result: passed
- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - result: exit code `124`
  - the run stayed healthy through cluster bootstrap, rollout startup, actor creation, and into actor checkpoint loading without reproducing an immediate optimizer-step death inside the 5-minute window

I also tried the saved train-only replay path with:

- `RELAX_LOAD_DEBUG_ROLLOUT_DATA=log/debug_capture_20260418_153613/rollout_data/{rollout_id}.pt`

That replay surfaced a different issue: an actor-init HIP OOM during DDP buffer allocation on GPU 0 before the train step even began. That replay-specific OOM is useful to know about, but it does not negate the main optimizer safety fix because the full launcher path cleared its foreground gate from a clean cluster.

### Key Points

- The optimizer-step crash boundary is now treated as a CPU-offload safety problem, not a rollout problem.
- The safer ROCm CPU-offload mode is narrower than a general launcher change: it only applies to the single-rank ROCm actor path.
- The replay path still has a separate actor-init OOM corner case on GPU 0.

### Links

- Optimizer helpers: `relax/backends/megatron/optimizer_utils.py`
- Model setup: `relax/backends/megatron/model.py`
- Tests: `tests/utils/test_megatron_model.py`, `tests/distributed/ray/test_train_actor.py`
- Foreground validation log: `log/amd-qwen3-4b-2gpu-20260419_104819.log`
- Replay capture: `log/debug_capture_20260418_153613`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-19 — Actor post-load backup boundary instrumentation

**Type:** Observation
**General description:** The next MI210 actor failure window moved out of `optimizer.step()` and back into actor initialization after checkpoint load. I instrumented the post-load backup path, added a safer non-pinned backup option for ROCm callers, and re-ran the clean 5-minute AMD validation.

### Details

The failure pattern before this pass was:

- `MegatronTrainRayActor` died during actor service creation
- the last visible logs were repeated `Load checkpoint from HuggingFace model into Megatron`
- there was no Python traceback after that

The actor code right after model init immediately does four heavyweight things:

1. create `TensorBackuper`
2. back up actor weights
3. load and back up the ref checkpoint
4. create the rollout weight updater

So this pass made that path explicit:

- `TensorBackuper.create()` now accepts `pin_memory=...`
- `_TensorBackuperNormal` respects that flag instead of always allocating pinned CPU tensors
- `relax/backends/megatron/actor.py` now logs:
  - weight backup initialization
  - start/end of actor backup
  - start/end of ref / teacher / old-actor backup
  - start/end of updater construction
- `relax/backends/megatron/optimizer_utils.py` now exposes `should_disable_pinned_host_weight_backups()` for single-rank ROCm actor safety decisions

Validation in this pass:

- `python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result: `20 passed`
- `bash -n amd_qwen3_4b_2gpu_e2e.sh`
  - result: passed
- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - result: exit code `124`
  - the run survived through:
    - rollout startup
    - actor startup
    - Megatron checkpoint load
    - initial actor backup
    - entry into ref checkpoint load

The new stage logs show the actor got farther than the old opaque boundary before the foreground timeout ended.

### Key Points

- The current MI210 boundary is no longer an unobservable actor-init death.
- We now have exact stage markers for the post-load actor path.
- The clean 5-minute gate still passes from a fresh cluster.

### Links

- Actor init path: `relax/backends/megatron/actor.py`
- Weight backup utility: `relax/utils/training/tensor_backper.py`
- Safety helper: `relax/backends/megatron/optimizer_utils.py`
- Tests: `tests/utils/test_megatron_model.py`, `tests/utils/test_tensor_backper.py`, `tests/distributed/ray/test_train_actor.py`
- Foreground validation log: `log/amd-qwen3-4b-2gpu-20260419_115632.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-19 — Disable pinned host backups for single-rank ROCm actor init

**Type:** Observation
**General description:** The previous actor-init crash still happened after the first actor backup and during the ref-checkpoint path. The dead worker logs showed the actor was still using `pinned_host_weight_backups=True`, so I broadened the safety rule to disable pinned host backups for the single-rank ROCm actor path entirely.

### Details

The decisive evidence from the dead worker log was:

- `Finished backing up actor weights`
- `Loading ref checkpoint into actor-side backup state`
- then no further actor-side logs before EOF / `ActorDiedError`

That strongly suggested the pinned host backup path was still too fragile for the real ROCm actor init sequence, especially because it would need to hold multiple model snapshots (`actor`, then `ref`) on the host side.

The code change in this pass was:

- change `should_disable_pinned_host_weight_backups()` so it applies to the single-rank ROCm actor path regardless of `offload_train`
- keep the new stage logs inside `load_other_checkpoint()` so the next boundary is still visible if the actor dies later

Validation in this pass:

- `python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result: `20 passed`
- `bash -n amd_qwen3_4b_2gpu_e2e.sh`
  - result: passed
- `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - result: exit code `124`
  - the real actor path now logs:
    - `Initializing weight backup state for actor path (pinned_host_weight_backups=False)`
    - `Backing up actor weights after Megatron initialization`

So the actual launcher path is now using the safer non-pinned host backup configuration, and the 5-minute foreground gate still passes from a clean cluster.

### Key Points

- The actor is no longer using pinned host snapshots on the single-rank ROCm path.
- The 5-minute foreground validation remains healthy from a fresh cluster.
- If the run still dies later, the next boundary should now be later than the old ref-backup crash window and better instrumented.

### Links

- Actor init path: `relax/backends/megatron/actor.py`
- Safety helper: `relax/backends/megatron/optimizer_utils.py`
- Weight backup utility: `relax/utils/training/tensor_backper.py`
- Tests: `tests/utils/test_megatron_model.py`, `tests/utils/test_tensor_backper.py`, `tests/distributed/ray/test_train_actor.py`
- Foreground validation log: `log/amd-qwen3-4b-2gpu-20260419_152335.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 — HybridDeviceOptimizer pinned CPU params bug

**Type:** Observation
**General description:** The overnight run proved the actor-init fixes held: the stack reached real rollout generation, reward transfer, and the first Megatron train step. The remaining wedge was still at `optimizer.step()`, and the next concrete issue turned out to be inside the local Megatron `HybridDeviceOptimizer`.

### Details

What the overnight run showed:

- rollout completed step 0 generation
- the actor consumed `train_0`
- the actor logged:
  - `finished forward_backward`
  - `starting optimizer.step`
- then the run wedged/orphaned with no later actor-side progress

That ruled out the old startup and ref-backup boundaries. The next useful inspection was the local Megatron CPU-offload optimizer code. There, `_get_sub_optimizer_param_groups()` was still doing:

```python
param.detach().clone().cpu().pin_memory()
```

for offloaded parameter copies unconditionally, even though the Relax-side ROCm safety path had already set `pin_cpu_params=False`.

So this pass patched the local Megatron checkout to respect `self.pin_cpu_params` before calling `.pin_memory()`.

Validation in this pass:

- `python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result: `20 passed`
- `bash -n amd_qwen3_4b_2gpu_e2e.sh`
  - result: passed
- clean foreground AMD run:
  - `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - submission and startup succeeded
  - the validation remained healthy through startup and into actor initialization again

The old overnight artifact is still useful because it narrowed the remaining failure to the real optimizer/update path rather than the earlier actor-init path.

### Key Points

- The current blocker is the step-0 optimizer path, not startup.
- Relax-side `pin_cpu_params=False` was not enough on its own because the local Megatron optimizer ignored it.
- The local Megatron CPU-offload path now respects that knob.

### Links

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Relax safety helper: `relax/backends/megatron/optimizer_utils.py`
- Tests: `tests/utils/test_megatron_model.py`, `tests/utils/test_tensor_backper.py`, `tests/distributed/ray/test_train_actor.py`
- Overnight run log: `log/amd-qwen3-4b-2gpu-20260419_152924.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - HybridDeviceOptimizer async copies still assumed pinned host buffers

After the local Megatron patch that made `_get_sub_optimizer_param_groups()` respect `pin_cpu_params=False`, I reran the clean production path in `tmux-2`. That run still died at the same narrow boundary:

- `train_one_step rollout=0 step=0: finished forward_backward`
- `train_one_step rollout=0 step=0: starting optimizer.step`
- then Ray reported `MegatronTrainRayActor` EOF / `SYSTEM_ERROR`

The dead worker log for pid `3009002` showed no Python traceback after `starting optimizer.step`, but it did confirm the intended ROCm safety args were active:

- distributed optimizer overlap features were disabled for the single-rank actor path
- `pin_cpu_grads=False`
- `pin_cpu_params=False`
- the local CPUAdam patch was active

That pushed the suspicion one level deeper into the local Megatron `HybridDeviceOptimizer`. Reading the code showed that the optimizer still used `non_blocking=True` for host/device copies even when the corresponding host buffers were no longer pinned:

- GPU grad -> CPU grad copy
- CPU param -> GPU param copy-back

So this pass patched the local Megatron checkout again:

- grad copies now use `non_blocking=self.pin_cpu_grads`
- param copy-back now uses `non_blocking=self.pin_cpu_params`

This is a more coherent ROCm safety path: disabling pinned host buffers now also disables the async copies that assume those buffers are pinned.

Immediate next step after this note: rerun the clean 5-minute foreground gate from a fresh Ray cluster and check whether the actor finally survives past `optimizer.step`.

### Links

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Dead worker logs: `/tmp/ray/session_latest/logs/worker-d2fe17a9f8054d440898398f654fbad24397529d7a040e63eb260ff1-02000000-3009002.err`
- Core worker log: `/tmp/ray/session_latest/logs/python-core-worker-d2fe17a9f8054d440898398f654fbad24397529d7a040e63eb260ff1_3009002.log`

## 2026-04-20 - Clean replay exposed stale-driver contamination, then moved the optimizer crash deeper

**Type:** Observation
**General description:** A supposedly clean replay still carried old Relax driver processes outside Ray's process tree. After fixing that launcher contamination, the single-rank ROCm actor still died in `optimizer.step()`, but the failure moved much deeper once CPU-offloaded params were stepped sequentially.

### Details

What happened first:

- a replay that looked clean was still emitting `Mismatched WorkerID` noise
- process inspection showed stale `ray job submit` and `python3 -m relax.entrypoints.train` processes survived earlier retries
- that proved `ray stop --force` alone was not enough to guarantee a clean Relax retry

The launcher was then updated to kill only stale Relax-specific driver / job-submit processes before starting Ray.

After that cleanup fix, the next clean replay removed the old contamination but still died at the same training boundary:

- `train_one_step rollout=0 step=0: finished forward_backward`
- `train_one_step rollout=0 step=0: starting optimizer.step`

The local Megatron optimizer was instrumented more tightly. That showed:

- the old first CPU grad copy boundary was gone
- the actor now survived many sequential CPU sub-optimizer updates
- the crash moved as far as:
  - `HybridDeviceOptimizer step: cpu sub-optimizer 46 step begin`

with no matching `step done` before the worker died with Ray EOF / `SYSTEM_ERROR`.

That established two important facts:

1. The stale-driver contamination was real, but not the root cause of the optimizer death.
2. The real ROCm bug is deeper than first grad staging; it persists later in the sequential CPU-offload update loop.

### Key Points

- `ray stop --force` was not sufficient to guarantee a clean Relax retry.
- Sequential CPU optimizer stepping was a real improvement because it moved the boundary far beyond the first host grad copy.
- The remaining death is still inside or immediately around the per-sub-optimizer step/copy-back phase.

### Links

- Launcher: `amd_qwen3_4b_2gpu_e2e.sh`
- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Clean replay log: `log/amd-qwen3-4b-2gpu-20260420_082144.log`
- Deeper-boundary replay log: `log/amd-qwen3-4b-2gpu-20260420_084049.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - Manual ROCm param copy-back replaces CPU optimizer post-hooks

**Type:** Fix
**General description:** The local Megatron optimizer now bypasses CPU optimizer post-hooks on the unpinned, non-overlap single-rank ROCm path. Instead, it copies updated CPU-offloaded params back to GPU explicitly after each CPU sub-optimizer step.

### Details

Why this change was made:

- on the latest clean replay, the worker survived dozens of sequential CPU sub-optimizers
- the last visible boundary was still:
  - `HybridDeviceOptimizer step: cpu sub-optimizer 46 step begin`
- there was no `step done`, which meant the dangerous boundary was still hidden inside `cpu_optimizer.step(...)`

That made the optimizer post-hook a likely suspect, because the CPU param copy-back was still happening implicitly inside `step()` rather than as an explicit logged phase.

So this pass changed the local Megatron `HybridDeviceOptimizer`:

- skip CPU optimizer post-hook registration on the unpinned, non-overlap ROCm path
- add an explicit `_copy_cpu_optimizer_params_back_to_gpu()` helper
- after each sequential `cpu_optimizer.step()`, log:
  - `param copy-back begin`
  - `param copy-back done`

This keeps the step math and the H2D param write-back as separate observable phases.

Validation in this pass:

- `python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result: `20 passed`

The next runtime check is a fresh bounded replay from a clean cluster to see whether the crash moves past the old `cpu sub-optimizer 46` boundary.

### Key Points

- The current ROCm optimizer path now avoids hidden CPU param copy-back inside optimizer post-hooks.
- The next replay should tell us whether the hook was the remaining unstable boundary.
- Focused Relax tests still pass after the local Megatron patch.

### Links

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Tests: `tests/utils/test_megatron_model.py`, `tests/utils/test_tensor_backper.py`, `tests/distributed/ray/test_train_actor.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - Chunk Sequential CPU Optimizers and Trim Secondary W&B Payloads

### Details

This pass addressed two separate boundaries on the MI210 path.

First, the local Megatron `HybridDeviceOptimizer` was changed again so the unpinned, non-overlap ROCm fallback no longer creates one CPU AdamW instance per single parameter. The sequential ROCm path now groups four parameters per CPU sub-optimizer. The goal is to keep staged grad sync / step / copy-back behavior while reducing first-step Adam state overhead and the number of tiny CPU optimizer instances.

Second, the actor-init failure that appeared during the next clean validation was traced to `init_wandb_secondary()`. Secondary shared-mode W&B clients were re-sending `config=args.__dict__`, even though the primary process had already created the run and uploaded config. On this machine that caused `wandb.errors.errors.CommError` during actor initialization. The fix was to keep config upload only in `init_wandb_primary()` and make secondary clients attach by run ID only.

### Validation

- `python -m pytest -q tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result after chunked optimizer change: `20 passed`
- `python -m pytest -q tests/utils/test_wandb_adapter.py tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py`
  - result after W&B secondary-init fix: `22 passed`
- clean foreground validation:
  - `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - result: exit code `124`
  - healthy boundary after W&B fix:
    - actor initialization completed
    - `set_rollout_manager` succeeded
    - `Actor training step 0/200`
    - rollout generation active
    - SGLang decode throughput active

### Key Points

- Secondary W&B shared clients should not upload full config again.
- The latest clean 5-minute validation is back past actor init and into real step-0 work.
- The remaining question is still whether the chunked ROCm optimizer path eliminates or only moves the later `optimizer.step()` death.

### Links

- W&B adapter: `relax/utils/metrics/adapters/wandb.py`
- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Log: `log/amd-qwen3-4b-2gpu-20260420_101349.log`

## 2026-04-20 - Failed SGLang init left orphaned scheduler trees on MI210

**Type:** Observation
**General description:** The next clean foreground validation did not regress in Megatron; it failed earlier because stale SGLang child processes from older failed startups were still holding GPU memory.

### Details

After the latest ROCm optimizer patch, I ran another clean foreground
validation. The visible failure moved away from the old `optimizer.step()`
boundary and into rollout startup:

- `RolloutManager.__init__()` died during `SGLangEngine.init()`
- `launch_server_process()` raised
  `Exception: Server process terminated unexpectedly.`
- the direct SGLang child traceback showed the real failure at
  `Load weight begin`:
  `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 48.00 MiB`

The machine state explained why. `fuser /dev/kfd` and `ps` showed two stale
SGLang trees from earlier failed retries were still alive:

- orphaned `sglang::scheduler`
- orphaned `sglang::detokenizer`
- orphaned `multiprocessing.spawn` parents

So the new SGLang worker was not actually the only process on the GPU. It was
trying to load weights into a GPU that still had old scheduler processes
holding most of the memory.

This pass fixed that lifecycle gap in two places:

1. `relax/backends/sglang/sglang_engine.py` now kills the full spawned process
   tree if `_wait_server_healthy()` fails during `launch_server_process()`.
2. `amd_qwen3_4b_2gpu_e2e.sh` now removes orphaned `sglang::scheduler` /
   `sglang::detokenizer` trees whose parent is an orphaned
   `multiprocessing.spawn` worker before starting a new Ray cluster.

Focused regression coverage was added for the first half of that fix so failed
SGLang bring-up cannot silently skip process-tree cleanup again.

### Key Points

- The new rollout OOM was stale VRAM, not a fresh model-fit regression.
- `ray stop --force` alone is not enough cleanup when failed SGLang startup
  leaks child trees outside Ray's direct worker hierarchy.
- The next meaningful validation is another clean 5-minute run after clearing
  those orphaned SGLang GPU holders.

### Links

- Runtime files: `relax/backends/sglang/sglang_engine.py`, `amd_qwen3_4b_2gpu_e2e.sh`
- Test: `tests/backends/sglang/test_sglang_engine.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - Revert ROCm CPU sub-optimizer grouping back to single-parameter chunks

**Type:** Fix
**General description:** The latest clean long run proved the MI210 actor still died inside a grouped CPU AdamW step, so the local Megatron ROCm fallback now returns to the smallest possible CPU optimizer unit.

### Details

The SGLang stale-process leak was fixed, and the next clean long run got all the
way back into real step-0 training. That let the optimizer-step logging narrow
the remaining death much further than before:

- `train_one_step rollout=0 step=0: finished forward_backward`
- `train_one_step rollout=0 step=0: starting optimizer.step`
- `HybridDeviceOptimizer` completed CPU sub-optimizers `0` through `11`
- the last line before actor death was:
  `HybridDeviceOptimizer step: cpu sub-optimizer 12 step begin`

That mattered because the previous compromise on this path had changed the
local Megatron optimizer to pack four parameters into each CPU AdamW instance.
The goal was to reduce the overhead of one-optimizer-per-parameter stepping.
The new logs showed that the grouped step itself was still too large or too
fragile on this MI210 ROCm path.

This pass therefore reverted the unpinned, non-overlap ROCm fallback in the
local Megatron `HybridDeviceOptimizer` back to the most conservative setting:

- `params_per_optimizer=1`

I also added a focused test in `tests/utils/test_megatron_model.py` that loads
the local Megatron `hybrid_optimizer.py` file directly and asserts the ROCm
safety path now creates one CPU sub-optimizer per parameter when:

- `pin_cpu_grads=False`
- `pin_cpu_params=False`
- `overlap_cpu_optimizer_d2h_h2d=False`

### Validation

- `python -m pytest -q tests/backends/sglang/test_sglang_engine.py tests/utils/test_megatron_model.py tests/utils/test_tensor_backper.py tests/distributed/ray/test_train_actor.py tests/utils/test_wandb_adapter.py`
  - result: `35 passed`
- clean foreground validation:
  - `timeout 300s bash ./amd_qwen3_4b_2gpu_e2e.sh`
  - result: exit code `124`
  - healthy boundary after the revert:
    - rollout and SGLang startup completed
    - actor init completed
    - `set_rollout_manager()` succeeded
    - weight sync to SGLang succeeded
    - `Actor training step 0/200` appeared
    - rollout generation and decode throughput were active before timeout

### Key Points

- The current root cause is still the local Megatron CPU-offload optimizer path,
  not Ray or SGLang startup.
- The grouped CPU AdamW compromise moved the boundary but did not close it.
- The single-parameter fallback passed the required clean 5-minute gate, so the
  next correct step is a fresh tmux production rerun from a cleaned cluster.

### Links

- Local Megatron optimizer: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- Test: `tests/utils/test_megatron_model.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - Retrospective on the single-parameter ROCm CPU-offload rerun

**Type:** Retrospective
**General description:** The most conservative local Megatron CPU-offload path passed the 5-minute gate again, but the long run still degraded into the familiar `train_0` wedge with the actor gone.

### Details

This retrospective covers the latest MI210 pass after reverting the local
Megatron `HybridDeviceOptimizer` ROCm fallback back to one CPU optimizer per
parameter.

What we tried:

- kept the earlier startup fixes in place:
  - rollout-first service startup
  - SGLang/Megatron import isolation
  - stale SGLang process cleanup
  - single-rank ROCm CPU-offload safety args
  - torch `AdamW` wrapper for Megatron CPU offload
- reverted the unpinned, non-overlap ROCm CPU-offload path back to
  `params_per_optimizer=1`
- added a focused test that loads the local Megatron optimizer file directly
  and asserts that the ROCm safety path creates one CPU sub-optimizer per
  parameter

What worked:

- focused validation remained green (`35 passed`)
- the clean 5-minute foreground gate passed with exit code `124`
- the foreground run again reached:
  - rollout and SGLang startup
  - actor initialization
  - `set_rollout_manager()`
  - weight sync
  - `Actor training step 0/200`
  - rollout generation / decode activity

What failed:

- the long tmux rerun still did not stabilize end to end
- the current `tmux-1` run produced `train_0` and then wedged
- `Rollout` is polling forever on:
  - `Current partitions: ['train_0']`
- `SGLangEngine`, `RolloutManager`, the driver, and Ray control-plane processes
  stayed alive
- the `MegatronTrainRayActor` process disappeared again

Open questions:

- did the actor still die inside `HybridDeviceOptimizer.step()` on this run,
  or did the single-parameter fallback move the failure farther downstream into
  post-step or train-batch finalization?
- does the dead worker log for this specific rerun contain a later optimizer
  boundary than the previous grouped-sub-optimizer crash?
- is the current wedge still one native actor death masked by Ray lifecycle
  behavior, or do we now have a second issue in the transfer-queue/train-batch
  completion path after the actor has already consumed part of `train_0`?

### Key Points

- The single-parameter ROCm CPU-offload fallback is safer, but it is not yet a
  complete fix.
- Passing the 5-minute gate is necessary but not sufficient on this MI210 path.
- The next debugging pass should inspect the dead worker logs for this exact
  rerun before changing more launcher or rollout behavior.

### Links

- Runtime files: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`, `amd_qwen3_4b_2gpu_e2e.sh`
- Latest log: `log/amd-qwen3-4b-2gpu-20260420_122550.log`
- Troubleshooting: `references/troubleshooting.md`

## 2026-04-20 - Keep ROCm CPU-offload params in bf16 instead of forcing fp32

**Type:** Runtime fix
**General description:** The eager-preinit experiment proved the remaining MI210
failure was persistent CPU optimizer memory, not just lazy state allocation, so
the local Megatron ROCm-safe path now keeps CPU-offloaded params in bf16
instead of forcing fp32 copies.

### Details

The latest long-run actor death no longer happened at the first CPU
sub-optimizer. With the single-parameter fallback in place, the actor survived
through CPU sub-optimizer 50 and then died at:

```text
HybridDeviceOptimizer step: cpu sub-optimizer 51 step begin
```

There was still no Python traceback from the worker, which made lazy CPU AdamW
state creation look like the next likely boundary. I tested that hypothesis by
eagerly preinitializing all AdamW state in the Relax wrapper. That changed the
failure shape, but not for the better: the actor now died during init after
logging repeated allocations like `380.00 MiB`, `190.00 MiB`, `80.00 MiB`, and
`120.00 MiB` per single-parameter CPU optimizer.

That result made the underlying problem clearer. The single-rank ROCm safe path
was still forcing offloaded CPU params into fp32, which also forced fp32 AdamW
state. I verified separately that plain CPU `torch.optim.AdamW` works with
bf16 parameters and bf16 state, so the next correct fix is to shrink the
persistent CPU optimizer memory rather than move its allocation earlier.

What changed:

- reverted the eager-preinit behavior from `TorchCPUAdamW`
- kept the wrapper on the simplest `foreach=False` path
- patched local `HybridDeviceOptimizer` so the ROCm safe path disables
  `param_update_in_fp32`
- added a direct test asserting the local ROCm-safe path keeps CPU-offloaded
  params in bf16

Expected impact:

- reduce persistent CPU optimizer memory on the ROCm offload path
- avoid both the late step-0 crash and the eager-preinit init-time crash caused
  by fp32 CPU-offloaded params plus fp32 AdamW state

### Key Points

- The remaining MI210 issue is still inside the Megatron CPU offload optimizer
  path, not Ray or rollout.
- The earlier eager-preinit patch was diagnostic, not the right steady-state
  fix.
- The new patch is intentionally narrow and only changes the single-rank ROCm
  CPU-offload path.
- The next required check is the standard focused pytest run plus a clean
  foreground 5-minute validation.

### Links

- Code: `Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`, `relax/backends/megatron/optimizer_utils.py`
- Test: `tests/utils/test_megatron_model.py`
- Troubleshooting: `references/troubleshooting.md`

## 2026-06-22 - Completion-length reward convergence depends on Qwen3 thinking mode

**Type:** Experiment retrospective
**General description:** The toy `reward = -len(completion_tokens)` task did
not converge under the default Qwen3 chat template because rollout samples
usually spent the short generation budget on reasoning/opening text; disabling
thinking mode created reward variance and produced clear improvement.

### What we tried

- Built and ran the completion-length reward environment where the optimal
  policy emits EOS immediately.
- Compared fully async, sync non-colocate, and sync colocate paths.
- Tested Qwen3-Mock-0.5B, real Qwen3-0.6B, and real Qwen3-4B.
- Moved from short flat probes to a known-good real Qwen3-4B TP2 shape:
  `ACTOR_RESOURCE_GPUS=2`, `ROLLOUT_RESOURCE_GPUS=2`,
  `TENSOR_MODEL_PARALLEL_SIZE=2`, and `ENABLE_SEQUENCE_PARALLEL=1`.
- Increased sampled batch size to `ROLLOUT_BATCH_SIZE=4`,
  `N_SAMPLES_PER_PROMPT=4`, `GLOBAL_BATCH_SIZE=16`, `MICRO_BATCH_SIZE=4`,
  and reduced weight-sync overhead with `UPDATE_WEIGHTS_INTERVAL=5`.
- Added `APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'` for the
  final Qwen3-4B TP2 convergence probe.

### Key findings

- The default-thinking Qwen3-4B TP2 b4/n4 run (`64t0zmav`) completed 30/30
  rollout/train steps but all 480 completions hit `ROLLOUT_MAX_RESPONSE_LEN=4`;
  reward stayed `-4.0`, advantages stayed zero, and actor gradients stayed zero.
- The thinking-disabled Qwen3-4B TP2 run (`fjjfcs2l`) completed 30/30 steps and
  improved `rollout/reward/mean` from `-3.0625` to `-2.0`.
- In that run, `rollout/response_len/mean` fell from `3.0625` to `2.0`,
  truncation fell from `0.125` to `0.0`, and `train/grad_norm` was nonzero on
  27/30 steps.
- The saved rollout JSONL files contained 218 length-2 samples, 228 length-3
  samples, and 34 length-4 samples; the final rollout had all 16 samples at
  length 2.
- The logged scalar mean of `rollout/advantages` can be misleadingly near zero
  because group-normalized advantages average out; reward variance plus nonzero
  `train/grad_norm` are the better indicators of a real learning signal.

### What failed

- Fully async longer runs exposed TransferQueue capacity/backpressure failures
  before they could serve as clean convergence evidence.
- Sync colocate runs hit ROCm SGLang/Megatron same-GPU handoff failures, so
  non-colocate TP2 remains the reliable convergence-debug path for now.
- Real model size alone did not fix convergence: both 0.6B and 4B stayed flat
  when the rollout distribution had no reward variance.
- Larger sampled batches and less frequent weight sync improved throughput, but
  did not solve EOS discovery under the default thinking template.
- The thinking-disabled run improved to `-2.0`, but still did not reach the
  theoretical immediate-EOS optimum `-1`; the chat-style assistant turn still
  tends to emit a short textual answer such as `OK` or `Okay` before EOS.

### Open questions

- Would a raw/non-chat prompt or a different assistant prefill make immediate
  EOS reachable and allow reward `-1`?
- Should the smoke test use a reward/stop setup that treats `<|im_end|>` as the
  expected terminal token while avoiding an instruction-like assistant answer?
- Can fully async convergence be revisited after TransferQueue capacity or
  producer/consumer backpressure is fixed?

### Links

- W&B flat default-thinking run: `https://wandb.ai/erlandpg/relax-amd-completion-length/runs/64t0zmav`
- W&B thinking-disabled run: `https://wandb.ai/erlandpg/relax-amd-completion-length/runs/fjjfcs2l`
- Launcher: `scripts/training/multimodal/amd_qwen3_4b_2gpu_e2e.sh`
- Skill: `skills/completion-length-reward-smoke-test/SKILL.md`

## 2026-06-22 - EOS-SFT mini Qwen3 did not create an EOS-learning signal

**Type:** Experiment result
**General description:** Tested the Hugging Face branch
`Erland/mini-qwen3-0.5b@Erland/EOS-SFT` on the completion-length reward task to
see whether SFT on the EOS task would make the toy RL run converge from a
better initialization.

### What we tried

- Downloaded the branch to
  `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets/Qwen3-Mock-0.5B-EOS-SFT`.
- Ran the existing Qwen3 mock 2-GPU sync non-colocate e2e path with:
  `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES_PER_PROMPT=4`,
  `GLOBAL_BATCH_SIZE=16`, `MICRO_BATCH_SIZE=4`,
  `ROLLOUT_MAX_RESPONSE_LEN=4`, `UPDATE_WEIGHTS_INTERVAL=5`,
  and `APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`.
- Logged online to W&B run `omaybwka`.

### Result

- The run completed 30/30 rollout/train steps and saved a torch_dist checkpoint
  at iteration 29.
- All 480 saved samples were `status="truncated"` with
  `response_length=4`, `reward=-4.0`, and no reward variance.
- Actor training therefore had no policy-gradient signal:
  `rollout/advantages=0.0`, `rollout/returns=0.0`, `train/loss=0.0`, and
  `train/grad_norm=0.0` on every step.
- The most common response was `system\nEmit` (365/480 samples), including the
  first and last saved rollouts.

### Interpretation

The failure was not a Relax optimizer or W&B logging failure. A direct
Transformers generation check reproduced the same behavior before RL training:

```text
chat template with enable_thinking=false -> "system\nEmit EOS immediately. Do"
assistant prefix without thinking block -> "Emit EOS immediately. Do not write"
```

So this EOS-SFT branch appears to have learned an instruction-like continuation
around "Emit EOS immediately" rather than assigning high probability to the EOS
token under the Qwen chat prompt used by Relax. Because every sampled completion
hit the response cap, GRPO had zero within-group reward variance and could not
update the policy.

### Links

- W&B run: `https://wandb.ai/erlandpg/relax-amd-completion-length/runs/omaybwka`
- Log: `log/completion-length-eos-sft-mini-30-20260622.log`
- Rollouts:
  `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets/Qwen3-Mock-0.5B-EOS-SFT_mcore_2gpu-20260622_095932/rollout_result/train/`

## 2026-06-23 - EOS-SFT mini reaches optimal reward with matched system prompt

**Type:** Experiment result
**General description:** Retried the EOS-SFT mini checkpoint with a Relax
prompt file that matches the SFT dataset format: a system instruction
`Emit EOS immediately. Do not write any text.` plus the user message, followed
by Qwen3 `enable_thinking=false` chat-template rendering.

### What changed

- Added `examples/completion_length/prompts_system_eos.jsonl` with the system
  instruction used by the SFT dataset.
- Reused the same local checkpoint:
  `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets/Qwen3-Mock-0.5B-EOS-SFT`.
- Ran the sync non-colocate mock 2-GPU e2e path with:
  `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES_PER_PROMPT=4`,
  `GLOBAL_BATCH_SIZE=16`, `MICRO_BATCH_SIZE=4`,
  `ROLLOUT_MAX_RESPONSE_LEN=4`, `UPDATE_WEIGHTS_INTERVAL=5`,
  and `APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`.

### Result

- The run completed 30/30 rollout/train steps and saved a torch_dist
  checkpoint at iteration 29.
- All 480 saved rollout samples generated exactly `<|im_end|>`.
- Reward was optimal from the first rollout through the last:
  `rollout/reward/mean=-1.0`, `rollout/response_len/mean=1.0`, and
  `rollout/truncated_ratio=0.0`.
- Because every group was already at the same optimum, GRPO correctly had no
  update signal: `rollout/zero_std/count_-1.0=4`, `train/loss=0.0`, and
  `train/grad_norm=0.0`.

### Interpretation

This confirms the previous EOS-SFT failure was prompt mismatch, not a bad SFT
checkpoint and not a Relax reward-plumbing issue. The checkpoint behaves
correctly when Relax rollout uses the same system-message structure as the SFT
dataset. This run proves the model can represent the optimal EOS policy under
the matched prompt, but it is not a learning curve because the policy starts
already optimal and has zero reward variance.

### Links

- W&B run: `https://wandb.ai/erlandpg/relax-amd-completion-length/runs/523l4j1i`
- Log: `log/completion-length-eos-sft-system-30-20260623.log`
- Prompt file: `examples/completion_length/prompts_system_eos.jsonl`
- Rollouts:
  `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron_root/relax_assets/Qwen3-Mock-0.5B-EOS-SFT_mcore_2gpu-20260623_081453/rollout_result/train/`

## 2026-07-14 - Replace the EOS learning probe with a fixed-length A/B bandit

**Type:** Retrospective and experiment design
**General description:** The matched EOS-SFT runs were separated into what
they actually prove versus what remains untested, the local reduced-model
inventory was audited, and a fixed-length two-action task was prepared as the
next minimal GRPO learning test.

### What we tried

- The original completion-length run with default Qwen3 thinking generated to
  the response cap, produced constant reward `-4`, and had zero gradients.
- Disabling thinking on a non-saturated Qwen3-4B policy produced length and
  reward variation and nonzero gradients, showing that prompt/template
  behavior—not model scale alone—controlled the useful learning signal.
- The first prompt-matched EOS-SFT became fully saturated: all 480 responses
  were immediate EOS, every reward was `-1`, every GRPO group had zero reward
  variance, and all 30 actor steps had zero loss and gradient norm.
- A tied, randomly initialized mixed EOS/`OK + EOS` SFT recovered variation.
  Its strict sampling check measured first-token probabilities of about 0.522
  for EOS and 0.432 for `OK`, with 0.954 combined mass and about 0.998
  probability of EOS after `OK`. A live one-step Relax run then observed
  response lengths 1 and 2, mean reward `-1.625`, and gradient norm `14.489`.
- The SFT workspace was audited for other reduced model families. Historical
  EOS and Reverse-Text checkpoints exist for Qwen3, GLM-4-MoE, and MiniMax-M2,
  but only Qwen3 has a validated reduced-model ROCm Relax launcher and weight
  conversion path. The corrected tied SFT recipe is also Qwen-specific.
- A new Qwen3 fixed-length A/B initializer was trained from config with no
  pretrained weights: 32 `A + EOS` targets and 32 `B + EOS` targets under the
  exact Relax prompt. The 100-step SFT completed in about 70 seconds in W&B run
  `2qypr13o` and wrote only its final model.

### Key findings

- The saturated EOS run is a successful regression test for prompt alignment,
  SGLang stop handling, reward routing, safe normalization of constant groups,
  actor orchestration, logging, and shutdown. It is not evidence of learning.
- The mixed EOS/OK run does exercise nonzero advantages and optimizer work,
  but completion length remains entangled with the action being rewarded.
- A/B is the cleaner next test because both valid trajectories have exactly
  two rollout tokens: one content token plus EOS. Any change in reward must
  therefore come from action probability, not response length.
- The accepted A/B initializer is close to the desired 50/50 policy. On 256
  samples it produced 120 A, 131 B, and 5 invalid responses: A rate `0.46875`,
  B rate `0.51172`, valid rate `0.98047`, mixed eight-sample group rate `1.0`,
  and truncation rate `0.0`. First-token A/B probability mass was `0.98226`;
  EOS probability after either action was about `0.9989`.
- Qwen3 is not the only architecture represented in the SFT directory, but it
  is the only reduced architecture currently ready for this end-to-end Relax
  experiment. Treating the GLM or MiniMax files as drop-in alternatives would
  hide missing model-integration work.

### What failed or was rejected

- Training only immediate EOS solves the synthetic objective during SFT and
  removes the within-group variance GRPO needs.
- Changing model scale without controlling Qwen thinking behavior did not fix
  the constant-reward failure.
- Reusing the current Qwen-specific tied SFT script for GLM-4-MoE would change
  its intended untied architecture and mis-handle its assistant template.
- MiniMax-M2 has no Relax model provider, converter, or launch recipe in this
  repository, so its historical SFT checkpoint cannot validate this pipeline.
- A one-token rollout cap was rejected for the bandit because it truncates
  before EOS. A cap of exactly two also makes SGLang report `length` when EOS
  lands at the cap. The final launcher allows three generated tokens but the
  reward accepts only the exact two-token action-plus-EOS trajectory.

### Open questions

- In the 30-step Relax run, does `rollout/action_a/mean` rise from roughly
  0.47 while `action_b` falls after per-step weight synchronization?
- Do mixed early groups produce nonzero gradient norms, followed by naturally
  increasing zero-variance all-A groups near convergence?
- Does the accepted SFT preserve its 98% valid-action rate through the SGLang
  and Megatron weight-conversion path?
- Cross-family coverage should be considered separately: GLM-4-MoE needs a
  reduced ROCm launcher and prompt-aware SFT recipe; MiniMax-M2 needs full
  Relax model integration before either can be compared fairly with Qwen3.

### One-step Relax integration result

The first one-step integration exposed and fixed a parser mismatch: Relax keeps
Qwen's visible `<|im_end|>` marker in `Sample.response`, so the reward must
classify `A<|im_end|>` and `B<|im_end|>` while still requiring
`response_length == 2`. After that correction, W&B run `kihpsxz1` sampled 16
valid actions with A rate `0.375`, B rate `0.625`, reward mean `0.375`, reward
range `[0, 1]`, and actor gradient norm `9.0062`. The optimizer step and
immediate actor-to-SGLang weight synchronization both completed. The run used
`SAVE_CHECKPOINTS=0`, exited successfully, and left all GPUs idle.

## 2026-07-28 - Frozen CPU vision execution plan

**Type:** Development and benchmark plan

The Qwen3-VL visual-XOR path now has an opt-in PyTorch CPU service that
computes the frozen final and DeepStack visual features once and routes the
same bundle to SGLang and Megatron. The next work is gated deliberately:

1. Propagate immutable feature/revision identities and prove CPU versus native
   GPU feature parity.
2. Prove native versus precomputed logits inside SGLang and Megatron, then
   compare synchronized rollout/training logits.
3. Complete a two-cycle shared-feature smoke before removing any visual
   weights from GPU model instances.
4. Measure native GPU vision, CPU vision with resident GPU weights, and CPU
   vision with omitted GPU weights.
5. Tune CPU threads, explicit image batches, and Ray Serve replicas only after
   correctness and VRAM gates pass.

No persistent database, Prima.cpp/llama.cpp backend, or PP-aware DeepStack
routing will be added before the PyTorch baseline identifies a measured need.
The detailed gates and transport thresholds are recorded in
`docs/draft/frozen_cpu_vision_encoder.md`.

## 2026-07-28 - Frozen CPU vision implementation checkpoint

**Type:** Development and local CPU validation

Implemented and unit-tested:

- immutable content-addressed feature IDs and frozen-vision revisions;
- final plus three DeepStack feature parity metrics and exact gates;
- a language-only Hugging Face Qwen3-VL precomputed-feature forward;
- parameterless, fail-fast omission sentinels for Megatron and SGLang;
- omission propagation through the real SGLang rollout runtime environment;
- independent Ray Serve replica count and CPU slots per replica;
- cache/backend-work service counters;
- a versioned multiprocessing CPU scaling benchmark; and
- safe launcher defaults: one replica and GPU omission disabled.

Local evidence:

- the real refinement checkpoint contains 27,084,160 visual parameters
  (51.66 MiB BF16 per full model instance);
- one real image produced a `64 x 1024` final stream and three `64 x 1024`
  DeepStack streams;
- the feature bundle occupied 524,312 bytes; and
- the one-image/two-thread warmed benchmark smoke measured approximately
  0.057 seconds of backend encode time and 0.060 seconds wall time.

The smoke used an artificial demand of 0.01 images/second solely to validate
the harness. It is not a scaling recommendation. No live GPU parity,
Ray/SGLang/Megatron two-cycle smoke, overlap result, or VRAM delta is claimed
without a supplied cluster address and dedicated idle devices.

## 2026-07-30 - Frozen CPU vision Hugging Face parity passes

**Type:** Correctness gate

The eight-image Qwen3-VL CPU-versus-native-GPU parity diagnostic passed on one
MI210 after two diagnostic compatibility fixes:

- Transformers 5.3 requires `mm_token_type_ids` in `get_rope_index`, even when
  its value is `None`.
- Requiring every BF16 feature element to satisfy `rtol=0.01`, `atol=0.01` was
  brittle across CPU and GPU kernels. The gate now requires at least `99.99%`
  elementwise agreement, cosine similarity at least `0.999`, and maximum
  absolute error at most `0.05`, while retaining the response KL, top-token,
  and A/B probability gates.

Observed results:

- all final and DeepStack cosine similarities exceeded `0.999993`;
- all streams had maximum absolute error `0.03125`;
- the lowest elementwise close fraction was approximately `0.999979`;
- mean full-vocabulary response KL was approximately `1.65e-8`;
- all eight top tokens matched; and
- maximum A/B probability delta was below `8e-7`.

Artifact:
`benchmark_results/cpu_vision/20260730_hf_parity/parity.json`.

This closes the local Hugging Face parity gate. It does not yet prove the live
Ray/SGLang/Megatron shared-feature path, GPU-weight omission, overlap, or VRAM
reduction; the two-cycle GPU-resident CPU-vision smoke is next.

Retrospective:
`training_reports/2026-07-30-qwen3-vl-cpu-gpu-vision-parity.md`.

## 2026-07-31 - Frozen CPU vision live-system retrospective

**Type:** Retrospective and three-mode systems benchmark

**General description:** The frozen Qwen3-VL CPU vision path progressed from
standalone feature/logit parity through two complete Relax cycles and real GPU
weight omission, but the final native-versus-precomputed serving comparison
exposed a live behavioral mismatch that blocks performance optimization.

### What we tried

- Proved standalone Hugging Face CPU-versus-GPU parity on eight fixed images
  for the final visual projection, all three DeepStack projections, and the
  full next-token distribution.
- Ran a fully asynchronous four-MI210 Relax smoke with one CPU vision replica,
  one Ray CPU, a 1 GiB LRU, two rollouts, two optimizer steps per rollout,
  actor DP2 on cards 0-1, SGLang on card 2, and actor-forward on card 3.
- Repeated the same two-cycle workload with GPU visual weights omitted from
  both SGLang and Megatron model instances.
- Ran a matched three-mode comparison: native frozen GPU vision, CPU vision
  with resident GPU weights, and CPU vision with omitted GPU weights. Every
  mode started from the same idle per-card VRAM baseline and saved no
  checkpoint.
- Added cumulative and interval CPU cache/backend counters to the normal Relax
  metric batches so evaluation and rollout snapshots appear in logs and W&B.

### Key findings

- The standalone representation is sound under the tested checkpoint and
  software stack: every final/DeepStack cosine exceeded `0.999993`, mean
  full-vocabulary KL was about `1.65e-8`, all top tokens matched, and maximum
  A/B probability drift was below `8e-7`.
- The live shared-feature data path is structurally complete. Evaluation,
  rollout, transfer queue, Megatron actor-forward, DP2 backward/optimizer,
  actor-to-SGLang synchronization, and final shutdown all work with final plus
  three DeepStack streams.
- GPU visual weight omission also works end to end. The omitted run completed
  768 evaluation requests, both 64-sample rollouts, four optimizer updates,
  final synchronization, and clean GPU release.
- The clean omission comparison is CPU-resident versus CPU-omitted, not native
  versus omitted. Omission reduced the simultaneous four-card peak from
  `19.090 GiB` to `18.938 GiB`, a `155.73 MiB` reduction. Summing each card's
  independently observed peak reduction gives `206.76 MiB`, but those peaks
  did not occur simultaneously.
- The evaluation LRU behaved exactly as intended with one replica: 768
  requests, 128 unique misses/encodes, 640 hits, an `83.33%` hit rate, 128
  entries, and zero evictions. Each rollout then introduced eight new images,
  so each interval correctly contained eight misses and no hits.
- CPU-resident and CPU-omitted wall times were `1006.9 s` and `1011.5 s`,
  versus `658.8 s` for native GPU vision. On the one-core correctness
  allocation, the CPU path was about `1.53x` native and weight omission itself
  produced no throughput improvement.
- CPU backend encoding consumed only about 53 seconds during baseline
  evaluation. The larger wall-time gap therefore also contains JSON feature
  serialization, Ray/Serve transport, SGLang reconstruction/consumption,
  scheduling, and generation overhead.
- The decisive unresolved result is behavioral. Native GPU vision scored
  `0.6816` on held-out visual XOR with a balanced A/B distribution. CPU
  resident and CPU omitted scored `0.5039` and `0.5000` and were biased toward
  A (`0.6875` and `0.7148`). The two CPU modes agree closely with one another,
  so omission is unlikely to be the source of the difference.

### What failed or required correction

- Reserving eight CPUs for the first vision service prevented scheduling on an
  allocation with only two physical CPU cores. The correctness launcher now
  reserves one CPU and leaves scaling to an explicitly CPU-rich allocation.
- The first precomputed SGLang request paired tokenizer-only prompt IDs with a
  processor-expanded visual grid and failed Qwen3-VL MRoPE construction. The
  precomputed path must send processor-expanded prompt IDs while raw-image and
  text-only paths keep their original behavior.
- Megatron initially received separately numbered DeepStack keys while its
  wrapper expected one ordered tuple. The wrapper now assembles numbered
  streams according to `deepstack_visual_indexes` and rejects mixed or missing
  representations explicitly.
- Local Hugging Face parity did not imply live SGLang parity. It proved that
  the CPU encoder can produce equivalent features and logits in isolation,
  but it did not cover SGLang's processor-expanded IDs, position construction,
  serialized feature reconstruction, DeepStack injection, or live policy
  version.
- The CPU-resident Ray workload succeeded, but the outer VRAM monitor recorded
  launcher status `127`. Its log contains all optimizer updates,
  `Main func successfully`, clean Ray shutdown, and Ray's final `Job ...
  succeeded`, with no matching command-not-found message. Preserve this as an
  unresolved wrapper-status anomaly rather than rewriting the artifact or
  labeling the training run failed.

### Recommended order from here

1. Freeze one checkpoint/policy version, image, prompt, processor-expanded
   token sequence, and sampling configuration.
2. Inside the live SGLang transformers backend, compare native-image and
   precomputed-feature full next-token logits, KL, top token, A/B
   probabilities, position IDs, visual grid, final stream, and all DeepStack
   stream identities.
3. Repeat the same fixed-input boundary check inside Megatron actor-forward.
4. Only after live semantic parity passes, profile binary/shared-memory
   transport, JSON overhead, CPU threads, dynamic batching, and replica
   scaling.
5. When multiple replicas are tested, aggregate per-replica counters and
   measure duplicate encodes before interpreting hit rate or capacity.

### Open questions

- At which exact SGLang boundary do native and precomputed logits first
  diverge: prompt IDs, MRoPE positions, reconstructed final features,
  DeepStack routing, dtype/device conversion, or policy synchronization?
- Does Megatron actor-forward preserve the same logits when given the exact
  bundle consumed by SGLang?
- After parity is restored, what portion of the roughly 53% slowdown is CPU
  encoding versus JSON serialization, serving transport, or SGLang feature
  ingestion?
- Can feature-ID-sticky routing avoid duplicate encodes across multiple
  independent replica LRUs, or is a shared object/binary transport required?
- Why did the successful CPU-resident outer launcher return status `127` after
  Ray reported success?

Reports:

- `training_reports/2026-07-30-qwen3-vl-cpu-gpu-vision-parity.md`
- `training_reports/2026-07-30-qwen3-vl-cpu-vision-two-cycle-smoke.md`
- `training_reports/2026-07-31-qwen3-vl-vision-three-mode-vram.md`

Artifacts:

- `benchmark_results/cpu_vision/20260730_hf_parity/parity.json`
- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/native_gpu_vram.json`
- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/cpu_resident_vram.json`
- `benchmark_results/cpu_vision/20260731_three_mode_two_cycle/cpu_omitted_vram.json`
