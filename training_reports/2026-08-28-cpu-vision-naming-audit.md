# CPU/GPU vision naming audit

- **Date:** 2026-08-28
- **Scope:** All project-owned CPU/GPU vision runtime interfaces, metrics, analyzers, launchers, tests, design docs,
  reports, presentation sources, experiment indexes, benchmark artifacts, and raw-log paths.
- **Compatibility policy:** Intentional breaking rename. No aliases, implicit translation, or old-schema fallback.
- **Runtime validation:** CPU-only tests only; no Ray, SGLang, GPU, Slurm, or training workload was launched.

## Result

The CPU/GPU vision work now uses one direct vocabulary from configuration through paper claims. Names identify the
device, cache owner, measured boundary, or actual comparison. Numbered stages and terms that implied stronger
statistics or online behavior than the implementation provides have been removed.

The canonical glossary is [docs/naming.md](../docs/naming.md). `AGENTS.md` requires future CPU/GPU vision changes to
follow it.

## Main replacements

| Earlier name | Current name |
|---|---|
| native vision | GPU vision |
| adaptive placement | automatic device choice from a versioned plan |
| route | device choice |
| calibration | timing measurements, timing model, or vision-device plan |
| hysteresis | keep the previous device within the minimum gap |
| confident decision | measured CPU/GPU gap exceeds the minimum gap |
| oracle route / d-oracle | faster device |
| regret | extra time |
| Tier 1 / Tier 2 | CPU cache / SGLang cache |
| cache prewarm / cache oracle | full-dataset preload |
| actor-cycle time | training-cycle time |
| rollout wall | rollout time |
| CPU resident / CPU omitted | CPU with GPU encoder kept / CPU with GPU encoder skipped |
| `M0`–`M3`, `D0`–`D3`, `E0`–`E4`, `B0`–`B3` | explicit setting, workload, study, and `repeat_N` names |

Statistical terms remain only when they are statistical. The reports use paired repeat-level Student-$t$ 95%
confidence intervals. The minimum gap is explicitly not called confidence.

## Public runtime and artifact schema

The direct runtime interfaces are:

```text
--vision-encoder-device={gpu,cpu}
--skip-gpu-vision-encoder
--vision-device-mode={fixed,automatic}
--vision-device-plan=PATH
--vision-device-minimum-gap=FLOAT
--preload-vision-features
--vision-encoder-max-images-per-request=INT
POST /relax/vision-features/cache
VISION_DEVICE_CHOICE
VISION_FEATURE_PRELOAD
```

The vision-device plan, full-dataset-preload result, and comparison artifacts require `schema_version: 2`. Important
schema names now include:

- `VisionTimeModel`, `time_models`, and `timing_model_fit`;
- `predicted_cpu_seconds_after_overlap`, `predicted_gap`, and `minimum_gap`;
- run names `gpu`, `cpu`, and `automatic`;
- `total_cycles`, `cycles_over_minimum_gap`, and `device_matches`;
- `match_rate` and `mean_extra_time`; and
- `training_cycle_seconds` and `rollout_seconds`.

Older experimental schemas are rejected. They are not accepted and silently normalized during a paper run.

## File and study names

Project-owned filenames now state their action or comparison. Examples include:

- `vision_device.py`;
- `build_vision_device_plan.py` and `build_vision_workload.py`;
- `compare_vision_device_choices.py` and `compare_vision_settings.py`;
- `measure_cpu_vision_demand.py`, `measure_cpu_vision_scaling.py`, `measure_vision_cache_reuse.py`, and
  `measure_full_dataset_preload.py`;
- `qwen3_vl_compare_vision_devices.sh` and `qwen3_vl_preload_all_vision_features.sh`; and
- Slurm wrappers named after demand, scaling, worker-layout, cache-reuse, or vision-setting comparisons.

Historical report filenames now use `gpu-cpu-vision`, `reuse-frozen-vision-features`, `compare-vision-settings`,
`preload-all-vision-features`, and `claim-evidence` rather than numbered or mechanism-inflating labels.

## Historical artifact migration

The migration utility is `scripts/apply_cpu_vision_naming_migration.py`. It is dry-run by default and refuses a
numeric-token change. The complete local benchmark tree contained 169 files. Every applied ledger reports:

```text
file_count = 169
all_numeric_values_preserved = true
```

The versioned ledgers are the files matching:

```text
training_reports/2026-08-28-cpu-vision-naming-migration*.json
```

They record old and new paths, old and new SHA-256 values, applied replacements, and the numeric-preservation result
for every artifact. Benchmark directories and files now use names such as:

- `20260731_gpu_cpu_vision_two_cycle`;
- `20260801_gpu_cpu_vision_performance`;
- `slurm_141944_cpu_vision_demand`;
- `slurm_142804_cpu_worker_layouts`;
- `slurm_142817_vision_cache_reuse`;
- `slurm_154079_vision_settings`; and
- `full_dataset_preload_retry2_20260827_081348`.

The audit found that the first migration implementation treated lowercase `d0`–`d3` labels as unrestricted
substrings. That could insert workload names into hexadecimal feature IDs, Ray IDs, and UUIDs while leaving numeric
measurements unchanged. The migration now applies experiment-label replacements only at token boundaries. Two
versioned repair passes restored every inserted workload name found adjacent to hexadecimal characters; the final
scan found zero such occurrences. This repair is recorded separately rather than hidden.

Thirty-seven raw log paths were also renamed from `native`, `resident`, `omitted`, and numbered `e0`–`e4` stages to
their direct GPU, CPU, GPU-encoder-skipped, demand, scaling, worker-layout, and cache-reuse names.

The repository ignore rule now keeps the renamed counterparts of the 22 previously tracked benchmark artifacts
visible to Git. The other 147 local benchmark files remain ignored, so the rename does not begin tracking unrelated
experiment output.

## Intentional exceptions

The audit does not rename established terms when they are accurate:

- Ray placement groups;
- SGLang network routing and `torch_native`;
- PyTorch native operators;
- DeepStack, `image_grid_thw`, LRU, p95, and Student-$t$ intervals;
- database-driver fields containing `oracle` as a database product name; and
- Python mocks and fabricated model implementations that are actual test doubles.

These exceptions are domain terms, not aliases for the CPU/GPU vision concepts removed above.

## Validation record

The consolidated CPU-only suite passed:

```text
329 passed, 17 warnings
```

The warnings are the existing Ray/Pydantic and SWIG deprecations. Validation also passed for:

- Python byte compilation of every modified or untracked Python file;
- `bash -n` for every modified or untracked Bash and Slurm file;
- all 40 JSON artifacts under `benchmark_results/cpu_vision`;
- all seven applied migration ledgers, each covering 169 files and reporting numeric preservation;
- the 119-column limit for every added Python line and every line in a new Python file;
- `git diff --check`;
- removal of the superseded public vocabulary outside the glossary, migration utility, migration ledgers, and this
  audit's explicit replacement table; and
- zero workload-name insertions adjacent to hexadecimal characters in the benchmark tree.

Ruff is not installed in the existing `relaxrl_rocm_after_fix` environment, so it was not run and no dependency was
installed merely for this audit. Byte compilation also displayed the existing `relax/utils/arguments.py:413`
`SyntaxWarning` for an invalid escape in an unrelated regex help example; compilation still passed and this audit did
not broaden scope to change that example.

No GPU integration run is needed for a terminology-only schema migration, but the next live comparison must use only
the version-2 interfaces and artifacts described here.
