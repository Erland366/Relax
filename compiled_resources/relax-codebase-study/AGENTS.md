# Relax Codebase Study Instructions

This directory is a study workspace for the Relax ROCm Megatron repository at:

`/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron`

Use `relax-rocm-megatron/` as the subject-codebase entry point. It is a symlink
to the active checkout, not a clone. Do not clone the repository again for this
study unless the user explicitly asks for a separate copy.

## Scope

These instructions apply only under `compiled_resources/relax-codebase-study/`.
When reading or editing the subject repository through the symlink, also follow
the repository-level instructions that govern the target files.

## Study Workflow

- Put architecture notes in `notes/architecture/`.
- Put component notes in `notes/components/`.
- Put contribution or change-planning notes in `notes/contributions/`.
- Put reusable module deep dives in `resources/modules/`.
- Track cross-cutting observations in `references/study-log.md`.
- Track reproducible failure modes in `references/troubleshooting.md`.
- Keep `resources/README.md` updated when adding module documents.

For `/deep-dive` style research, write the generated document to
`resources/modules/<topic-slug>.md` and update `resources/README.md`.

For `<advise>` and `<retrospective>` workflows, use the local skill registry in
`.codex/skills/registry.json`.

## Initial Map

| Area | Primary files |
| --- | --- |
| Entry points and launch | `relax/entrypoints/train.py`, `scripts/entrypoint/ray-job.sh`, `scripts/training/text/`, `scripts/training/multimodal/` |
| Core orchestration | `relax/core/controller.py`, `relax/core/service.py`, `relax/core/registry.py` |
| Ray Serve components | `relax/components/actor.py`, `relax/components/rollout.py`, `relax/components/advantages.py`, `relax/components/base.py` |
| Ray distributed runtime | `relax/distributed/ray/actor_group.py`, `relax/distributed/ray/train_actor.py`, `relax/distributed/ray/rollout.py`, `relax/distributed/ray/placement_group.py` |
| Megatron backend | `relax/backends/megatron/actor.py`, `relax/backends/megatron/model.py`, `relax/backends/megatron/checkpoint.py`, `relax/backends/megatron/arguments.py` |
| Megatron weight sync | `relax/backends/megatron/weight_update/`, `relax/backends/megatron/weight_conversion/`, `tests/distributed/ray/test_weight_sync.py` |
| SGLang rollout | `relax/backends/sglang/sglang_engine.py`, `relax/engine/rollout/sglang_rollout.py`, `relax/engine/router/router.py` |
| Data and rewards | `relax/engine/rollout/data_source.py`, `relax/utils/data/`, `relax/engine/rewards/` |
| Metrics and tracking | `relax/utils/metrics/service.py`, `relax/utils/metrics/metrics_service_adapter.py`, `relax/utils/metrics/adapters/wandb.py`, `relax/utils/tracking_utils.py` |
| Configuration | `relax/utils/arguments.py`, `relax/backends/megatron/arguments.py`, `relax/backends/sglang/arguments.py`, `configs/env.yaml` |
| Distributed checkpoint | `relax/distributed/checkpoint_service/`, `relax/backends/megatron/checkpoint.py`, `tests/distributed/checkpoint_service/` |
| Validation surface | `tests/`, `docs/en/guide/`, `docs/zh/guide/`, `references/troubleshooting.md` |

## Study Discipline

- Prefer code facts over guesses. Cite file paths and line numbers in notes when
  possible.
- Keep notes current with the active checkout; this study is not a historical
  snapshot.
- Avoid following the `relax-rocm-megatron/compiled_resources/relax-codebase-study`
  symlink loop during recursive scans.
- Do not put generated caches, model weights, logs, or large binary artifacts in
  this study directory.
