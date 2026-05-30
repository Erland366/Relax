<!-- LAST_INDEXED: 2026-05-29 -->

# Relax Codebase Study

This directory is a local codebase-study scaffold for the current Relax ROCm
Megatron checkout.

## Subject Codebase

- Repository path:
  `/vast/users/qirong.ho/erland/Python_project/Relax-rocm-megatron`
- Subject symlink: `relax-rocm-megatron -> ../..`
- Origin: `https://github.com/Erland366/Relax.git`
- Branch at initialization: `dev`
- Commit at initialization: `a0eecd0`

This study intentionally references the active checkout instead of cloning it.
Treat source files under `relax-rocm-megatron/` as live repository state, not a
frozen snapshot.

## Contents

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Local instructions for studying this codebase. |
| `relax-rocm-megatron/` | Symlink to the repository root being studied. |
| `notes/` | Human-readable architecture, component, and contribution notes. |
| `resources/modules/` | Deep-dive module documents generated from code study. |
| `references/` | Study log, troubleshooting notes, and cross-cutting findings. |
| `training_reports/` | Experiment or validation reports relevant to the study. |
| `.codex/skills/` | Local study skill wiring for advise and retrospective flows. |

## Initial Study Areas

1. Entry points and launch scripts.
2. Core orchestration and service lifecycle.
3. Ray Serve components and Ray actor management.
4. Megatron training backend, ROCm adaptations, and weight conversion.
5. SGLang rollout backend and request routing.
6. Metrics, tracking, and observability.
7. Configuration and argument parsing.
8. Distributed checkpoint and weight synchronization.
9. Existing tests, docs, and validation gaps.

## Freshness

Update this README and `../README.md` when adding new resources or changing the
study structure. The `LAST_INDEXED` marker should reflect the latest manual
index update date.
