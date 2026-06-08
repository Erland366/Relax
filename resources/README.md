# Resources

This directory contains repo-local reference notes for understanding and debugging Relax.

## Contents

- `modules/`: Generated and maintained module notes for specific code paths and subsystems.
  - `actor-rollout-startup-debugging.md`: Startup call graph, failure surfaces, and a practical workflow for debugging actor/rollout bring-up on this codebase.
  - `relax-profiling-gap-analysis.md`: Step-by-step profiling runbook for finding framework gaps across rollout, actor training, checkpointing, weight sync, metrics, and Ray/SGLang processes.
