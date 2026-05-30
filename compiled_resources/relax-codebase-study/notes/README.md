# Study Notes

Use this directory for durable notes about the Relax ROCm Megatron codebase.

## Progress Tracker

| Area | Status | Notes |
| --- | --- | --- |
| Entry points and launch scripts | Not started | Start from `relax/entrypoints/train.py` and `scripts/entrypoint/ray-job.sh`. |
| Core orchestration and lifecycle | Not started | Focus on `relax/core/controller.py` and service startup. |
| Ray Serve components | Not started | Map component deployments and request flow. |
| Ray distributed runtime | Not started | Map actor groups, placement groups, and train actor calls. |
| Megatron backend | Not started | Separate upstream Megatron assumptions from ROCm-specific behavior. |
| SGLang rollout backend | Not started | Trace rollout request routing and engine initialization. |
| Data, rewards, and advantages | Not started | Map batch shape contracts and reward outputs. |
| Metrics and tracking | Not started | Include W&B step semantics and service metrics. |
| Distributed checkpoint and weight sync | Not started | Trace checkpoint service and Megatron weight update paths. |
| Tests and docs | Not started | Link existing tests to the modules they protect. |
