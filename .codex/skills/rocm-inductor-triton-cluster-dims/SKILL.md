---
name: rocm-inductor-triton-cluster-dims
description: >
  Diagnose the ROCm TorchInductor/Triton failure where launcher generation expects
  `KernelMetadata.cluster_dims` but the compiled kernel metadata does not provide it.
  Use when: Relax reaches real training execution on ROCm and then dies during TorchInductor/Triton compilation,
  after earlier HIP-sensitive helper paths have already been reduced or disabled.
metadata:
  short-description: "TorchInductor ROCm cluster_dims failure pattern"
  tags:
    - rocm
    - torchinductor
    - triton
    - compiler
    - megatron
  domain: research
  created: 2026-04-15
  author: Codex
---

# Rocm Inductor Triton Cluster Dims

## General Description

This skill records the later-stage ROCm failure that remained after the Relax MI210 bring-up work had already succeeded through rollout generation and reward execution. It is specifically about the TorchInductor/Triton compiler boundary, where generated launcher code expected `cluster_dims` in kernel metadata and the ROCm-side object did not provide it.

## When to Apply

Use this knowledge when:
- The run already reaches the first actor training step on ROCm.
- The failure is raised as `torch._inductor.exc.InductorError`.
- The stack trace ends in `torch/_inductor/runtime/triton_heuristics.py` with `KernelMetadata` missing `cluster_dims`.

Do NOT use when:
- The run still fails in SGLang import, model loading, or earlier Megatron configuration stages.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Failure stage | First actor training step | Rollout and reward execution had already succeeded |
| Typical time to failure | ~345s after launch | Seen repeatedly in the overnight loop |
| Error signature | `KernelMetadata.cluster_dims` missing | Raised inside TorchInductor/Triton launcher generation |
| Earlier mitigations before this stage | HIP-safe eager fallback for small PPO helpers | Separate from the later `cluster_dims` compiler/runtime issue |

## Recommended Practice

Classify this as a compiler/runtime compatibility bug first. Do not keep retesting higher-level Relax launcher changes once the run has already crossed into actor training and the stack trace points into TorchInductor launcher generation. Preserve the exact PyTorch, Triton, and ROCm versions and bisect from the compiled path outward.

### Step 1: Freeze the software versions that produced the failure

Record the exact environment versions before changing anything else:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton import failed", exc)
PY
```

### Step 2: Reproduce with a short foreground validation run

Run the launcher in the foreground long enough to confirm that the failure still happens at the first actor training step. Do not hide this class of failure inside an overnight retry loop until the compiler path is narrowed.

### Step 3: Narrow the compiled path

Reduce or disable the specific compiled Megatron path that triggers TorchInductor launcher generation, then validate again. If the error disappears, the next action is version or compiler-path adjustment, not more Ray or SGLang changes.

### Step 4: Remove small helper noise before blaming the later compiler failure

Some HIP failures in this stack came from tiny `torch.compile(dynamic=True)` PPO helpers rather than the main later-stage Megatron path. Convert those helpers to eager on HIP first. If the run then moves forward and a later `cluster_dims` crash remains, you have successfully separated a helper-level ROCm issue from the main compiler/runtime incompatibility.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Overnight retries kept labeling the issue as startup failure | The failure happened at a similar wall-clock time, but at a much later execution stage | Use the stack trace and execution boundary, not elapsed time, to classify the problem |
| Further launcher-level tuning after rollout success | The real issue was inside TorchInductor/Triton | Once rollout and reward execution work, shift effort to compiler/runtime compatibility |
| Small HIP helper compile bugs got mixed into the same bucket | The run contained more than one ROCm compilation problem | Strip out helper-level `torch.compile` issues before concluding the remaining failure is the main `cluster_dims` bug |

## Configuration

```yaml
debug_protocol:
  classify_as: compiler_runtime
  required_versions:
    - torch
    - triton
    - rocm
  precheck:
    - disable_or_avoid_small_hip_sensitive_compiled_helpers
  validation_mode: foreground
  retry_loop: disabled_until_reproduced_cleanly
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`
- Troubleshooting: `references/troubleshooting.md`
