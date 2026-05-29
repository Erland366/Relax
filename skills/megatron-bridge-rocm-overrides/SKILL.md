---
name: megatron-bridge-rocm-overrides
description: >
  Capture the ROCm-specific Megatron-Bridge override pitfalls that caused disabled fusions
  to remain active in practice.
  Use when: a Relax launcher flag appears correct, but Megatron behavior on ROCm still follows the fused CUDA path
  or a TE-less/offload path still assumes optional NVIDIA dependencies are present.
metadata:
  short-description: "Megatron-Bridge override checks for ROCm"
  tags:
    - rocm
    - megatron
    - bridge
    - overrides
    - fusion
  domain: research
  created: 2026-04-15
  author: Codex
---

# Megatron Bridge Rocm Overrides

## General Description

This skill captures a specific failure mode where the Relax launcher passed AMD-safe flags, but Megatron still entered CUDA-only fused paths because the provider override layer did not propagate every setting. It is useful when the parsed CLI looks reasonable while runtime behavior still contradicts the intended ROCm configuration.

## When to Apply

Use this knowledge when:
- A launcher flag such as `--no-masked-softmax-fusion` is present, but runtime still imports CUDA-only fused kernels.
- The run uses Relax with Megatron-Bridge on ROCm.
- You need to verify whether the effective provider configuration matches the user-facing launcher flags.

Do NOT use when:
- The failure is clearly inside non-compiled pure Python logic with no provider override involvement.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Relevant flag | `masked_softmax_fusion=False` | Observed in parsed config |
| Actual failure before fix | `scaled_masked_softmax_cuda` import | Runtime still entered fused path |
| Related AMD-safe layout | `qkv_format=bshd` | Avoided packed-sequence `DotProductAttention` failure |
| Outcome after override and fallback fixes | Failure boundary moved forward | Later failures moved into optimizer/offload and compiler/runtime layers |

## Recommended Practice

Treat launcher flags and provider overrides as separate contracts. On ROCm, any CUDA-only fusion flag that is disabled at the CLI must also be verified in the effective Megatron provider configuration. If a disabled feature still triggers a CUDA import, patch the propagation path before assuming the launcher is wrong.

### Step 1: Verify the parsed config and the provider path

Confirm the parsed arguments reflect the intended disabled fusion. Then inspect the Relax provider bridge path to ensure that key is forwarded into Megatron configuration construction.

### Step 2: Add explicit propagation for ROCm-sensitive settings

Make sure ROCm-sensitive flags such as `masked_softmax_fusion`, attention layout choices, and similar fused-path toggles are included in the provider override key set. Do not assume adjacent attention settings imply the same behavior.

### Step 3: Keep upstream fused paths failure-tolerant

If upstream Megatron attempts to import NVIDIA-only extensions, missing imports should mean "kernel unavailable" rather than a hard crash. That fallback behavior is the only sensible default on AMD bring-up.

### Step 4: Treat TE absence as part of the override surface

If a ROCm path enables features such as CPU optimizer offload and those code paths consult Transformer Engine capability checks, ensure the capability probe returns `False` when TE is absent. Optional-dependency version checks should not crash actor initialization.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| CLI disabled masked softmax fusion, but runtime still imported CUDA extension | Provider override path dropped `masked_softmax_fusion` | Check propagation all the way into Megatron, not just the launcher script |
| Upstream fused softmax path crashed instead of falling back | Missing CUDA extension was treated as fatal | ROCm execution needs import-safe kernel availability checks |
| ROCm path still entered unsupported packed-sequence attention | The launcher/provider combination still allowed a layout that reached plain `DotProductAttention` | Keep the AMD path explicit with `bshd` layout when TE-backed attention is unavailable |
| CPU offload path crashed on missing TE version | Capability probe assumed TE existed and returned a real version object | Optional TE checks must degrade to `False` on ROCm without TE |

## Configuration

```yaml
launcher_flags:
  masked_softmax_fusion: false
  qkv_format: bshd
provider_checks:
  - masked_softmax_fusion
  - qkv_format
  - local_spec_provider_when_te_missing
fallback_policy:
  missing_cuda_extension: treat_as_unavailable
  missing_transformer_engine: treat_as_unsupported
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`
- Troubleshooting: `references/troubleshooting.md`
