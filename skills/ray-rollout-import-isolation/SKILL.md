---
name: ray-rollout-import-isolation
description: >
  Keep Megatron and other heavyweight backend imports out of Ray Serve control-plane and rollout-side
  processes until the intended SGLang transformers path is ready.
  Use when: Relax rollout startup on Ray Serve emits Megatron or Transformer Engine warnings too early,
  or when a supposedly lightweight control-plane import already loads backend-specific modules.
metadata:
  short-description: "Isolate rollout-side imports on Ray Serve"
  tags:
    - ray
    - serve
    - rollout
    - imports
    - sglang
    - megatron
  domain: research
  created: 2026-04-17
  author: Codex
---

# Ray Rollout Import Isolation

## General Description

This skill captures the import-surface debugging pattern that narrowed the remaining AMD rollout failures in Relax. It is for cases where the intended runtime backend is SGLang transformers, but Ray Serve control-plane or rollout-side processes still import Megatron-backed code before the engine is actually launched.

## When to Apply

Use this knowledge when:
- `Rollout`, `RolloutManager`, or related control-plane services emit Megatron or Transformer Engine warnings during startup.
- The configured rollout backend is SGLang transformers, but the process already looks contaminated before `SGLangEngine` finishes startup.
- A plain interpreter import of a lightweight module unexpectedly imports `megatron` or `sglang`.

Do NOT use when:
- The failure is clearly inside a later training step after rollout generation and reward execution are already stable.

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Reproduced import leaks | DCS package init, `core.registry` via `advantages`, rollout-side function loading | All were observed on the MI210 path |
| Effective fixes | Lazy package exports, lazy Megatron imports, earlier rollout-side isolation | These moved the clean run to SGLang weight loading |
| Latest validated state | `SGLangEngine` reached `Load weight begin` on the transformers backend | The remaining issue is narrower than the original startup contamination |

## Recommended Practice

Treat startup contamination as a sequence of import boundaries, not as one giant runtime problem.

### Step 1: Prove the leak outside Ray first

Before touching distributed code, reproduce the contamination in a plain interpreter whenever possible. Useful probes are:

```bash
python - <<'PY'
import sys
import relax.core.registry
mods = sorted(name for name in sys.modules if name.startswith(("megatron", "sglang")))
print(len(mods))
print(mods[:20])
PY
```

If a control-plane import already loads `megatron` or `sglang`, fix that before debugging actor lifecycles.

### Step 2: Treat package `__init__` files as control-plane boundaries

If a package serves both lightweight coordinator code and heavy backend code, do not eagerly re-export backend symbols from `__init__.py`. Use lazy `__getattr__` exports so imports such as:

- `relax.distributed.checkpoint_service.coordinator.service`

do not immediately import backend-only modules like `device_direct.py`.

### Step 3: Move heavyweight imports into the exact branches that need them

If a registry or service module imports a helper that is only needed on one training path, move that import into the branch that actually uses it. On this stack, `relax.components.advantages` had to stop importing Megatron-backed helpers at file scope so `relax.core.registry` and `relax.core.controller` could stay lightweight.

### Step 4: Install rollout-side isolation before loading rollout functions

Do not rely only on `SGLangEngine` isolation. On this stack, `RolloutManager.__init__` loads the configured rollout function before any engine actor exists. Install the transformers-mode Megatron isolation at the start of:

- `Rollout.__init__`
- `RolloutManager.__init__`
- `SGLangEngine.__init__`

in that order.

### Step 5: Separate safe import-time pruning from hard import blocking

Import-time pruning of local Megatron paths and cached modules is safe. A hard `MetaPathFinder` blocker at module import time is not always safe in Ray workers because it can interfere with actor creation or deserialization. Install stronger blocking later in the actor lifecycle.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Coordinator/control-plane import already loaded Megatron | Package `__init__` eagerly re-exported backend symbols | Keep `__init__` lazy on mixed control-plane/backend packages |
| `core.registry` was heavy at import time | `advantages.py` imported Megatron helpers at file scope | Move heavyweight imports into the exact runtime branches that need them |
| `RolloutManager` still loaded Megatron-backed code before engine startup | Isolation only existed inside `SGLangEngine` | Install rollout-side isolation before loading rollout functions |
| `SGLangEngine` actor creation died immediately | Hard Megatron blocking was installed too early at module import time | Use early pruning, then later blocking inside actor/process startup |
| Clean run still emitted early rollout warnings | Another shared bootstrap/import path remained | Keep narrowing the import surface instead of jumping straight to overnight retries |

## Configuration

```yaml
control_plane_import_checks:
  - relax.distributed.checkpoint_service.coordinator.service
  - relax.core.registry
  - relax.core.controller
rollout_side_isolation_order:
  - Rollout.__init__
  - RolloutManager.__init__
  - SGLangEngine.__init__
import_time_policy:
  prune_megatron_paths: true
  block_megatron_imports: late_only
validation:
  first_step: plain_python_import_probe
  second_step: clean_5min_ray_validation
```

## References

- Related reports: `references/experiment-log.md`
- Related skills: `rocm-relax-bringup`
- Troubleshooting: `references/troubleshooting.md`
