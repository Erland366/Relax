---
name: model-integration
description: Guide for integrating a model into Relax. Use when adding an architecture, writing Megatron-to-HF converters, implementing TP gather/chunk logic, debugging weight sync, adapting colocate or fully-async execution, routing precomputed multimodal features, or profiling their cache and transport path. Covers Megatron bridge/raw modes, FSDP, and live multimodal parity gates.
---

# Model Integration Guide

## Overview

Relax 支持两种训练后端和两种执行模式，新模型接入需根据组合适配不同的模块。

### 后端 x 模式矩阵

|              | Colocate (共卡)                             | Fully-Async (全异步)                            |
|--------------|---------------------------------------------|-------------------------------------------------|
| **Megatron** | Bridge 自动转换；Actor/Rollout 时分共享 GPU | Bridge 或 raw 转换；独立 GPU 集群，DCS 权重同步 |
| **FSDP**     | HF 原生权重名，DTensor 自动 redistribute    | 暂不支持                                        |

### 决策树

```
新模型接入
  ├─ Megatron 后端？
  │    ├─ Bridge 支持？ → [快速路径] 仅需 Step 1-2
  │    └─ Bridge 不支持 → [完整路径] Step 1-5
  └─ FSDP 后端？ → [FSDP 路径] Step 6
```

## Megatron Backend

### 权重同步 Pipeline

训练完成后，权重需从 Megatron 内部格式同步到 Rollout 引擎（SGLang）：

```
Training Step Complete
        │
        ├─── [bridge 模式] ─────────────────────┐
        │    AutoBridge.export_hf_weights()     │
        │    自动处理 Megatron→HF 转换          │
        │                                       ▼
        ├─── [raw 模式] ────────────────────────┐
        │    all_gather_param (common.py)       │  TP-sharded → full
        │            │                          │
        │            ▼                          │
        │    convert_to_hf (__init__.py)        │  Megatron name → HF name
        │            │                          │
        │            ▼                          │
        ├────────────┴──────────────────────────┘
        │
        ├─── [共卡] UpdateWeightFromTensor
        │    GPU→CPU serialize → Gloo gather → Ray IPC
        │
        └─── [全异步] UpdateWeightFromDistributed / DeviceDirectBackend
             NCCL broadcast / HTTP push
                     │
                     ▼
             chunk_param (checkpoint_engine/utils.py)
             full → TP-sharded (逆操作，仅全异步)
```

**共卡 vs 全异步选择逻辑**（`actor.py:176`）：
```python
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
```

### Step 1: Model Provider (Bridge 模式 — 快速路径)

**前提**：模型已有 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 支持。

Bridge 模式下，`model_provider.py` 自动完成：
- **模型构建**：`AutoBridge.from_hf_pretrained()` → `bridge.to_megatron_provider()`
- **HF→Megatron 加载**：`checkpoint.py` 中 `bridge.load_hf_weights(model)` 自动转换权重
- **Megatron→HF 导出**：`hf_weight_iterator_bridge.py` 中 `bridge.export_hf_weights()` 自动导出

**你需要做的**：
1. 确认 `AutoBridge.from_hf_pretrained(hf_checkpoint)` 能正确加载你的模型
2. 如果模型有非标准 TP 分片（参见 Step 5），仍需添加自定义 all_gather/chunk 逻辑

### Step 2: Model Config & Launch Script

**File**: `scripts/models/<model_name>.sh` — 模型架构参数

从 HF config 提取对应的 Megatron 命令行参数。具体样例见 `references/examples.md#model-config-script`。

**File**: `scripts/training/<category>/run-<model>-<size>-<gpus>gpu[-async].sh` — 训练启动脚本

关键区分参数：
- 共卡：无特殊 flag（默认即共卡）
- 全异步：`--fully-async --resource '...' --max-staleness N`
- 转换模式：`--megatron-to-hf-mode bridge`（推荐）或 `raw`

具体样例见 `references/examples.md#launch-script-colocate` 和 `references/examples.md#launch-script-fully-async`。

> **Bridge 模式到此结束**。以下 Step 3-4 仅在 `--megatron-to-hf-mode raw` 时需要。

### Step 3: Megatron-to-HF Converter (raw 模式)

**File**: `slime/backends/megatron_utils/megatron_to_hf/<model_name>.py`

转换器是纯函数，将单个 Megatron 参数映射为一或多个 HF 参数：

```python
def convert_<model>_to_hf(
    args, name: str, param: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
```

**必须覆盖的参数映射**：

| Megatron Name Pattern | HF Target | Notes |
|---|---|---|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` | |
| `output_layer.weight` | `lm_head.weight` | |
| `decoder.final_layernorm.weight` | `model.norm.weight` | |
| `self_attention.linear_qkv.weight` | `q/k/v_proj.weight` | 按 GQA 拆分 |
| `self_attention.linear_proj.weight` | `o_proj.weight` | |
| `mlp.linear_fc1.weight` | `gate_proj` + `up_proj` | GLU: chunk(2) |
| `mlp.linear_fc2.weight` | `down_proj.weight` | |
| `*.layer_norm_weight` | `*_layernorm.weight` | fused norm |

MoE、多模态、MTP 等扩展映射见 `references/examples.md`。

**参考**：简单模型看 `qwen2.py`(71行)，复杂模型看 `qwen3_5.py`(345行)。

### Step 4: Register Converter (raw 模式)

**File**: `slime/backends/megatron_utils/megatron_to_hf/__init__.py`

```python
# 1. 顶部添加 import
from .<model_name> import convert_<model>_to_hf

# 2. 在 _convert_to_hf_core() 添加路由（具体名字在前）
if "<model_name>" in model_name:
    converted_named_tensors = convert_<model>_to_hf(args, name, param)
```

`model_name` 来自 `type(AutoConfig.from_pretrained(hf_checkpoint)).__name__.lower()`。注意路由顺序：具体名字（如 `qwen3_5`）必须在通用名字（如 `qwen3`）之前。

### Step 5: Custom TP Sharding (bridge 和 raw 均适用)

大多数标准层的 TP 分片由框架通用逻辑处理。**仅当模型有非标准 TP 分片方式时**才需要此步骤。

需要同时修改两个互为严格逆操作的函数：

| 函数 | 文件 | 方向 | 场景 |
|---|---|---|---|
| `all_gather_param` | `slime/.../update_weight/common.py` | TP-sharded → full | 训练侧发送权重 |
| `chunk_param` | `relax/checkpoint_engine/utils.py` | full → TP-sharded | Rollout 侧加载权重(全异步) |

**核心不变量**：`chunk_param(all_gather_param(p))[tp_rank] == p`

在通用逻辑前按参数名 pattern 插入模型特有分支。具体代码见 `references/examples.md#custom-tp-all-gather`。

## FSDP Backend

### Step 6: FSDP Model Adaptation

FSDP 后端直接使用 HuggingFace 模型，权重名就是 HF 原生名称，**不需要写转换器**。

**权重同步**（`slime/backends/fsdp_utils/update_weight_utils.py`）：
- `model.state_dict()` 遍历所有参数
- DTensor 通过 `redistribute(Replicate)` 自动 all-gather
- 按 buffer 大小分桶，序列化推送到 Rollout

这套逻辑对所有模型通用，无需模型特化。

**需要模型特化的场景**：仅 MoE 模型需要自定义 expert dispatch 和 fused kernel（如 `slime/backends/fsdp_utils/models/qwen3_moe.py`），放在 `slime/backends/fsdp_utils/models/` 下。

## Common Pitfalls

### 1. module. prefix 不一致
不同模型 prefix 深度可能不同。如 Qwen3.5 多了 `language_model` 层，需要在转换器开头归一化：
```python
if name.startswith("module.module.language_model."):
    name = "module.module." + name[len("module.module.language_model."):]
```

### 2. 多模态子模型绕过 TP
非 Megatron 的视觉/音频编码器没有 `tensor_model_parallel` 属性，需在 `all_gather_param` 中跳过：
```python
if ".vision_model." in name:
    if not hasattr(param, "tensor_model_parallel"):
        return param
```

### 3. ZeroCenteredRMSNorm 偏移
Megatron 的 ZeroCenteredRMSNorm 权重中心为 0，HF 的 RMSNorm 中心为 1：
```python
return [(f"{prefix}.norm.weight", param + 1)]
```

### 4. GLU linear_fc1 layout
框架已内置 SwiGLU 的交错 layout 处理，不需要额外代码。

### 5. Bridge 模式仍需自定义 TP 逻辑
Bridge 自动处理名称转换，但 `all_gather_param`/`chunk_param` 中的自定义 TP 逻辑仍然需要——TP 分片/还原发生在转换之前/之后。

### 6. _convert_to_hf_core 路由顺序
`in` 匹配 model_name 时，具体名字必须在前：`qwen3_5` → `qwen3vl` → `qwen3`。

### 7. HF Config 动态读取
模型特有参数通过 `get_hf_config()` 获取（`@lru_cache`，只加载一次）：
```python
from slime.utils.misc import get_hf_config
config = get_hf_config(args.hf_checkpoint)  # .text_config.xxx
```

### 8. 共卡模式下 named_params_and_buffers 的 convert_to_global_name
共卡模式使用 `UpdateWeightFromTensor`，内部通过 `HfWeightIteratorBase.create()` 根据 `megatron_to_hf_mode` 选择迭代器。raw 模式需要 `convert_to_global_name=True`（将 VP/PP 局部层索引转换为全局索引），bridge 模式不需要：
```python
# actor.py:154
convert_to_global_name=args.megatron_to_hf_mode == "raw"
```

## Precomputed Multimodal Feature Parity

When offloading a frozen vision/audio encoder or injecting precomputed
features, validate the following gates in order. Stop at the first failure;
do not use a later end-to-end success to waive an earlier semantic mismatch.

For every live consumer, distinguish these states explicitly:

```text
payload produced -> transported -> parsed -> injected into model -> logits match
```

The first three states are structural evidence only. Cache hits, valid feature
IDs, successful request parsing, GPU-weight omission, or a completed RL cycle
do not prove that the consumer passed the supplied representation into its
language-model forward.

```text
structural contract
  -> standalone representation parity
  -> standalone language-logit parity
  -> live rollout-backend parity
  -> live training-backend parity
  -> end-to-end behavioral parity
  -> resident-versus-omitted VRAM
  -> throughput and scaling
```

1. **Structural contract:** compare feature revision/ID, stream names and
   order, shapes, grid metadata, dtype, and expected language-layer injection
   indexes. For Qwen3-VL, check the final projected stream and every projected
   DeepStack stream independently.
2. **Standalone representation parity:** compare native and precomputed
   features with cosine, absolute-error distribution, finite-value checks, and
   elementwise-close fraction. Do not require bitwise identity across CPU and
   GPU BF16 kernels.
3. **Standalone language-logit parity:** hold prompt, processor output, model
   revision, and tokens fixed; compare full next-token logits/KL, top tokens,
   and task-action probabilities.
4. **Live rollout parity:** repeat the fixed-input comparison inside SGLang or
   the selected rollout backend. Record processor-expanded prompt IDs,
   position IDs, grid metadata, reconstructed feature dtype/device, all
   multimodal streams, and policy version. Verify the actual language-model
   call consumes those tensors; observing them in a parsed request object is
   not sufficient.
5. **Live training parity:** feed the exact same immutable bundle and token
   sequence to actor-forward/training and compare logits before relying on
   importance sampling or policy synchronization diagnostics.
6. **End-to-end behavior:** compare held-out task accuracy, action
   distribution, validity, and controls. A completed rollout/optimizer smoke
   proves plumbing, not semantic parity.
7. **Memory isolation:** compare the same precomputed-feature path first with
   GPU encoder weights resident and then omitted. Do not attribute the full
   native-versus-offloaded VRAM difference to omitted parameters.
8. **Performance:** tune transport, CPU threads, batching, routing, and replica
   count only after every semantic gate passes. Track total requests separately
   from backend encode work, and aggregate counters across replicas.

### Performance gate for cached precomputed features

Treat each downstream reuse boundary separately. A CPU encoder cache hit proves
only that encoder execution was avoided; it does not prove reuse of
serialization, transport, reconstruction, H2D, or prefill.

Before scaling CPU threads or replicas, record:

- unique feature count, service requests, backend encodes, hits, and evictions;
- raw unique-feature bytes and serialized/on-wire request bytes;
- conversion, request-build, service-round-trip, and response-wait timing; and
- generation requests per unique feature, including `n_samples_per_prompt`.

Compute JSON expansion (`mean request bytes / mean raw feature bytes`) and
effective amplification (`total request bytes / unique raw feature bytes`). If
request construction or repeated transport dominates backend encode work, fix
transport first.

Before adding a registry, test whether the rollout engine can branch multiple
samples from one multimodal request. For SGLang, `sampling_params.n` can remove
within-prompt retransmission while preserving one parsed precomputed bundle.
Gate it on exact ordered response cardinality, scalar-equivalent log-probs,
request bytes, and the framework's seed semantics. Keep deterministic mode on
scalar requests unless the engine proves distinct per-branch seeds; also keep
partial/resumed/custom generations scalar until their abort and continuation
contracts are proven. With multiple engines, keep non-round-robin routing
scalar so grouping does not erase per-sample routing decisions; any routing
policy is safe when exactly one engine exists. Count grouped HTTP timing and
bytes once, not once per returned sample.

Before adding mutable server state, benchmark a compact lossless inline
representation. For BF16 features, send contiguous bytes plus explicit dtype
and shape in the existing request envelope; base64 is acceptable as the
stateless first gate. Require an exact BF16 bit round trip and repeat the live
native-versus-precomputed logit gate. Decode before the rollout backend's base
multimodal processor reads `feature`, remove transport-only fields afterward,
and guard the decoder so native PIL/media objects retain their established
path. Preserve a legacy reader only when migration compatibility is explicit.

If repeated transport across distinct requests remains material afterward,
prefer one explicit bounded engine-local cache keyed by feature schema,
`vision_revision`, and `feature_id`: publish each immutable bundle inline
once, then use ID-only generation requests. Fail on missing IDs, revision,
schema, grid, or shape mismatches and wrong-engine routing; never silently
recompute raw vision. Use the same feature-sticky routing key for publication,
lookup, and recovery. A marked miss may republish inline exactly once; other
HTTP errors and a second failure must propagate. Instrument publication,
ID-only hit/miss, republish, request bytes, reconstruction/H2D, prefill, and
eviction before comparing throughput again.

The Qwen3-VL implementation exposes this cache through
`SGLANG_VISION_FEATURE_CACHE_MAX_BYTES`, disabled by default. It is
process-local, job-lifetime host RAM rather than a persistent registry. The
grouped hit path delays request serialization so ID-only hits do not rebuild
BF16/base64 payloads. Enabled multi-engine operation requires consistent
hashing, and the actor's tensor-only batch contract remains unchanged.

For fully asynchronous VRAM comparisons, retain per-role device peaks and
phase context. One simultaneous cluster maximum is schedule-sensitive; isolate
omission with repeated runs or synchronized phase-specific probes.

The corrected Qwen3-VL example exposed the pattern: a 524,312-byte feature
became a 2.907 MB JSON request, 768 evaluation requests sent 2.233 GB, and the
CPU backend spent only 49-54 seconds encoding 128 unique images. See the
2026-08-01 performance report; do not treat these model-specific values as
universal thresholds.

The 2026-08-04 follow-up grouped stochastic evaluation independently of reward
grouping and replaced nested numeric JSON with inline BF16 base64. Under the
same omitted-weight workload, evaluation fell from 1.117 GB grouped nested
traffic to 268.9 MB packed traffic, and each 64-sample rollout sent 5.60 MB.
Live native-versus-packed tokens matched with maximum A/B log-probability delta
`1.43e-6`; all four optimizer updates passed. See the packed-transport report
before proposing a feature registry.

Qwen3-VL precomputed requests must use processor-expanded prompt IDs that
match the visual grid. Preserve numbered DeepStack streams in index order or
assemble them into the exact ordered container expected by the language-model
wrapper; reject mixed or missing representations instead of falling back.

SGLang's generic transformers wrapper historically parsed
`item.precomputed_embeddings` while its ordinary multimodal forward gathered
only raw `item.feature` values. The corrected Qwen3-VL path uses an explicit
all-precomputed prefill adapter: it validates and splits the packed final plus
DeepStack streams, scatters the final stream into image-token positions, and
passes the visual-position mask and ordered DeepStack streams to the language
model. Native-image requests and decode keep their original path, while mixed
native/precomputed prefill fails explicitly.

### CPU offload capacity protocol

Do not choose CPU threads, replicas, dynamic batching, sticky routing, or a
feature registry from isolated encoder throughput. Apply these gates in order:

```text
representation parity
  -> language-logit parity
  -> live consumer demand
  -> offline CPU supply
  -> live topology validation
  -> cache-locality decision
```

Measure consumer demand before supply. For grouped RL workloads, keep total
generated responses constant while increasing unique images so that actor
sample volume does not confound CPU demand. Compute peak demand from steady
cycles as unique feature IDs consumed divided by the corresponding actor-cycle
wall time. A performance-only `n=1` saturation probe must not be described as
a meaningful GRPO learning configuration.

Validate CPU topology before Ray startup. Count scheduler-affinity CPUs and
distinct `(physical_package_id, core_id)` pairs; SMT siblings are not distinct
physical cores. Reserve an explicit CPU budget for ViT and leave headroom for
Ray, Serve, processor work, and control actors. The Qwen3-VL scaling protocol
requires 16 affinity CPUs, at least eight physical cores, and at most eight
reserved ViT CPUs.

For multi-replica telemetry, do not call a load-balanced `get_metrics()` and
present one arbitrary replica as deployment state. Each encode response should
carry replica ID, feature ID, cache status, backend wall time, actual backend
batch size, and a cumulative replica-local snapshot. Retain the newest
snapshot per replica, derive intervals from prior per-replica snapshots, and
track response-level feature IDs independently so stale out-of-order responses
cannot corrupt cumulative counters.

In the offline sweep, compare images per second, backend process CPU seconds,
wall time, scaling efficiency, peak RSS, replica layout, and explicit batch
size. Choose the smallest CPU reservation sustaining at least `1.25` times
measured peak consumer demand; break ties by throughput, fewer replicas, fewer
threads, then smaller batch size.

An explicit backend batch does not prove live request coalescing. Make dynamic
batching eligible only if a batch size above one improves throughput by at
least 20% over batch size one at the same replica/thread layout. Select the
smallest batch within 5% of the best batched throughput. Until that artifact
exists, keep the live batch-wait timeout at zero and reject positive values
rather than silently ignoring them.

Validate candidate layouts live with exact image and response accounting,
per-replica load distribution, process CPU utilization, peak RSS, service RTT
mean/p95, actor data-wait ratio, staleness backpressure, rollout/actor overlap,
optimizer completion, final synchronization, idle-memory recovery, and clean
shutdown. Require actor wait below 5% and capacity of at least `1.25` times
consumer demand. Keep weight-update time visible as a covariate, not as the
CPU-ViT optimization objective.

After selecting a live topology, restore the normal cache and compute
aggregate backend encodes divided by unique feature IDs. A ratio above `1.25`
justifies testing feature-sticky routing next; it does not justify jumping
directly to a database or shared feature registry.

The 2026-08-06 Qwen3-VL run explains why the ordering matters: one uncached
one-thread CPU replica kept actor wait at `0.88%` with full rollout/actor
overlap even though packed requests were `383.5` times larger than native
requests. The August 11 allocation then exposed only two SMT siblings of one
physical core, so the scaling matrix correctly remained unrun. The later E1
allocation measured a global peak demand of `5.8588356399` images/s and a
`7.3235445499` images/s capacity target; D2 actor wait still remained below
1%. Read these reports before claiming a CPU layout, cache speedup, or batching
result:

- `training_reports/2026-08-06-qwen3-vl-cpu-vision-steady-state-overlap.md`
- `training_reports/2026-08-11-qwen3-vl-cpu-vit-scaling-readiness.md`
- `training_reports/2026-08-18-qwen3-vl-policy-invariant-representation-cache.md`

### Qwen3-VL live regression probe

Run the fixed-image native-versus-precomputed comparison inside one real
SGLang transformers server before benchmarking the CPU path:

```bash
python -m examples.visual_xor.validate_sglang_cpu_vision_parity \
  --checkpoint "$VISUAL_REFINEMENT_SFT" \
  --dataset "$VISUAL_REFINEMENT_DATA/refinement_rl_eval.parquet" \
  --output benchmark_results/cpu_vision/sglang_live_parity.json \
  --num-images 8 \
  --host 127.0.0.1 \
  --port 31000 \
  --base-gpu-id 0
```

Use the same SGLang Python environment and ROCm `sgl_kernel` paths as the
Relax launcher. The validated 2026-07-31 run matched generated tokens and
decoded text on all eight images. Its maximum A/B log-probability delta was
`1.430511474609375e-06`, maximum action-margin delta was
`1.043081283569336e-07`, and maximum normalized action-probability delta was
`1.7429432036530912e-08`.

The follow-up omitted-weight Relax run kept rollout-versus-Megatron
sampled-token log-probability mean absolute differences below `5e-7` at both
optimizer updates. This is a live actual-sample cross-backend gate, not a
within-Megatron native-versus-precomputed full-vocabulary comparison.

Evidence and measured examples:

- `training_reports/2026-07-30-qwen3-vl-cpu-gpu-vision-parity.md`
- `training_reports/2026-07-30-qwen3-vl-cpu-vision-two-cycle-smoke.md`
- `training_reports/2026-07-31-qwen3-vl-vision-three-mode-vram.md`
- `training_reports/2026-07-31-qwen3-vl-live-precomputed-deepstack-parity.md`
- `training_reports/2026-08-01-qwen3-vl-corrected-three-mode-performance.md`
- `training_reports/2026-08-03-qwen3-vl-sglang-parallel-sampling.md`
- `training_reports/2026-08-04-qwen3-vl-packed-cpu-vision-transport.md`

## Validation Checklist

1. **转换器覆盖率** (raw 模式)：遍历所有参数名，确认无 `ValueError`
2. **all_gather/chunk 对称性**：构造随机 tensor → 分片 → all_gather → chunk → 验证一致
3. **端到端**：小规模跑 1-2 iter，检查 log 无 `Unknown parameter name`，Rollout 引擎成功加载权重
4. **多模态预计算路径**：按上述 parity ladder 逐级验证，不以端到端 smoke 代替固定输入 live-logit parity

## File Reference Map

| File | Role | When to Modify |
|---|---|---|
| `slime/backends/megatron_utils/megatron_to_hf/<model>.py` | Megatron→HF 映射 (raw) | raw 模式 |
| `slime/backends/megatron_utils/megatron_to_hf/__init__.py` | model_name 路由 (raw) | raw 模式 |
| `slime/backends/megatron_utils/update_weight/common.py` | TP all_gather | 非标准 TP 时 |
| `relax/checkpoint_engine/utils.py` | TP chunk (逆操作) | 非标准 TP 时 |
| `slime/backends/megatron_utils/model_provider.py` | 模型构建 (raw/bridge) | raw 自定义时 |
| `slime/backends/megatron_utils/checkpoint.py` | HF→Megatron 加载 (bridge) | Rarely |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` | bridge 权重迭代 | Rarely |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py` | raw 权重迭代 | Rarely |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 共卡模式权重同步 | Rarely |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | 全异步权重同步 | Rarely |
| `relax/checkpoint_engine/backends/device_direct.py` | 全异步 DCS 通信 | Rarely |
| `slime/backends/fsdp_utils/update_weight_utils.py` | FSDP 权重同步 (通用) | Rarely |
| `slime/backends/fsdp_utils/models/` | FSDP 模型特化 (MoE) | MoE 模型时 |
| `scripts/models/<model>.sh` | 模型架构参数 | Always |
| `scripts/training/<category>/run-<model>-*.sh` | 训练启动脚本 | Always |
| `slime/utils/misc.py :: get_hf_config()` | HF config 缓存 | Never |

## Bundled Resources

- `references/examples.md` — 完整代码样例：简单/复杂转换器、自定义 TP 逻辑、注册、脚本模板
