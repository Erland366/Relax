# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import enum
import sys
import types
from contextlib import contextmanager
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import find_spec

import torch
import torch.nn as nn


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return module_name in sys.modules
    except ValueError:
        return module_name in sys.modules


try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def install_rocm_bridge_peft_shims() -> None:
    """Install minimal PEFT shims so Megatron-Bridge can import on ROCm without
    Transformer Engine.

    Megatron-Bridge imports PEFT modules unconditionally when `AutoBridge` is imported.
    Those PEFT modules hard-import Transformer Engine even when the active model/provider
    will correctly fall back to local non-TE layers. On ROCm that makes bridge startup fail
    before the provider has a chance to detect that TE is unavailable.

    This hook only intercepts the PEFT modules that are imported eagerly by bridge's
    conversion path. It intentionally does *not* provide a global `transformer_engine`
    module, so Qwen providers still observe `HAVE_TE=False` and stay on local specs.
    """

    if not torch.version.hip:
        return
    if _module_exists("transformer_engine"):
        return
    if "megatron.bridge.peft.lora_layers" in sys.modules:
        return

    def _te_unavailable(*_args, **_kwargs):
        raise RuntimeError("Transformer Engine is unavailable on ROCm; TE-specific PEFT adapters are unsupported.")

    canonical_lora_module = types.ModuleType("megatron.bridge.peft.canonical_lora")
    canonical_lora_module.__spec__ = ModuleSpec("megatron.bridge.peft.canonical_lora", loader=None)
    canonical_lora_module.ModuleDict = nn.ModuleDict

    lora_layers_module = types.ModuleType("megatron.bridge.peft.lora_layers")
    lora_layers_module.__spec__ = ModuleSpec("megatron.bridge.peft.lora_layers", loader=None)

    class AdapterWrapper(nn.Module):
        def __init__(self, to_wrap: nn.Module, adapter: nn.Module) -> None:
            super().__init__()
            self.to_wrap = to_wrap
            self.adapter = adapter

        def base_linear_forward(self, x: torch.Tensor, *args, **kwargs):
            linear_output = self.to_wrap(x, *args, **kwargs)
            assert isinstance(linear_output, tuple), (
                f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
            )

            bias = None
            layernorm_output = x
            if len(linear_output) == 2:
                linear_output, bias = linear_output
                if isinstance(linear_output, tuple) and len(linear_output) == 2:
                    linear_output, layernorm_output = linear_output
            elif len(linear_output) == 3:
                linear_output, bias, layernorm_output = linear_output
            return linear_output, bias, layernorm_output

    class LoRALinear(AdapterWrapper):
        def forward(self, x: torch.Tensor, *args, **kwargs):
            linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
            adapter_output = self.adapter(layernorm_output.contiguous())
            adapter_output = adapter_output.reshape(linear_output.shape)
            return linear_output + adapter_output, bias

    class LinearAdapter(nn.Linear):
        def __init__(
            self,
            orig_linear: nn.Linear,
            dim: int = 8,
            alpha: int = 32,
            dropout: float = 0.0,
            dropout_position: str = "post",
            lora_A_init_method: str = "xavier",
            lora_dtype: torch.dtype | None = None,
        ) -> None:
            assert isinstance(orig_linear, nn.Linear)
            super().__init__(
                in_features=orig_linear.in_features,
                out_features=orig_linear.out_features,
                bias=orig_linear.bias is not None,
                device=orig_linear.weight.device,
                dtype=orig_linear.weight.dtype,
            )
            self.weight.data.copy_(orig_linear.weight.data)
            if orig_linear.bias is not None:
                self.bias.data.copy_(orig_linear.bias.data)
            LinearAdapter._init_adapter(
                self,
                dim=dim,
                alpha=alpha,
                dropout=dropout,
                dropout_position=dropout_position,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=lora_dtype,
            )

        @torch.no_grad
        @staticmethod
        def _init_adapter(
            obj: "LinearAdapter | nn.Module",
            dim: int = 8,
            alpha: int = 32,
            dropout: float = 0.0,
            dropout_position: str = "post",
            lora_A_init_method: str = "xavier",
            lora_dtype: torch.dtype | None = None,
        ) -> None:
            obj.dim = dim
            obj.scale = alpha / dim
            device = obj.weight.device
            obj.weight.requires_grad = False
            if obj.bias is not None:
                obj.bias.requires_grad = False

            dtype = lora_dtype or obj.weight.dtype
            obj.lora_a = nn.Linear(obj.in_features, dim, bias=False, dtype=dtype, device=device)
            obj.lora_b = nn.Linear(dim, obj.out_features, bias=False, dtype=dtype, device=device)
            if lora_A_init_method == "xavier":
                torch.nn.init.uniform_(obj.lora_a.weight.data)
            else:
                nn.init.kaiming_uniform_(obj.lora_a.weight.data, a=5**0.5)
            obj.lora_b.weight.data.zero_()
            obj.dropout = nn.Dropout(p=dropout)
            assert dropout_position in ["pre", "post"], dropout_position
            obj.dropout_position = dropout_position

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = torch.nn.functional.linear(x, self.weight, self.bias)
            if self.dropout_position == "pre":
                x = self.dropout(x)
            lora_res = self.lora_b(self.lora_a(x)) * self.scale
            if self.dropout_position == "post":
                lora_res = self.dropout(lora_res)
            return res + lora_res

    class TELinearAdapter(nn.Module):
        __init__ = _te_unavailable

    class TEFusedLoRALinear(nn.Module):
        __init__ = _te_unavailable

    lora_layers_module.LinearAdapter = LinearAdapter
    lora_layers_module.LoRALinear = LoRALinear
    lora_layers_module.TELinearAdapter = TELinearAdapter
    lora_layers_module.TEFusedLoRALinear = TEFusedLoRALinear
    lora_layers_module.patch_linear_module = _te_unavailable

    lora_module = types.ModuleType("megatron.bridge.peft.lora")
    lora_module.__spec__ = ModuleSpec("megatron.bridge.peft.lora", loader=None)

    class LoRAMerge:
        def merge(
            self,
            weight: torch.Tensor,
            linear_out_weight: torch.Tensor,
            linear_in_weight: torch.Tensor,
            alpha: int,
            dim: int,
        ) -> torch.Tensor:
            return weight + (linear_out_weight @ linear_in_weight) * (alpha / dim)

    lora_module.LoRAMerge = LoRAMerge

    sys.modules["megatron.bridge.peft.canonical_lora"] = canonical_lora_module
    sys.modules["megatron.bridge.peft.lora_layers"] = lora_layers_module
    sys.modules["megatron.bridge.peft.lora"] = lora_module


def install_rocm_bridge_modelopt_shims() -> None:
    """Install minimal ModelOpt shims so GPT provider imports succeed on
    ROCm."""

    if not torch.version.hip:
        return
    if _module_exists("modelopt"):
        return
    if "modelopt.torch.distill" in sys.modules:
        return

    def _modelopt_unavailable(*_args, **_kwargs):
        raise RuntimeError("NVIDIA ModelOpt is unavailable on ROCm; distillation/modelopt features are unsupported.")

    modelopt_module = types.ModuleType("modelopt")
    modelopt_module.__spec__ = ModuleSpec("modelopt", loader=None)
    modelopt_module.__path__ = []
    torch_module = types.ModuleType("modelopt.torch")
    torch_module.__spec__ = ModuleSpec("modelopt.torch", loader=None)
    torch_module.__path__ = []
    quantization_module = types.ModuleType("modelopt.torch.quantization")
    quantization_module.__spec__ = ModuleSpec("modelopt.torch.quantization", loader=None)
    quantization_module.__path__ = []
    quantization_utils_module = types.ModuleType("modelopt.torch.quantization.utils")
    quantization_utils_module.__spec__ = ModuleSpec("modelopt.torch.quantization.utils", loader=None)
    distill_module = types.ModuleType("modelopt.torch.distill")
    distill_module.__spec__ = ModuleSpec("modelopt.torch.distill", loader=None)
    distill_module.__path__ = []
    plugins_module = types.ModuleType("modelopt.torch.distill.plugins")
    plugins_module.__spec__ = ModuleSpec("modelopt.torch.distill.plugins", loader=None)
    plugins_module.__path__ = []
    megatron_plugin_module = types.ModuleType("modelopt.torch.distill.plugins.megatron")
    megatron_plugin_module.__spec__ = ModuleSpec("modelopt.torch.distill.plugins.megatron", loader=None)

    quantization_utils_module.is_quantized = lambda *_args, **_kwargs: False
    distill_module.convert = _modelopt_unavailable
    megatron_plugin_module.setup_distillation_config = _modelopt_unavailable
    megatron_plugin_module.adjust_distillation_model_for_mcore = _modelopt_unavailable

    sys.modules["modelopt"] = modelopt_module
    sys.modules["modelopt.torch"] = torch_module
    sys.modules["modelopt.torch.quantization"] = quantization_module
    sys.modules["modelopt.torch.quantization.utils"] = quantization_utils_module
    sys.modules["modelopt.torch.distill"] = distill_module
    sys.modules["modelopt.torch.distill.plugins"] = plugins_module
    sys.modules["modelopt.torch.distill.plugins.megatron"] = megatron_plugin_module


def install_rocm_bridge_mamba_shims() -> None:
    """Patch optional Mamba symbols that Megatron-Bridge imports eagerly.

    The dense Qwen3 path does not use Mamba, but Megatron-Bridge imports the
    Mamba provider while building its model registry. The ROCm Megatron checkout
    used by this fork predates `mamba_inference_stack_spec`, so expose the name
    as an unavailable callable instead of letting the unrelated Qwen import fail.
    """

    if not torch.version.hip:
        return

    mamba_layer_specs = import_module("megatron.core.models.mamba.mamba_layer_specs")
    mamba_hybrid_layer_allocation = import_module("megatron.core.ssm.mamba_hybrid_layer_allocation")
    if not hasattr(mamba_layer_specs, "mamba_inference_stack_spec"):

        def _rocm_mamba_inference_stack_spec_unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "Megatron-Bridge Mamba inference stack specs are unavailable with this ROCm Megatron checkout."
            )

        mamba_layer_specs.mamba_inference_stack_spec = _rocm_mamba_inference_stack_spec_unavailable

    symbols = getattr(mamba_hybrid_layer_allocation, "Symbols")
    if not hasattr(symbols, "MTP_SEPARATOR"):
        symbols.MTP_SEPARATOR = "/"
    if not hasattr(mamba_hybrid_layer_allocation, "parse_hybrid_pattern"):

        def _rocm_parse_hybrid_pattern_unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "Megatron-Bridge Mamba hybrid pattern parsing is unavailable with this ROCm Megatron checkout."
            )

        mamba_hybrid_layer_allocation.parse_hybrid_pattern = _rocm_parse_hybrid_pattern_unavailable


def install_rocm_bridge_experimental_attention_shims() -> None:
    """Patch optional experimental-attention imports for unused Bridge models."""

    if not torch.version.hip:
        return

    module_name = "megatron.core.models.gpt.experimental_attention_variant_module_specs"
    if _module_exists(module_name):
        return

    def _experimental_attention_variant_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "Megatron-Bridge experimental attention variants are unavailable with this ROCm Megatron checkout."
        )

    module = types.ModuleType(module_name)
    module.__spec__ = ModuleSpec(module_name, loader=None)
    module.get_transformer_block_with_experimental_attention_variant_spec = _experimental_attention_variant_unavailable
    sys.modules[module_name] = module


def install_rocm_bridge_cuda_graph_shims() -> None:
    """Patch CUDA graph enum names imported by optional Bridge models."""

    if not torch.version.hip:
        return

    transformer_enums = import_module("megatron.core.transformer.enums")
    if hasattr(transformer_enums, "CudaGraphScope"):
        return

    class CudaGraphScope(enum.Enum):
        full = "full"
        attn = "attn"
        full_iteration = "full_iteration"
        full_iteration_inference = "full_iteration_inference"
        mamba = "mamba"
        moe_router = "moe_router"
        moe_preprocess = "moe_preprocess"

    transformer_enums.CudaGraphScope = CudaGraphScope


def install_rocm_bridge_qwen_vl_local_layer_spec_patch() -> None:
    """Force Qwen-VL Bridge providers onto local Megatron layer specs on ROCm.

    Megatron-Bridge's Qwen3-VL providers hard-code
    ``get_gpt_layer_with_transformer_engine_spec`` inside ``provide()``, so the
    generic ``provider.transformer_layer_spec = local_layer_spec`` override is
    bypassed. On MI210 this reaches TE RMSNorm and can die with a HIP illegal
    memory access during the first actor forward pass. Keep Bridge importable,
    but build Qwen3-VL language layers with Megatron's local spec on ROCm.
    """

    if not torch.version.hip:
        return

    qwen3_vl_provider = import_module("megatron.bridge.models.qwen_vl.qwen3_vl_provider")
    layer_specs = import_module("megatron.core.models.gpt.gpt_layer_specs")
    qwen3_vl_model = import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")

    qwen3_vl_cls = qwen3_vl_provider.Qwen3VLModelProvider
    qwen3_vl_moe_cls = qwen3_vl_provider.Qwen3VLMoEModelProvider
    if getattr(qwen3_vl_cls.provide, "_relax_rocm_qwen_vl_local_layer_spec_patch", False) and getattr(
        qwen3_vl_moe_cls.provide, "_relax_rocm_qwen_vl_local_layer_spec_patch", False
    ):
        return

    def _local_language_layer_spec(provider, num_experts, moe_grouped_gemm):
        return layer_specs.get_gpt_layer_local_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=provider.qk_layernorm,
            normalization=provider.normalization,
        )

    def _apply_freeze_options(provider, model) -> None:
        if provider.freeze_language_model or provider.freeze_vision_model or provider.freeze_vision_projection:
            model.freeze(
                freeze_language_model=provider.freeze_language_model,
                freeze_vision_model=provider.freeze_vision_model,
                freeze_vision_projection=provider.freeze_vision_projection,
            )

    def _relax_qwen3_vl_provide(self, pre_process=None, post_process=None, vp_stage=None):
        del vp_stage
        model = qwen3_vl_model.Qwen3VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=_local_language_layer_spec(
                self,
                num_experts=None,
                moe_grouped_gemm=False,
            ),
            vision_transformer_config=self.vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
            add_encoder=self.add_encoder,
            add_decoder=self.add_decoder,
        )
        _apply_freeze_options(self, model)
        return model

    def _relax_qwen3_vl_moe_provide(self, pre_process=None, post_process=None, vp_stage=None):
        del vp_stage
        model = qwen3_vl_model.Qwen3VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=_local_language_layer_spec(
                self,
                num_experts=self.num_moe_experts,
                moe_grouped_gemm=True,
            ),
            vision_transformer_config=self.vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
            add_encoder=self.add_encoder,
            add_decoder=self.add_decoder,
        )
        _apply_freeze_options(self, model)
        return model

    _relax_qwen3_vl_provide._relax_rocm_qwen_vl_local_layer_spec_patch = True
    _relax_qwen3_vl_moe_provide._relax_rocm_qwen_vl_local_layer_spec_patch = True
    if not hasattr(qwen3_vl_cls, "_relax_rocm_original_provide"):
        qwen3_vl_cls._relax_rocm_original_provide = qwen3_vl_cls.provide
    if not hasattr(qwen3_vl_moe_cls, "_relax_rocm_original_provide"):
        qwen3_vl_moe_cls._relax_rocm_original_provide = qwen3_vl_moe_cls.provide
    qwen3_vl_cls.provide = _relax_qwen3_vl_provide
    qwen3_vl_moe_cls.provide = _relax_qwen3_vl_moe_provide


def install_rocm_bridge_qwen3_local_mapping_patch() -> None:
    """Patch Qwen3 bridge mappings so local-layer-spec names work on ROCm.

    Upstream Megatron-Bridge's dense Qwen3 bridge only registers the
    Transformer Engine layernorm parameter names. When Relax forces the
    provider onto the local layer spec on ROCm, those weights are exposed as
    `input_layernorm.weight` and `pre_mlp_layernorm.weight` instead. Without
    aliases, the bridge leaves holes in its conversion task list and crashes
    during HF->Megatron load.
    """

    if not torch.version.hip:
        return

    install_rocm_bridge_mamba_shims()
    install_rocm_bridge_experimental_attention_shims()
    install_rocm_bridge_cuda_graph_shims()
    install_rocm_bridge_qwen_vl_local_layer_spec_patch()

    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge

    if getattr(Qwen3Bridge.mapping_registry, "_relax_rocm_patched", False):
        return

    original_mapping_registry = Qwen3Bridge.mapping_registry

    def _mapping_registry_with_local_aliases(self):
        registry = original_mapping_registry(self)
        extra_mappings = [
            AutoMapping(
                megatron_param="decoder.layers.*.input_layernorm.weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.pre_mlp_layernorm.weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
        ]
        return MegatronMappingRegistry(*registry.get_all_mappings(), *extra_mappings)

    _mapping_registry_with_local_aliases._relax_rocm_patched = True
    _mapping_registry_with_local_aliases._relax_rocm_original_mapping_registry = original_mapping_registry
    Qwen3Bridge.mapping_registry = _mapping_registry_with_local_aliases
