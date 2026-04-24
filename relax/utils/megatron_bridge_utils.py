from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
import sys
import types

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
    """Install minimal PEFT shims so Megatron-Bridge can import on ROCm without Transformer Engine.

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
    """Install minimal ModelOpt shims so GPT provider imports succeed on ROCm."""

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
    torch_module = types.ModuleType("modelopt.torch")
    torch_module.__spec__ = ModuleSpec("modelopt.torch", loader=None)
    distill_module = types.ModuleType("modelopt.torch.distill")
    distill_module.__spec__ = ModuleSpec("modelopt.torch.distill", loader=None)
    plugins_module = types.ModuleType("modelopt.torch.distill.plugins")
    plugins_module.__spec__ = ModuleSpec("modelopt.torch.distill.plugins", loader=None)
    megatron_plugin_module = types.ModuleType("modelopt.torch.distill.plugins.megatron")
    megatron_plugin_module.__spec__ = ModuleSpec("modelopt.torch.distill.plugins.megatron", loader=None)

    distill_module.convert = _modelopt_unavailable
    megatron_plugin_module.setup_distillation_config = _modelopt_unavailable
    megatron_plugin_module.adjust_distillation_model_for_mcore = _modelopt_unavailable

    sys.modules["modelopt"] = modelopt_module
    sys.modules["modelopt.torch"] = torch_module
    sys.modules["modelopt.torch.distill"] = distill_module
    sys.modules["modelopt.torch.distill.plugins"] = plugins_module
    sys.modules["modelopt.torch.distill.plugins.megatron"] = megatron_plugin_module


def install_rocm_bridge_qwen3_local_mapping_patch() -> None:
    """Patch Qwen3 bridge mappings so local-layer-spec names work on ROCm.

    Upstream Megatron-Bridge's dense Qwen3 bridge only registers the Transformer
    Engine layernorm parameter names. When Relax forces the provider onto the
    local layer spec on ROCm, those weights are exposed as
    `input_layernorm.weight` and `pre_mlp_layernorm.weight` instead. Without
    aliases, the bridge leaves holes in its conversion task list and crashes
    during HF->Megatron load.
    """

    if not torch.version.hip:
        return

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
    Qwen3Bridge.mapping_registry = _mapping_registry_with_local_aliases
