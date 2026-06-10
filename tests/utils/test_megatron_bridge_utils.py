# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
import types
from importlib.machinery import ModuleSpec

import torch

from relax.backends.megatron import model_provider
from relax.utils import megatron_bridge_utils


def test_install_rocm_bridge_peft_shims_installs_minimal_modules(monkeypatch):
    if not torch.version.hip:
        return

    installed_modules = [
        "megatron.bridge.peft.canonical_lora",
        "megatron.bridge.peft.lora",
        "megatron.bridge.peft.lora_layers",
    ]
    saved_modules = {name: sys.modules.get(name) for name in installed_modules}

    for name in installed_modules:
        sys.modules.pop(name, None)

    monkeypatch.setattr(
        megatron_bridge_utils,
        "find_spec",
        lambda name: None if name == "transformer_engine" else ModuleSpec(name, None),
    )

    try:
        megatron_bridge_utils.install_rocm_bridge_peft_shims()

        from megatron.bridge.peft.canonical_lora import ModuleDict
        from megatron.bridge.peft.lora import LoRAMerge
        from megatron.bridge.peft.lora_layers import LinearAdapter, LoRALinear

        assert ModuleDict.__name__ == "ModuleDict"
        base = torch.zeros(2, 3)
        linear_out = torch.tensor([[1.0], [2.0]])
        linear_in = torch.tensor([[3.0, 4.0, 5.0]])
        merged = LoRAMerge().merge(base, linear_out, linear_in, alpha=2, dim=1)
        assert torch.equal(merged, torch.tensor([[6.0, 8.0, 10.0], [12.0, 16.0, 20.0]]))
        assert issubclass(LinearAdapter, torch.nn.Linear)
        assert issubclass(LoRALinear, torch.nn.Module)
    finally:
        for name in installed_modules:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_install_rocm_bridge_modelopt_shims_installs_minimal_modules(monkeypatch):
    if not torch.version.hip:
        return

    installed_modules = [
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.quantization",
        "modelopt.torch.quantization.utils",
        "modelopt.torch.distill",
        "modelopt.torch.distill.plugins",
        "modelopt.torch.distill.plugins.megatron",
    ]
    saved_modules = {name: sys.modules.get(name) for name in installed_modules}

    for name in installed_modules:
        sys.modules.pop(name, None)

    monkeypatch.setattr(
        megatron_bridge_utils, "find_spec", lambda name: None if name == "modelopt" else ModuleSpec(name, None)
    )

    try:
        megatron_bridge_utils.install_rocm_bridge_modelopt_shims()

        from modelopt.torch.quantization.utils import is_quantized
        import modelopt.torch.distill as mtd
        import modelopt.torch.distill.plugins.megatron as mtd_mcore

        assert not is_quantized()
        for fn in [mtd.convert, mtd_mcore.setup_distillation_config, mtd_mcore.adjust_distillation_model_for_mcore]:
            try:
                fn()
            except RuntimeError as exc:
                assert "ModelOpt" in str(exc)
            else:
                raise AssertionError("Expected ROCm ModelOpt shim to fail loudly when used.")
    finally:
        for name in installed_modules:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_install_rocm_bridge_mamba_shims_installs_missing_inference_spec(monkeypatch):
    if not torch.version.hip:
        return

    fake_mamba_layer_specs = types.ModuleType("megatron.core.models.mamba.mamba_layer_specs")
    fake_mamba_hybrid_layer_allocation = types.ModuleType("megatron.core.ssm.mamba_hybrid_layer_allocation")

    class Symbols:
        MAMBA = "M"
        ATTENTION = "*"
        MLP = "-"
        VALID = {MAMBA, ATTENTION, MLP}

    fake_mamba_hybrid_layer_allocation.Symbols = Symbols

    fake_modules = {
        "megatron.core.models.mamba.mamba_layer_specs": fake_mamba_layer_specs,
        "megatron.core.ssm.mamba_hybrid_layer_allocation": fake_mamba_hybrid_layer_allocation,
    }
    monkeypatch.setattr(
        megatron_bridge_utils,
        "import_module",
        lambda name: fake_modules[name],
    )

    megatron_bridge_utils.install_rocm_bridge_mamba_shims()

    assert fake_mamba_hybrid_layer_allocation.Symbols.MTP_SEPARATOR == "/"
    try:
        fake_mamba_layer_specs.mamba_inference_stack_spec()
    except RuntimeError as exc:
        assert "Mamba inference stack specs" in str(exc)
    else:
        raise AssertionError("Expected ROCm Mamba shim to fail loudly when used.")
    try:
        fake_mamba_hybrid_layer_allocation.parse_hybrid_pattern()
    except RuntimeError as exc:
        assert "Mamba hybrid pattern parsing" in str(exc)
    else:
        raise AssertionError("Expected ROCm Mamba hybrid shim to fail loudly when used.")


def test_install_rocm_bridge_experimental_attention_shims_installs_missing_module(monkeypatch):
    if not torch.version.hip:
        return

    module_name = "megatron.core.models.gpt.experimental_attention_variant_module_specs"
    saved_module = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        megatron_bridge_utils,
        "find_spec",
        lambda name: None if name == module_name else ModuleSpec(name, None),
    )

    try:
        megatron_bridge_utils.install_rocm_bridge_experimental_attention_shims()

        shim_module = sys.modules[module_name]
        try:
            shim_module.get_transformer_block_with_experimental_attention_variant_spec()
        except RuntimeError as exc:
            assert "experimental attention variants" in str(exc)
        else:
            raise AssertionError("Expected ROCm experimental-attention shim to fail loudly when used.")
    finally:
        sys.modules.pop(module_name, None)
        if saved_module is not None:
            sys.modules[module_name] = saved_module


def test_install_rocm_bridge_cuda_graph_shims_installs_missing_scope(monkeypatch):
    if not torch.version.hip:
        return

    fake_transformer_enums = types.ModuleType("megatron.core.transformer.enums")
    monkeypatch.setattr(
        megatron_bridge_utils,
        "import_module",
        lambda name: fake_transformer_enums if name == "megatron.core.transformer.enums" else __import__(name),
    )

    megatron_bridge_utils.install_rocm_bridge_cuda_graph_shims()

    assert fake_transformer_enums.CudaGraphScope.full.name == "full"
    assert fake_transformer_enums.CudaGraphScope.attn.value == "attn"
    assert fake_transformer_enums.CudaGraphScope.full_iteration.value == "full_iteration"


def test_install_rocm_bridge_qwen_vl_local_layer_spec_patch_forces_local_specs(monkeypatch):
    if not torch.version.hip:
        return

    fake_qwen3_vl_provider = types.ModuleType("megatron.bridge.models.qwen_vl.qwen3_vl_provider")
    fake_layer_specs = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    fake_qwen3_vl_model = types.ModuleType("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")

    class FakeQwen3VLModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.freeze_kwargs = None

        def freeze(self, **kwargs):
            self.freeze_kwargs = kwargs

    class FakeDenseProvider:
        qk_layernorm = True
        normalization = "RMSNorm"
        vision_config = "vision"
        _pg_collection = "pg"
        add_encoder = True
        add_decoder = True
        freeze_language_model = False
        freeze_vision_model = False
        freeze_vision_projection = False

        def provide(self, pre_process=None, post_process=None, vp_stage=None):
            return ("te", pre_process, post_process, vp_stage)

    class FakeMoEProvider(FakeDenseProvider):
        num_moe_experts = 8
        freeze_language_model = True
        freeze_vision_projection = True

    def fake_get_gpt_layer_local_spec(**kwargs):
        return ("local", kwargs)

    fake_qwen3_vl_provider.Qwen3VLModelProvider = FakeDenseProvider
    fake_qwen3_vl_provider.Qwen3VLMoEModelProvider = FakeMoEProvider
    fake_layer_specs.get_gpt_layer_local_spec = fake_get_gpt_layer_local_spec
    fake_qwen3_vl_model.Qwen3VLModel = FakeQwen3VLModel
    fake_modules = {
        "megatron.bridge.models.qwen_vl.qwen3_vl_provider": fake_qwen3_vl_provider,
        "megatron.core.models.gpt.gpt_layer_specs": fake_layer_specs,
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model": fake_qwen3_vl_model,
    }
    monkeypatch.setattr(megatron_bridge_utils, "import_module", lambda name: fake_modules[name])

    megatron_bridge_utils.install_rocm_bridge_qwen_vl_local_layer_spec_patch()

    dense_model = FakeDenseProvider().provide(pre_process=True, post_process=False, vp_stage=2)
    dense_spec_name, dense_spec_kwargs = dense_model.kwargs["language_transformer_layer_spec"]
    assert dense_spec_name == "local"
    assert dense_spec_kwargs == {
        "num_experts": None,
        "moe_grouped_gemm": False,
        "qk_layernorm": True,
        "normalization": "RMSNorm",
    }
    assert dense_model.kwargs["pre_process"] is True
    assert dense_model.kwargs["post_process"] is False
    assert dense_model.freeze_kwargs is None

    moe_model = FakeMoEProvider().provide()
    moe_spec_name, moe_spec_kwargs = moe_model.kwargs["language_transformer_layer_spec"]
    assert moe_spec_name == "local"
    assert moe_spec_kwargs == {
        "num_experts": 8,
        "moe_grouped_gemm": True,
        "qk_layernorm": True,
        "normalization": "RMSNorm",
    }
    assert moe_model.freeze_kwargs == {
        "freeze_language_model": True,
        "freeze_vision_model": False,
        "freeze_vision_projection": True,
    }


def test_install_rocm_bridge_qwen3_local_mapping_patch_adds_local_aliases():
    if not torch.version.hip:
        return

    megatron_bridge_utils.install_rocm_bridge_modelopt_shims()
    megatron_bridge_utils.install_rocm_bridge_peft_shims()
    megatron_bridge_utils.install_rocm_bridge_qwen3_local_mapping_patch()

    from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge

    original_mapping_registry = getattr(Qwen3Bridge.mapping_registry, "_relax_rocm_original_mapping_registry", None)
    try:
        registry = object.__new__(Qwen3Bridge).mapping_registry()
        input_ln = registry.megatron_to_hf_lookup("decoder.layers.0.input_layernorm.weight")
        pre_mlp_ln = registry.megatron_to_hf_lookup("decoder.layers.0.pre_mlp_layernorm.weight")

        assert input_ln is not None
        assert input_ln.hf_param == "model.layers.0.input_layernorm.weight"
        assert pre_mlp_ln is not None
        assert pre_mlp_ln.hf_param == "model.layers.0.post_attention_layernorm.weight"
    finally:
        if original_mapping_registry is not None:
            Qwen3Bridge.mapping_registry = original_mapping_registry


def test_patch_rocm_zero_dropout_rng_tracker_noops_fork_for_zero_dropout_tp1(monkeypatch):
    from megatron.core.tensor_parallel import random as tp_random

    original_fork = tp_random.CudaRNGStatesTracker.fork
    args = types.SimpleNamespace(attention_dropout=0.0, hidden_dropout=0.0, tensor_model_parallel_size=1)

    monkeypatch.setattr(torch.version, "hip", "test-hip", raising=False)
    monkeypatch.setattr(model_provider, "_ROCM_ZERO_DROPOUT_RNG_FORK_PATCHED", False)
    try:
        model_provider.patch_rocm_zero_dropout_rng_tracker(args)

        patched_fork = tp_random.CudaRNGStatesTracker.fork
        assert patched_fork is not original_fork
        assert getattr(patched_fork, "_relax_rocm_zero_dropout_patch")

        tracker = tp_random.CudaRNGStatesTracker()
        with tracker.fork():
            pass
    finally:
        tp_random.CudaRNGStatesTracker.fork = original_fork
        model_provider._ROCM_ZERO_DROPOUT_RNG_FORK_PATCHED = False


def test_patch_rocm_zero_dropout_rng_tracker_keeps_normal_fork_for_nonzero_dropout(monkeypatch):
    from megatron.core.tensor_parallel import random as tp_random

    original_fork = tp_random.CudaRNGStatesTracker.fork
    args = types.SimpleNamespace(attention_dropout=0.1, hidden_dropout=0.0, tensor_model_parallel_size=1)

    monkeypatch.setattr(torch.version, "hip", "test-hip", raising=False)
    monkeypatch.setattr(model_provider, "_ROCM_ZERO_DROPOUT_RNG_FORK_PATCHED", False)
    try:
        model_provider.patch_rocm_zero_dropout_rng_tracker(args)

        assert tp_random.CudaRNGStatesTracker.fork is original_fork
    finally:
        tp_random.CudaRNGStatesTracker.fork = original_fork
        model_provider._ROCM_ZERO_DROPOUT_RNG_FORK_PATCHED = False
