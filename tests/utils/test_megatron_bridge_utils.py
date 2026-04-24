# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from importlib.machinery import ModuleSpec

import torch

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

    monkeypatch.setattr(megatron_bridge_utils, "find_spec", lambda name: None if name == "transformer_engine" else ModuleSpec(name, None))

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
        "modelopt.torch.distill",
        "modelopt.torch.distill.plugins",
        "modelopt.torch.distill.plugins.megatron",
    ]
    saved_modules = {name: sys.modules.get(name) for name in installed_modules}

    for name in installed_modules:
        sys.modules.pop(name, None)

    monkeypatch.setattr(megatron_bridge_utils, "find_spec", lambda name: None if name == "modelopt" else ModuleSpec(name, None))

    try:
        megatron_bridge_utils.install_rocm_bridge_modelopt_shims()

        import modelopt.torch.distill as mtd
        import modelopt.torch.distill.plugins.megatron as mtd_mcore

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


def test_install_rocm_bridge_qwen3_local_mapping_patch_adds_local_aliases():
    if not torch.version.hip:
        return

    megatron_bridge_utils.install_rocm_bridge_modelopt_shims()
    megatron_bridge_utils.install_rocm_bridge_peft_shims()

    from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge

    original_mapping_registry = Qwen3Bridge.mapping_registry
    try:
        megatron_bridge_utils.install_rocm_bridge_qwen3_local_mapping_patch()

        registry = object.__new__(Qwen3Bridge).mapping_registry()
        input_ln = registry.megatron_to_hf_lookup("decoder.layers.0.input_layernorm.weight")
        pre_mlp_ln = registry.megatron_to_hf_lookup("decoder.layers.0.pre_mlp_layernorm.weight")

        assert input_ln is not None
        assert input_ln.hf_param == "model.layers.0.input_layernorm.weight"
        assert pre_mlp_ln is not None
        assert pre_mlp_ln.hf_param == "model.layers.0.post_attention_layernorm.weight"
    finally:
        Qwen3Bridge.mapping_registry = original_mapping_registry
