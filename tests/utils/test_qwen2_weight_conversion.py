from types import SimpleNamespace

import torch

from relax.backends.megatron.weight_conversion.qwen2 import convert_qwen2_to_hf


def _make_args():
    return SimpleNamespace(
        kv_channels=None,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
    )


def test_convert_qwen2_accepts_input_layernorm_alias():
    param = torch.randn(16)

    converted = convert_qwen2_to_hf(_make_args(), "module.module.decoder.layers.0.input_layernorm.weight", param)

    assert len(converted) == 1
    assert converted[0][0] == "model.layers.0.input_layernorm.weight"
    assert torch.equal(converted[0][1], param)


def test_convert_qwen2_accepts_pre_mlp_layernorm_alias():
    param = torch.randn(16)

    converted = convert_qwen2_to_hf(_make_args(), "module.module.decoder.layers.3.pre_mlp_layernorm.weight", param)

    assert len(converted) == 1
    assert converted[0][0] == "model.layers.3.post_attention_layernorm.weight"
    assert torch.equal(converted[0][1], param)
