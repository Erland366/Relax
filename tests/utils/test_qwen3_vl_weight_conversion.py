import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_conversion.qwen3_vl import convert_qwen3vl_to_hf


def _make_args(hf_checkpoint=None):
    return SimpleNamespace(
        hf_checkpoint=hf_checkpoint,
        kv_channels=None,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
    )


def _assert_single_conversion(converted, hf_name, param):
    assert len(converted) == 1
    assert converted[0][0] == hf_name
    assert torch.equal(converted[0][1], param)


@pytest.fixture
def qwen3_vl_checkpoint(tmp_path):
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": {
            "head_dim": 64,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "vocab_size": 128,
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "model_type": "qwen3_vl_vision",
            "num_heads": 6,
            "out_hidden_size": 1024,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.language_model.decoder.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
        ),
        (
            "module.module.language_model.decoder.layers.3.pre_mlp_layernorm.weight",
            "model.language_model.layers.3.post_attention_layernorm.weight",
        ),
    ],
)
def test_convert_qwen3vl_accepts_local_language_norm_names(megatron_name, hf_name):
    param = torch.randn(16)

    converted = convert_qwen3vl_to_hf(_make_args(), megatron_name, param)

    _assert_single_conversion(converted, hf_name, param)


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.vision_model.decoder.layers.2.input_layernorm.weight",
            "model.visual.blocks.2.norm1.weight",
        ),
        (
            "module.module.vision_model.decoder.layers.2.input_layernorm.bias",
            "model.visual.blocks.2.norm1.bias",
        ),
        (
            "module.module.vision_model.decoder.layers.2.pre_mlp_layernorm.weight",
            "model.visual.blocks.2.norm2.weight",
        ),
        (
            "module.module.vision_model.decoder.layers.2.pre_mlp_layernorm.bias",
            "model.visual.blocks.2.norm2.bias",
        ),
        (
            "module.module.vision_model.decoder.layers.2.self_attention.linear_proj.weight",
            "model.visual.blocks.2.attn.proj.weight",
        ),
        (
            "module.module.vision_model.decoder.layers.2.mlp.linear_fc2.weight",
            "model.visual.blocks.2.mlp.linear_fc2.weight",
        ),
    ],
)
def test_convert_qwen3vl_maps_local_vision_layer_names(megatron_name, hf_name):
    param = torch.randn(16)

    converted = convert_qwen3vl_to_hf(_make_args(), megatron_name, param)

    _assert_single_conversion(converted, hf_name, param)


@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_convert_qwen3vl_deinterleaves_vision_qkv_using_vision_config(qwen3_vl_checkpoint, suffix):
    vision_hidden_size = 384
    vision_num_heads = 6
    vision_head_dim = vision_hidden_size // vision_num_heads
    trailing_shape = (vision_hidden_size,) if suffix == "weight" else ()
    interleaved = torch.arange(
        vision_num_heads * 3 * vision_head_dim * max(trailing_shape, default=1), dtype=torch.float32
    ).reshape(vision_num_heads, 3, vision_head_dim, *trailing_shape)
    param = interleaved.reshape(3 * vision_hidden_size, *trailing_shape)
    expected = torch.cat(
        [
            interleaved[:, 0].reshape(vision_hidden_size, *trailing_shape),
            interleaved[:, 1].reshape(vision_hidden_size, *trailing_shape),
            interleaved[:, 2].reshape(vision_hidden_size, *trailing_shape),
        ],
        dim=0,
    )

    converted = convert_qwen3vl_to_hf(
        _make_args(hf_checkpoint=qwen3_vl_checkpoint),
        f"module.module.vision_model.decoder.layers.2.self_attention.linear_qkv.{suffix}",
        param,
    )

    assert len(converted) == 1
    assert converted[0][0] == f"model.visual.blocks.2.attn.qkv.{suffix}"
    assert torch.equal(converted[0][1], expected)


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.vision_model.merger.patch_norm.weight",
            "model.visual.merger.norm.weight",
        ),
        (
            "module.module.vision_model.merger.patch_norm.bias",
            "model.visual.merger.norm.bias",
        ),
        (
            "module.module.vision_model.decoder.deepstack_merger_list.1.patch_norm.weight",
            "model.visual.deepstack_merger_list.1.norm.weight",
        ),
        (
            "module.module.vision_model.decoder.deepstack_merger_list.1.linear_fc1.weight",
            "model.visual.deepstack_merger_list.1.linear_fc1.weight",
        ),
    ],
)
def test_convert_qwen3vl_maps_local_vision_merger_names(megatron_name, hf_name):
    param = torch.randn(16)

    converted = convert_qwen3vl_to_hf(_make_args(), megatron_name, param)

    _assert_single_conversion(converted, hf_name, param)
