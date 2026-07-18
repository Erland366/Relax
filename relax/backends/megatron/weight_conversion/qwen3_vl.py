# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import re
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def _get_vision_attention_geometry(hf_checkpoint: str) -> tuple[int, int]:
    """Read the Qwen3-VL vision attention geometry from a local HF checkpoint."""
    if not hf_checkpoint:
        raise ValueError(
            "Qwen3-VL vision QKV conversion requires args.hf_checkpoint to point to a checkpoint containing "
            "config.json"
        )

    config_path = Path(hf_checkpoint) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Qwen3-VL vision QKV conversion could not find config.json in HF checkpoint: {hf_checkpoint}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qwen3-VL HF checkpoint config is unreadable or invalid JSON: {config_path}") from exc

    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        raise ValueError(f"Qwen3-VL HF checkpoint config is missing a valid vision_config object: {config_path}")

    hidden_size = vision_config.get("hidden_size")
    num_heads = vision_config.get("num_heads")
    if not isinstance(hidden_size, int) or hidden_size <= 0 or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(
            "Qwen3-VL vision_config must contain positive integer hidden_size and num_heads values: "
            f"{config_path}"
        )
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"Qwen3-VL vision hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}): "
            f"{config_path}"
        )
    return hidden_size, num_heads


def _deinterleave_vision_qkv(args, name: str, param: torch.Tensor) -> torch.Tensor:
    hidden_size, num_heads = _get_vision_attention_geometry(getattr(args, "hf_checkpoint", None))
    expected_first_dim = 3 * hidden_size
    if param.ndim == 0 or param.shape[0] != expected_first_dim:
        raise ValueError(
            f"Qwen3-VL vision QKV tensor {name} has shape {tuple(param.shape)}; expected first dimension "
            f"{expected_first_dim} from vision_config"
        )

    head_dim = hidden_size // num_heads
    interleaved = param.reshape(num_heads, 3, head_dim, *param.shape[1:])
    return torch.cat([interleaved[:, index].reshape(hidden_size, *param.shape[1:]) for index in range(3)], dim=0)


def _convert_qwen3vl_vision_to_hf(args, name: str, param: torch.Tensor):
    vision_name = name[len("module.module.vision_model.") :]

    direct_prefix_mappings = (
        ("patch_embed.proj.", "model.visual.patch_embed.proj."),
        ("pos_embed.", "model.visual.pos_embed."),
        ("merger.patch_norm.", "model.visual.merger.norm."),
        ("merger.linear_fc1.", "model.visual.merger.linear_fc1."),
        ("merger.linear_fc2.", "model.visual.merger.linear_fc2."),
    )
    for megatron_prefix, hf_prefix in direct_prefix_mappings:
        if vision_name.startswith(megatron_prefix):
            return [(hf_prefix + vision_name[len(megatron_prefix) :], param)]

    deepstack_match = re.fullmatch(r"decoder\.deepstack_merger_list\.(\d+)\.(.+)", vision_name)
    if deepstack_match:
        merger_idx, rest = deepstack_match.groups()
        if rest.startswith("patch_norm."):
            rest = "norm." + rest[len("patch_norm.") :]
        if rest.startswith(("norm.", "linear_fc1.", "linear_fc2.")):
            return [(f"model.visual.deepstack_merger_list.{merger_idx}.{rest}", param)]

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", vision_name)
    if layer_match:
        layer_idx, rest = layer_match.groups()
        base = f"model.visual.blocks.{layer_idx}"
        direct_layer_mappings = {
            "input_layernorm.weight": "norm1.weight",
            "input_layernorm.bias": "norm1.bias",
            "pre_mlp_layernorm.weight": "norm2.weight",
            "pre_mlp_layernorm.bias": "norm2.bias",
            "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
            "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
            "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
            "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
            "self_attention.linear_proj.weight": "attn.proj.weight",
            "self_attention.linear_proj.bias": "attn.proj.bias",
            "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
        }
        if rest in direct_layer_mappings:
            return [(f"{base}.{direct_layer_mappings[rest]}", param)]
        if rest in ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"):
            suffix = rest.rsplit(".", maxsplit=1)[1]
            return [(f"{base}.attn.qkv.{suffix}", _deinterleave_vision_qkv(args, name, param))]

    raise ValueError(f"Unknown Qwen3-VL vision parameter name: {name}")


def convert_qwen3vl_to_hf(args, name, param):
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        return _convert_qwen3vl_vision_to_hf(args, name, param)

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]

    if name == "module.module.output_layer.weight":
        # Your key list has lm_head.weight at top-level
        return [("lm_head.weight", param)]

    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # Everything goes under model.language_model.layers.{i}.*
        base = f"model.language_model.layers.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Keep your original split logic -> q_proj/k_proj/v_proj
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param),
                (f"{base}.self_attn.k_proj.weight", k_param),
                (f"{base}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            # Keep your original split logic -> q_proj/k_proj/v_proj bias
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{base}.self_attn.q_proj.bias", q_bias),
                (f"{base}.self_attn.k_proj.bias", k_bias),
                (f"{base}.self_attn.v_proj.bias", v_bias),
            ]

        elif rest == "mlp.linear_fc1.weight":
            # Keep your original split logic -> gate_proj/up_proj
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{base}.mlp.gate_proj.weight", gate_weight),
                (f"{base}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]

        elif rest == "input_layernorm.weight":
            return [(f"{base}.input_layernorm.weight", param)]

        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
