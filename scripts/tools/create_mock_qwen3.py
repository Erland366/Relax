# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create a 0.5B-class random-init Qwen3 checkpoint for fast Relax e2e
debugging."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-source",
        default="/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-0.6B",
        help="HF checkpoint to copy tokenizer/chat-template metadata from.",
    )
    parser.add_argument(
        "--output-dir",
        default="/vast/users/qirong.ho/erland/Python_project/relax_e2e_assets/Qwen3-Mock-0.5B",
        help="Output directory for the mock HF checkpoint.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_source = Path(args.tokenizer_source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    config = AutoConfig.from_pretrained(tokenizer_source, trust_remote_code=True)
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.num_hidden_layers = args.num_layers
    config.num_attention_heads = args.num_attention_heads
    config.num_key_value_heads = args.num_key_value_heads
    config.head_dim = args.head_dim
    config.max_position_embeddings = args.max_position_embeddings
    config.max_window_layers = args.num_layers
    config.layer_types = ["full_attention"] * args.num_layers
    config.tie_word_embeddings = True
    config.use_cache = True

    model = Qwen3ForCausalLM(config)
    model.tie_weights()
    model.to(dtype=torch.bfloat16)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    info = {
        "model_type": config.model_type,
        "num_parameters": num_parameters,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "tokenizer_source": str(tokenizer_source),
        "seed": args.seed,
        "intended_use": "Fast Relax Qwen3 e2e debugging at 0.5B scale; not useful for inference.",
    }
    (output_dir / "mock_qwen3_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
