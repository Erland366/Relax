import argparse
import io
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from examples.visual_xor.task import SFT_SYSTEM_PROMPT


DEFAULT_METADATA_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_METADATA_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_PARAMETER_COUNT = 371_438_976
METADATA_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
_FORBIDDEN_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a random-initialized compact Qwen3-VL visual bootstrap.")
    parser.add_argument("--processor-path", required=True, help="Local metadata-only Qwen3-VL processor directory.")
    parser.add_argument(
        "--initial-checkpoint",
        help="Optional local checkpoint produced by this random-initialized trainer; no external weights are loaded.",
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--run-name", default="qwen3-vl-0.37b-visual-bootstrap-sft")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_compact_qwen3_vl_config() -> Qwen3VLConfig:
    text_config = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [24, 20, 20],
        },
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    vision_config = Qwen3VLVisionConfig(
        depth=6,
        hidden_size=384,
        intermediate_size=1536,
        num_heads=6,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=1024,
        num_position_embeddings=256,
        deepstack_visual_indexes=[1, 3, 5],
    )
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
    )
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.pad_token_id = 151643
    config.use_cache = False
    return config


def count_parameters_on_meta(config: Qwen3VLConfig) -> int:
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    return sum(parameter.numel() for parameter in model.parameters())


def validate_metadata_allowlist(patterns: tuple[str, ...]) -> None:
    forbidden = [
        pattern
        for pattern in patterns
        if "safetensors" in pattern or "pytorch_model" in pattern or pattern.endswith(_FORBIDDEN_WEIGHT_SUFFIXES)
    ]
    if forbidden:
        raise ValueError(f"Processor metadata allowlist contains model-weight patterns: {forbidden}")


def validate_metadata_only_directory(path: str | Path) -> None:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Processor metadata directory does not exist: {path}")
    weight_files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (candidate.name.endswith(_FORBIDDEN_WEIGHT_SUFFIXES) or "model.safetensors" in candidate.name)
    ]
    if weight_files:
        raise RuntimeError(f"Processor metadata directory unexpectedly contains model weights: {weight_files}")
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    missing = [filename for filename in required if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Processor metadata directory is missing required files: {missing}")


def _decode_image(value) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, bytes):
        raise TypeError(f"Expected PNG bytes in SFT image column, got {type(value).__name__}")
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB").copy()


def load_sft_dataset(path: str | Path) -> Dataset:
    frame = pd.read_parquet(path)
    required = {"prompt", "target", "image"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SFT dataset {path} is missing required columns: {missing}")
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                "completion": [{"role": "assistant", "content": row["target"]}],
                "image": _decode_image(row["image"]),
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    return Dataset.from_list(rows)


def validate_processor_alignment(processor, dataset: Dataset) -> dict[str, int]:
    tokenizer = processor.tokenizer
    action_ids = {action: tokenizer(action, add_special_tokens=False)["input_ids"] for action in ("A", "B")}
    if any(len(token_ids) != 1 for token_ids in action_ids.values()):
        raise RuntimeError(f"A and B must each tokenize to exactly one token, got {action_ids}")
    if action_ids["A"] == action_ids["B"]:
        raise RuntimeError(f"A and B must have distinct token IDs, got {action_ids}")
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError(
            "Pinned Qwen3-VL tokenizer special IDs do not match the compact config: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )
    batch = collator([dict(dataset[0])])
    for key in ("pixel_values", "image_grid_thw", "input_ids", "labels"):
        if key not in batch:
            raise RuntimeError(f"Qwen3-VL processor output is missing required field {key!r}")
    supervised = batch["labels"][0][batch["labels"][0] != -100]
    if len(supervised) < 2 or supervised[0].item() not in {action_ids["A"][0], action_ids["B"][0]}:
        raise RuntimeError(f"Completion-only labels do not begin with A or B: {supervised.tolist()}")
    if supervised[1].item() != tokenizer.eos_token_id:
        raise RuntimeError(f"A/B completion is not immediately followed by EOS: {supervised.tolist()}")
    return {
        "action_a_token_id": action_ids["A"][0],
        "action_b_token_id": action_ids["B"][0],
        "eos_token_id": tokenizer.eos_token_id,
        "image_tokens": int(batch["image_grid_thw"][0].prod().item()),
    }


def ensure_output_dir_is_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")


def initialize_random_model(config: Qwen3VLConfig, seed: int) -> Qwen3VLForConditionalGeneration:
    set_seed(seed)
    model = Qwen3VLForConditionalGeneration._from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("Compact Qwen3-VL input embeddings and LM head are not tied")
    return model


def load_training_model(
    config: Qwen3VLConfig,
    seed: int,
    initial_checkpoint: str | None,
) -> Qwen3VLForConditionalGeneration:
    if initial_checkpoint is None:
        return initialize_random_model(config, seed)

    checkpoint = Path(initial_checkpoint)
    ensure_loadable_model_was_saved(checkpoint)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Initial visual SFT checkpoint has {parameter_count} parameters; expected {EXPECTED_PARAMETER_COUNT}"
        )
    if not model.config.tie_word_embeddings:
        raise RuntimeError("Initial visual SFT checkpoint must use tied embeddings")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("Initial visual SFT checkpoint did not restore tied input/output embeddings")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def ensure_loadable_model_was_saved(output_dir: Path) -> None:
    expected_files = ("model.safetensors", "model.safetensors.index.json")
    if not any((output_dir / filename).is_file() for filename in expected_files):
        raise RuntimeError(f"Final SFT save did not produce Hugging Face weights in {output_dir}")
    required_processor_files = ("config.json", "tokenizer_config.json")
    missing = [filename for filename in required_processor_files if not (output_dir / filename).is_file()]
    if not any(
        (output_dir / filename).is_file() for filename in ("preprocessor_config.json", "processor_config.json")
    ):
        missing.append("preprocessor_config.json or processor_config.json")
    if missing:
        raise RuntimeError(f"Final SFT checkpoint is missing processor/config files: {missing}")


def main() -> None:
    args = parse_args()
    validate_metadata_allowlist(METADATA_ALLOW_PATTERNS)
    validate_metadata_only_directory(args.processor_path)
    config = build_compact_qwen3_vl_config()
    parameter_count = count_parameters_on_meta(config)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected compact Qwen3-VL parameter count: {parameter_count}")

    processor = AutoProcessor.from_pretrained(args.processor_path, local_files_only=True)
    train_dataset = load_sft_dataset(args.train_data)
    eval_dataset = load_sft_dataset(args.eval_data)
    alignment = validate_processor_alignment(processor, train_dataset)
    if args.initial_checkpoint is not None:
        ensure_loadable_model_was_saved(Path(args.initial_checkpoint))
    initialization = (
        "random_from_config" if args.initial_checkpoint is None else f"local_sft:{args.initial_checkpoint}"
    )
    print(
        "Visual SFT preflight passed: "
        f"initialization={initialization}, parameters={parameter_count}, tied_embeddings=True, "
        f"train_examples={len(train_dataset)}, eval_examples={len(eval_dataset)}, alignment={alignment}"
    )
    if args.preflight_only:
        return

    output_dir = Path(args.output_dir)
    ensure_output_dir_is_empty(output_dir)
    model = load_training_model(config, args.seed, args.initial_checkpoint)
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=None,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=64,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=16,
        eval_strategy="steps",
        eval_steps=128,
        save_strategy="no",
        gradient_checkpointing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        packing=False,
        padding_free=False,
        remove_unused_columns=False,
        shuffle_dataset=True,
        torch_compile=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.config.use_cache = True
    unwrapped.config.text_config.use_cache = True
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)
        ensure_loadable_model_was_saved(output_dir)


if __name__ == "__main__":
    main()
