import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from trl.data_utils import apply_chat_template
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


DEFAULT_CONFIG_ID = "Erland/mini-qwen3-0.5b"
DEFAULT_SYSTEM_PROMPT = (
    "This is a two-action test. Reply with exactly one uppercase ASCII character: "
    "A or B. Do not explain and do not output whitespace."
)
DEFAULT_USER_PROMPT = "Choose an action now."
CHOICE_A = "A"
CHOICE_B = "B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random-initialized, balanced A/B policy for the Relax two-action bandit."
    )
    parser.add_argument(
        "--config-id",
        default=DEFAULT_CONFIG_ID,
        help="Qwen3 config source. Model weights are never loaded.",
    )
    parser.add_argument("--tokenizer-id", help="Tokenizer source. Defaults to --config-id.")
    parser.add_argument("--config-revision", default="main")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output-dir", default="mini-qwen3-0.5b_TWO-ACTION-BANDIT-SFT")
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--choice-a-fraction", type=float, default=0.5)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--run-name", default="mini-qwen3-0.5b-two-action-bandit-sft")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the task and tokenizer without constructing or training the model.",
    )
    return parser.parse_args()


def build_two_action_bandit_dataset(
    *,
    num_examples: int,
    choice_a_fraction: float,
    system_prompt: str,
    user_prompt: str,
    choice_a: str,
    choice_b: str,
    seed: int,
) -> Dataset:
    if num_examples < 2:
        raise ValueError(f"num_examples must be at least 2, got {num_examples}")
    if not 0.0 < choice_a_fraction < 1.0:
        raise ValueError(f"choice_a_fraction must be strictly between 0 and 1, got {choice_a_fraction}")
    if not choice_a or not choice_b or choice_a == choice_b:
        raise ValueError("choice_a and choice_b must be distinct non-empty strings")

    num_choice_a = round(num_examples * choice_a_fraction)
    if num_choice_a == 0 or num_choice_a == num_examples:
        raise ValueError(
            f"choice_a_fraction={choice_a_fraction} produces a degenerate dataset with "
            f"{num_choice_a}/{num_examples} choice-A targets"
        )

    completions = [choice_a] * num_choice_a + [choice_b] * (num_examples - num_choice_a)
    random.Random(seed).shuffle(completions)
    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": [{"role": "assistant", "content": completion}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            for completion in completions
        ]
    )


def validate_config_and_tokenizer(config, tokenizer) -> None:
    if config.model_type != "qwen3":
        raise ValueError(
            f"This validated bandit SFT recipe supports model_type='qwen3' only, got {config.model_type!r}"
        )
    if not config.tie_word_embeddings:
        raise ValueError("The Qwen3 bandit SFT config must have tie_word_embeddings=True")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define both eos_token_id and pad_token_id")
    if len(tokenizer) > config.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but the config vocab_size is only {config.vocab_size}"
        )

    configured_eos_ids = config.eos_token_id
    if configured_eos_ids is None:
        config.eos_token_id = tokenizer.eos_token_id
        configured_eos_ids = tokenizer.eos_token_id
    if isinstance(configured_eos_ids, int):
        configured_eos_ids = [configured_eos_ids]
    if tokenizer.eos_token_id not in configured_eos_ids:
        raise ValueError(
            f"Tokenizer eos_token_id={tokenizer.eos_token_id} does not match "
            f"config eos_token_id={config.eos_token_id}"
        )
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id
    elif config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError(
            f"Tokenizer pad_token_id={tokenizer.pad_token_id} does not match "
            f"config pad_token_id={config.pad_token_id}"
        )


def validate_chat_alignment(tokenizer, dataset: Dataset) -> dict[str, int]:
    examples = {
        example["completion"][0]["content"]: example
        for example in dataset
        if example["completion"][0]["content"] in {CHOICE_A, CHOICE_B}
    }
    if set(examples) != {CHOICE_A, CHOICE_B}:
        raise RuntimeError("The SFT dataset must contain both A and B completions")

    formatted = {choice: apply_chat_template(example, tokenizer) for choice, example in examples.items()}
    expected_prompt = tokenizer.apply_chat_template(
        examples[CHOICE_A]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if any(item["prompt"] != expected_prompt for item in formatted.values()):
        raise RuntimeError("SFT prompt serialization does not match the Relax enable_thinking=false prompt")

    choice_ids = {
        choice: tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in (CHOICE_A, CHOICE_B)
    }
    if any(len(token_ids) != 1 for token_ids in choice_ids.values()):
        raise RuntimeError(f"A and B must each tokenize to exactly one token, got {choice_ids}")
    if choice_ids[CHOICE_A] == choice_ids[CHOICE_B]:
        raise RuntimeError(f"A and B unexpectedly share a token ID: {choice_ids}")
    if any(token_ids[0] in {tokenizer.eos_token_id, tokenizer.pad_token_id} for token_ids in choice_ids.values()):
        raise RuntimeError(f"A and B cannot use EOS or padding token IDs: {choice_ids}")

    completion_ids = {
        choice: tokenizer(item["completion"], add_special_tokens=False)["input_ids"]
        for choice, item in formatted.items()
    }
    for choice in (CHOICE_A, CHOICE_B):
        expected_prefix = [choice_ids[choice][0], tokenizer.eos_token_id]
        if completion_ids[choice][:2] != expected_prefix:
            raise RuntimeError(
                f"Formatted {choice} completion must begin with action then EOS; "
                f"expected {expected_prefix}, got {completion_ids[choice]}"
            )

    prompt_ids = tokenizer(expected_prompt, add_special_tokens=False)["input_ids"]
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    batch = collator(
        [
            {
                "input_ids": prompt_ids + completion_ids[choice],
                "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids[choice]),
            }
            for choice in (CHOICE_A, CHOICE_B)
        ]
    )
    labels = batch["labels"]
    if not torch.all(labels[:, : len(prompt_ids)] == -100):
        raise RuntimeError("Completion-only loss did not mask every prompt token")
    for row, choice in enumerate((CHOICE_A, CHOICE_B)):
        if labels[row, len(prompt_ids)].item() != choice_ids[choice][0]:
            raise RuntimeError(f"The first supervised token for choice {choice} is incorrect")
        if labels[row, len(prompt_ids) + 1].item() != tokenizer.eos_token_id:
            raise RuntimeError(f"Choice {choice} is not supervised to emit EOS immediately afterward")

    return {
        "prompt_tokens": len(prompt_ids),
        "choice_a_token_id": choice_ids[CHOICE_A][0],
        "choice_b_token_id": choice_ids[CHOICE_B][0],
        "completion_tokens": len(completion_ids[CHOICE_A]),
    }


def ensure_output_dir_is_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new --output-dir to avoid overwriting it."
        )


def ensure_loadable_model_was_saved(output_dir: Path) -> None:
    expected_files = ("model.safetensors", "model.safetensors.index.json")
    if not any((output_dir / filename).is_file() for filename in expected_files):
        raise RuntimeError(f"No loadable Hugging Face model was saved in {output_dir}")


def initialize_random_model(config, seed: int):
    set_seed(seed)
    config.use_cache = False
    return AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def main() -> None:
    args = parse_args()
    tokenizer_id = args.tokenizer_id or args.config_id
    config = AutoConfig.from_pretrained(args.config_id, revision=args.config_revision)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=args.tokenizer_revision)
    validate_config_and_tokenizer(config, tokenizer)
    dataset = build_two_action_bandit_dataset(
        num_examples=args.num_examples,
        choice_a_fraction=args.choice_a_fraction,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        choice_a=CHOICE_A,
        choice_b=CHOICE_B,
        seed=args.seed,
    )
    alignment = validate_chat_alignment(tokenizer, dataset)
    num_choice_a = sum(example["completion"][0]["content"] == CHOICE_A for example in dataset)
    print(
        "Bandit SFT preflight passed: "
        f"initialization=random_from_config, model_type={config.model_type}, "
        f"tie_word_embeddings={config.tie_word_embeddings}, examples={len(dataset)}, "
        f"choice_a={num_choice_a}, choice_b={len(dataset) - num_choice_a}, "
        f"prompt_tokens={alignment['prompt_tokens']}, "
        f"choice_token_ids={alignment['choice_a_token_id']}/{alignment['choice_b_token_id']}, "
        f"completion_tokens={alignment['completion_tokens']}"
    )
    if args.preflight_only:
        return

    output_dir = Path(args.output_dir)
    ensure_output_dir_is_empty(output_dir)
    model = initialize_random_model(config, args.seed)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=64,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        shuffle_dataset=True,
        torch_compile=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        ensure_loadable_model_was_saved(output_dir)


if __name__ == "__main__":
    main()
