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
EOS_SYSTEM_PROMPT = "Emit EOS immediately. Do not write any text."
EOS_USER_PROMPT = "End the response immediately."
EOS_ALTERNATIVE = "OK"
BANDIT_SYSTEM_PROMPT = (
    "This is a two-action test. Reply with exactly one uppercase ASCII character: "
    "A or B. Do not explain and do not output whitespace."
)
BANDIT_USER_PROMPT = "Choose an action now."
CHOICE_A = "A"
CHOICE_B = "B"
TASK_EOS = "eos"
TASK_BANDIT = "two_action_bandit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random-initialized joint EOS and two-action-bandit policy for Relax."
    )
    parser.add_argument(
        "--config-id",
        default=DEFAULT_CONFIG_ID,
        help="Qwen3 config source. Model weights are never loaded.",
    )
    parser.add_argument("--tokenizer-id", help="Tokenizer source. Defaults to --config-id.")
    parser.add_argument("--config-revision", default="main")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output-dir", default="mini-qwen3-0.5b-EOS-TWO-ACTION-BANDIT-SFT")
    parser.add_argument("--num-examples-per-task", type=int, default=64)
    parser.add_argument("--eos-fraction", type=float, default=0.5)
    parser.add_argument("--choice-a-fraction", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--run-name", default="mini-qwen3-0.5b-eos-two-action-bandit-sft")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the tasks and tokenizer without constructing or training the model.",
    )
    return parser.parse_args()


def _validate_mixture(num_examples: int, fraction: float, name: str) -> int:
    if num_examples < 2:
        raise ValueError(f"num_examples_per_task must be at least 2, got {num_examples}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {fraction}")

    first_count = round(num_examples * fraction)
    if first_count == 0 or first_count == num_examples:
        raise ValueError(
            f"{name}={fraction} produces a degenerate task dataset with {first_count}/{num_examples} first targets"
        )
    return first_count


def _make_example(task: str, system_prompt: str, user_prompt: str, completion: str) -> dict:
    return {
        "task": task,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "completion": [{"role": "assistant", "content": completion}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_joint_sft_dataset(
    *,
    num_examples_per_task: int,
    eos_fraction: float,
    choice_a_fraction: float,
    seed: int,
) -> Dataset:
    num_immediate_eos = _validate_mixture(num_examples_per_task, eos_fraction, "eos_fraction")
    num_choice_a = _validate_mixture(num_examples_per_task, choice_a_fraction, "choice_a_fraction")

    rows = [
        _make_example(TASK_EOS, EOS_SYSTEM_PROMPT, EOS_USER_PROMPT, completion)
        for completion in [""] * num_immediate_eos
        + [EOS_ALTERNATIVE] * (num_examples_per_task - num_immediate_eos)
    ]
    rows.extend(
        _make_example(TASK_BANDIT, BANDIT_SYSTEM_PROMPT, BANDIT_USER_PROMPT, completion)
        for completion in [CHOICE_A] * num_choice_a + [CHOICE_B] * (num_examples_per_task - num_choice_a)
    )
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def validate_config_and_tokenizer(config, tokenizer) -> None:
    if config.model_type != "qwen3":
        raise ValueError(
            f"This validated joint SFT recipe supports model_type='qwen3' only, got {config.model_type!r}"
        )
    if not config.tie_word_embeddings:
        raise ValueError("The joint Qwen3 SFT config must have tie_word_embeddings=True")
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


def _example_by_target(dataset: Dataset, task: str, completion: str) -> dict:
    return next(
        example
        for example in dataset
        if example["task"] == task and example["completion"][0]["content"] == completion
    )


def validate_chat_alignment(tokenizer, dataset: Dataset) -> dict[str, int]:
    targets = {
        "eos": _example_by_target(dataset, TASK_EOS, ""),
        "ok": _example_by_target(dataset, TASK_EOS, EOS_ALTERNATIVE),
        "a": _example_by_target(dataset, TASK_BANDIT, CHOICE_A),
        "b": _example_by_target(dataset, TASK_BANDIT, CHOICE_B),
    }
    formatted = {name: apply_chat_template(example, tokenizer) for name, example in targets.items()}
    expected_prompts = {
        TASK_EOS: tokenizer.apply_chat_template(
            targets["eos"]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        TASK_BANDIT: tokenizer.apply_chat_template(
            targets["a"]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    }
    if (
        formatted["eos"]["prompt"] != expected_prompts[TASK_EOS]
        or formatted["ok"]["prompt"] != expected_prompts[TASK_EOS]
    ):
        raise RuntimeError("EOS SFT prompt serialization does not match the Relax enable_thinking=false prompt")
    if (
        formatted["a"]["prompt"] != expected_prompts[TASK_BANDIT]
        or formatted["b"]["prompt"] != expected_prompts[TASK_BANDIT]
    ):
        raise RuntimeError("Bandit SFT prompt serialization does not match the Relax enable_thinking=false prompt")

    target_ids = {
        name: tokenizer(text, add_special_tokens=False)["input_ids"]
        for name, text in {"ok": EOS_ALTERNATIVE, "a": CHOICE_A, "b": CHOICE_B}.items()
    }
    if any(len(token_ids) != 1 for token_ids in target_ids.values()):
        raise RuntimeError(f"OK, A, and B must each tokenize to exactly one token, got {target_ids}")
    if len({token_ids[0] for token_ids in target_ids.values()}) != len(target_ids):
        raise RuntimeError(f"OK, A, and B must have distinct token IDs, got {target_ids}")
    if any(token_ids[0] in {tokenizer.eos_token_id, tokenizer.pad_token_id} for token_ids in target_ids.values()):
        raise RuntimeError(f"OK, A, and B cannot use EOS or padding token IDs: {target_ids}")

    completion_ids = {
        name: tokenizer(item["completion"], add_special_tokens=False)["input_ids"]
        for name, item in formatted.items()
    }
    if not completion_ids["eos"] or completion_ids["eos"][0] != tokenizer.eos_token_id:
        raise RuntimeError(f"Immediate-EOS completion does not start with EOS: {completion_ids['eos']}")
    for name in ("ok", "a", "b"):
        expected_prefix = [target_ids[name][0], tokenizer.eos_token_id]
        if completion_ids[name][:2] != expected_prefix:
            raise RuntimeError(
                f"Formatted {name.upper()} completion must begin with its target then EOS; "
                f"expected {expected_prefix}, got {completion_ids[name]}"
            )

    prompt_ids = {
        task: tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for task, prompt in expected_prompts.items()
    }
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    rows = []
    for name, task in (("eos", TASK_EOS), ("ok", TASK_EOS), ("a", TASK_BANDIT), ("b", TASK_BANDIT)):
        rows.append(
            {
                "input_ids": prompt_ids[task] + completion_ids[name],
                "completion_mask": [0] * len(prompt_ids[task]) + [1] * len(completion_ids[name]),
            }
        )
    labels = collator(rows)["labels"]
    for row_index, (name, task) in enumerate(
        (("eos", TASK_EOS), ("ok", TASK_EOS), ("a", TASK_BANDIT), ("b", TASK_BANDIT))
    ):
        prompt_length = len(prompt_ids[task])
        if not torch.all(labels[row_index, :prompt_length] == -100):
            raise RuntimeError(f"Completion-only loss did not mask every {task} prompt token")
        expected_first_token = tokenizer.eos_token_id if name == "eos" else target_ids[name][0]
        if labels[row_index, prompt_length].item() != expected_first_token:
            raise RuntimeError(f"The first supervised token for {name} is incorrect")
        if name != "eos" and labels[row_index, prompt_length + 1].item() != tokenizer.eos_token_id:
            raise RuntimeError(f"The {name} target is not supervised to emit EOS afterward")

    return {
        "eos_prompt_tokens": len(prompt_ids[TASK_EOS]),
        "bandit_prompt_tokens": len(prompt_ids[TASK_BANDIT]),
        "ok_token_id": target_ids["ok"][0],
        "choice_a_token_id": target_ids["a"][0],
        "choice_b_token_id": target_ids["b"][0],
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
    dataset = build_joint_sft_dataset(
        num_examples_per_task=args.num_examples_per_task,
        eos_fraction=args.eos_fraction,
        choice_a_fraction=args.choice_a_fraction,
        seed=args.seed,
    )
    alignment = validate_chat_alignment(tokenizer, dataset)
    task_counts = {task: sum(example["task"] == task for example in dataset) for task in (TASK_EOS, TASK_BANDIT)}
    target_counts = {
        target: sum(example["completion"][0]["content"] == target for example in dataset)
        for target in ("", EOS_ALTERNATIVE, CHOICE_A, CHOICE_B)
    }
    print(
        "Joint SFT preflight passed: "
        f"initialization=random_from_config, model_type={config.model_type}, "
        f"tie_word_embeddings={config.tie_word_embeddings}, total_examples={len(dataset)}, "
        f"eos_task={task_counts[TASK_EOS]}, bandit_task={task_counts[TASK_BANDIT]}, "
        f"targets=EOS:{target_counts['']}/OK:{target_counts[EOS_ALTERNATIVE]}/"
        f"A:{target_counts[CHOICE_A]}/B:{target_counts[CHOICE_B]}, "
        f"prompt_tokens={alignment['eos_prompt_tokens']}/{alignment['bandit_prompt_tokens']}, "
        f"target_token_ids={alignment['ok_token_id']}/{alignment['choice_a_token_id']}/"
        f"{alignment['choice_b_token_id']}"
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
