import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.two_action_bandit.train_sft import (
    CHOICE_A,
    CHOICE_B,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a balanced A/B SFT checkpoint before Relax training.")
    parser.add_argument("model")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--min-choice-rate", type=float, default=0.2)
    parser.add_argument("--max-choice-rate", type=float, default=0.8)
    parser.add_argument("--min-mixed-group-rate", type=float, default=0.5)
    parser.add_argument("--min-target-mass", type=float, default=0.9)
    parser.add_argument("--min-eos-after-choice", type=float, default=0.9)
    parser.add_argument("--min-valid-response-rate", type=float, default=0.95)
    parser.add_argument("--max-truncated-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.num_samples % args.group_size != 0:
        raise ValueError("num_samples must be positive and divisible by group_size")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    choice_ids = {
        choice: tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in (CHOICE_A, CHOICE_B)
    }
    if any(len(token_ids) != 1 for token_ids in choice_ids.values()):
        raise RuntimeError(f"A and B must each tokenize to one token, got {choice_ids}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": DEFAULT_USER_PROMPT},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(args.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        first_token_logits = model(**encoded).logits[0, -1].float()
        first_token_probs = torch.softmax(first_token_logits, dim=-1)
        choice_probabilities = {
            choice: first_token_probs[token_ids[0]].item() for choice, token_ids in choice_ids.items()
        }
        top_two_ids = torch.topk(first_token_probs, k=2).indices.tolist()

        eos_after_choice = {}
        for choice, token_ids in choice_ids.items():
            choice_input_ids = torch.cat(
                (input_ids, torch.tensor([token_ids], device=args.device)),
                dim=1,
            )
            choice_attention_mask = torch.ones_like(choice_input_ids)
            logits = model(
                input_ids=choice_input_ids,
                attention_mask=choice_attention_mask,
            ).logits[0, -1].float()
            eos_after_choice[choice] = torch.softmax(logits, dim=-1)[tokenizer.eos_token_id].item()

    torch.manual_seed(args.seed)
    generated_rows = []
    with torch.inference_mode():
        for start in range(0, args.num_samples, args.batch_size):
            current_batch_size = min(args.batch_size, args.num_samples - start)
            outputs = model.generate(
                input_ids=input_ids.repeat(current_batch_size, 1),
                attention_mask=attention_mask.repeat(current_batch_size, 1),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_rows.extend(outputs[:, prompt_length:].cpu().tolist())

    sampled_choices = []
    truncated = 0
    for token_ids in generated_rows:
        if tokenizer.eos_token_id not in token_ids:
            truncated += 1
        if len(token_ids) >= 2 and token_ids[1] == tokenizer.eos_token_id:
            if token_ids[0] == choice_ids[CHOICE_A][0]:
                sampled_choices.append(CHOICE_A)
                continue
            if token_ids[0] == choice_ids[CHOICE_B][0]:
                sampled_choices.append(CHOICE_B)
                continue
        sampled_choices.append("invalid")

    groups = [
        sampled_choices[index : index + args.group_size]
        for index in range(0, args.num_samples, args.group_size)
    ]
    choice_counts = {choice: sampled_choices.count(choice) for choice in (CHOICE_A, CHOICE_B)}
    choice_rates = {choice: count / args.num_samples for choice, count in choice_counts.items()}
    mixed_groups = sum(CHOICE_A in group and CHOICE_B in group for group in groups)
    valid_response_rate = sum(choice_counts.values()) / args.num_samples
    mixed_group_rate = mixed_groups / len(groups)
    truncated_rate = truncated / args.num_samples
    target_mass = sum(choice_probabilities.values())
    summary = {
        "choice_token_ids": {choice: token_ids[0] for choice, token_ids in choice_ids.items()},
        "first_token_choice_probabilities": choice_probabilities,
        "first_token_target_mass": target_mass,
        "first_token_top_two_ids": top_two_ids,
        "eos_after_choice_probabilities": eos_after_choice,
        "num_samples": args.num_samples,
        "sampled_choice_counts": choice_counts,
        "sampled_choice_rates": choice_rates,
        "valid_response_rate": valid_response_rate,
        "mixed_groups": mixed_groups,
        "mixed_group_rate": mixed_group_rate,
        "truncated": truncated,
        "truncated_rate": truncated_rate,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    errors = []
    expected_top_two_ids = {token_ids[0] for token_ids in choice_ids.values()}
    if set(top_two_ids) != expected_top_two_ids:
        errors.append(f"top-two first tokens {top_two_ids} are not A/B {sorted(expected_top_two_ids)}")
    if target_mass < args.min_target_mass:
        errors.append(f"A/B probability mass {target_mass:.3f} is below {args.min_target_mass}")
    for choice in (CHOICE_A, CHOICE_B):
        if eos_after_choice[choice] < args.min_eos_after_choice:
            errors.append(
                f"p(EOS after {choice})={eos_after_choice[choice]:.3f} is below {args.min_eos_after_choice}"
            )
        if not args.min_choice_rate <= choice_rates[choice] <= args.max_choice_rate:
            errors.append(
                f"sampled {choice} rate {choice_rates[choice]:.3f} is outside "
                f"[{args.min_choice_rate}, {args.max_choice_rate}]"
            )
    if mixed_group_rate < args.min_mixed_group_rate:
        errors.append(f"mixed-group rate {mixed_group_rate:.3f} is below {args.min_mixed_group_rate}")
    if valid_response_rate < args.min_valid_response_rate:
        errors.append(
            f"valid A/B response rate {valid_response_rate:.3f} is below {args.min_valid_response_rate}"
        )
    if truncated_rate > args.max_truncated_rate:
        errors.append(f"truncated rate {truncated_rate:.3f} is above {args.max_truncated_rate}")
    if errors:
        raise SystemExit("Bandit SFT acceptance failed: " + "; ".join(errors))

    print("Bandit SFT acceptance passed: this checkpoint is suitable for the Relax learning run.")


if __name__ == "__main__":
    main()
