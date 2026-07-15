import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.eos_two_action_bandit.train_sft import (
    BANDIT_SYSTEM_PROMPT,
    BANDIT_USER_PROMPT,
    CHOICE_A,
    CHOICE_B,
    EOS_ALTERNATIVE,
    EOS_SYSTEM_PROMPT,
    EOS_USER_PROMPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate both tasks in a joint EOS and A/B SFT checkpoint.")
    parser.add_argument("model")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--min-target-rate", type=float, default=0.2)
    parser.add_argument("--max-target-rate", type=float, default=0.8)
    parser.add_argument("--min-mixed-group-rate", type=float, default=0.5)
    parser.add_argument("--min-target-mass", type=float, default=0.9)
    parser.add_argument("--min-eos-after-target", type=float, default=0.9)
    parser.add_argument("--min-valid-response-rate", type=float, default=0.95)
    parser.add_argument("--max-truncated-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _encode_prompt(tokenizer, system_prompt: str, user_prompt: str, device: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)


def _generate(
    model,
    tokenizer,
    encoded,
    *,
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
) -> list[list[int]]:
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_length = input_ids.shape[1]
    torch.manual_seed(seed)
    generated_rows = []
    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start)
            outputs = model.generate(
                input_ids=input_ids.repeat(current_batch_size, 1),
                attention_mask=attention_mask.repeat(current_batch_size, 1),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_rows.extend(outputs[:, prompt_length:].cpu().tolist())
    return generated_rows


def _first_token_probabilities(model, encoded) -> torch.Tensor:
    with torch.inference_mode():
        return torch.softmax(model(**encoded).logits[0, -1].float(), dim=-1)


def _eos_probability_after_token(model, encoded, token_id: int, eos_token_id: int) -> float:
    token = torch.tensor([[token_id]], device=encoded["input_ids"].device)
    input_ids = torch.cat((encoded["input_ids"], token), dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)[eos_token_id].item()


def _evaluate_eos(model, tokenizer, args: argparse.Namespace) -> tuple[dict, list[str]]:
    encoded = _encode_prompt(tokenizer, EOS_SYSTEM_PROMPT, EOS_USER_PROMPT, args.device)
    alternative_ids = tokenizer(EOS_ALTERNATIVE, add_special_tokens=False)["input_ids"]
    if len(alternative_ids) != 1:
        raise RuntimeError(f"{EOS_ALTERNATIVE!r} must tokenize to exactly one token, got {alternative_ids}")

    first_token_probs = _first_token_probabilities(model, encoded)
    eos_probability = first_token_probs[tokenizer.eos_token_id].item()
    alternative_probability = first_token_probs[alternative_ids[0]].item()
    top_two_ids = torch.topk(first_token_probs, k=2).indices.tolist()
    eos_after_alternative = _eos_probability_after_token(
        model, encoded, alternative_ids[0], tokenizer.eos_token_id
    )
    generated_rows = _generate(
        model,
        tokenizer,
        encoded,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=4,
        seed=args.seed,
    )

    response_lengths = []
    immediate_eos = 0
    alternative_then_eos = 0
    truncated = 0
    for token_ids in generated_rows:
        if token_ids[0] == tokenizer.eos_token_id:
            immediate_eos += 1
        if token_ids[:1] == alternative_ids and len(token_ids) > 1 and token_ids[1] == tokenizer.eos_token_id:
            alternative_then_eos += 1
        try:
            response_length = token_ids.index(tokenizer.eos_token_id) + 1
        except ValueError:
            response_length = len(token_ids)
            truncated += 1
        response_lengths.append(response_length)

    groups = [
        response_lengths[index : index + args.group_size]
        for index in range(0, args.num_samples, args.group_size)
    ]
    mixed_groups = sum(len(set(group)) > 1 for group in groups)
    immediate_eos_rate = immediate_eos / args.num_samples
    valid_response_rate = (immediate_eos + alternative_then_eos) / args.num_samples
    mixed_group_rate = mixed_groups / len(groups)
    truncated_rate = truncated / args.num_samples
    target_mass = eos_probability + alternative_probability
    summary = {
        "first_token_alternative_probability": alternative_probability,
        "first_token_eos_probability": eos_probability,
        "first_token_target_mass": target_mass,
        "first_token_top_two_ids": top_two_ids,
        "eos_after_alternative_probability": eos_after_alternative,
        "immediate_eos": immediate_eos,
        "immediate_eos_rate": immediate_eos_rate,
        "alternative_then_eos": alternative_then_eos,
        "mixed_groups": mixed_groups,
        "mixed_group_rate": mixed_group_rate,
        "truncated": truncated,
        "truncated_rate": truncated_rate,
        "valid_response_rate": valid_response_rate,
        "response_length_counts": {
            str(length): response_lengths.count(length) for length in sorted(set(response_lengths))
        },
    }

    errors = []
    if set(top_two_ids) != {tokenizer.eos_token_id, alternative_ids[0]}:
        errors.append(f"top-two first tokens {top_two_ids} are not EOS/OK")
    if target_mass < args.min_target_mass:
        errors.append(f"EOS/OK probability mass {target_mass:.3f} is below {args.min_target_mass}")
    if eos_after_alternative < args.min_eos_after_target:
        errors.append(
            f"p(EOS after OK)={eos_after_alternative:.3f} is below {args.min_eos_after_target}"
        )
    if not args.min_target_rate <= immediate_eos_rate <= args.max_target_rate:
        errors.append(
            f"immediate EOS rate {immediate_eos_rate:.3f} is outside "
            f"[{args.min_target_rate}, {args.max_target_rate}]"
        )
    if mixed_group_rate < args.min_mixed_group_rate:
        errors.append(f"mixed-group rate {mixed_group_rate:.3f} is below {args.min_mixed_group_rate}")
    if valid_response_rate < args.min_valid_response_rate:
        errors.append(
            f"valid EOS/OK response rate {valid_response_rate:.3f} is below {args.min_valid_response_rate}"
        )
    if truncated_rate > args.max_truncated_rate:
        errors.append(f"truncated rate {truncated_rate:.3f} is above {args.max_truncated_rate}")
    return summary, errors


def _evaluate_bandit(model, tokenizer, args: argparse.Namespace) -> tuple[dict, list[str]]:
    encoded = _encode_prompt(tokenizer, BANDIT_SYSTEM_PROMPT, BANDIT_USER_PROMPT, args.device)
    choice_ids = {
        choice: tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in (CHOICE_A, CHOICE_B)
    }
    if any(len(token_ids) != 1 for token_ids in choice_ids.values()):
        raise RuntimeError(f"A and B must each tokenize to exactly one token, got {choice_ids}")

    first_token_probs = _first_token_probabilities(model, encoded)
    choice_probabilities = {
        choice: first_token_probs[token_ids[0]].item() for choice, token_ids in choice_ids.items()
    }
    top_two_ids = torch.topk(first_token_probs, k=2).indices.tolist()
    eos_after_choice = {
        choice: _eos_probability_after_token(model, encoded, token_ids[0], tokenizer.eos_token_id)
        for choice, token_ids in choice_ids.items()
    }
    generated_rows = _generate(
        model,
        tokenizer,
        encoded,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=3,
        seed=args.seed + 1,
    )

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
        "sampled_choice_counts": choice_counts,
        "sampled_choice_rates": choice_rates,
        "valid_response_rate": valid_response_rate,
        "mixed_groups": mixed_groups,
        "mixed_group_rate": mixed_group_rate,
        "truncated": truncated,
        "truncated_rate": truncated_rate,
    }

    errors = []
    expected_top_two_ids = {token_ids[0] for token_ids in choice_ids.values()}
    if set(top_two_ids) != expected_top_two_ids:
        errors.append(f"top-two first tokens {top_two_ids} are not A/B {sorted(expected_top_two_ids)}")
    if target_mass < args.min_target_mass:
        errors.append(f"A/B probability mass {target_mass:.3f} is below {args.min_target_mass}")
    for choice in (CHOICE_A, CHOICE_B):
        if eos_after_choice[choice] < args.min_eos_after_target:
            errors.append(
                f"p(EOS after {choice})={eos_after_choice[choice]:.3f} is below {args.min_eos_after_target}"
            )
        if not args.min_target_rate <= choice_rates[choice] <= args.max_target_rate:
            errors.append(
                f"sampled {choice} rate {choice_rates[choice]:.3f} is outside "
                f"[{args.min_target_rate}, {args.max_target_rate}]"
            )
    if mixed_group_rate < args.min_mixed_group_rate:
        errors.append(f"mixed-group rate {mixed_group_rate:.3f} is below {args.min_mixed_group_rate}")
    if valid_response_rate < args.min_valid_response_rate:
        errors.append(
            f"valid A/B response rate {valid_response_rate:.3f} is below {args.min_valid_response_rate}"
        )
    if truncated_rate > args.max_truncated_rate:
        errors.append(f"truncated rate {truncated_rate:.3f} is above {args.max_truncated_rate}")
    return summary, errors


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.num_samples % args.group_size != 0:
        raise ValueError("num_samples must be positive and divisible by group_size")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    eos_summary, eos_errors = _evaluate_eos(model, tokenizer, args)
    bandit_summary, bandit_errors = _evaluate_bandit(model, tokenizer, args)
    summary = {
        "model": args.model,
        "num_samples_per_task": args.num_samples,
        "group_size": args.group_size,
        "eos": eos_summary,
        "two_action_bandit": bandit_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    errors = [f"EOS: {error}" for error in eos_errors]
    errors.extend(f"bandit: {error}" for error in bandit_errors)
    if errors:
        raise SystemExit("Joint SFT acceptance failed: " + "; ".join(errors))
    print("Joint SFT acceptance passed: both task prompts are suitable for mixed Relax GRPO.")


if __name__ == "__main__":
    main()
