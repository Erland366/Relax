import argparse
import io
import json
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from examples.visual_xor.task import (
    SFT_SYSTEM_PROMPT,
    SFT_USER_PROMPT,
    XOR_USER_PROMPT,
    render_visual_xor_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the visual-bootstrap acceptance gate to a Qwen3-VL SFT.")
    parser.add_argument("model")
    parser.add_argument("--sft-eval-data", required=True)
    parser.add_argument("--xor-eval-data", required=True)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--counterfactual-images", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _decode_image(value) -> Image.Image:
    if isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) == 1:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("bytes")
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB").copy()


def _messages(prompt: str, image: Image.Image, *, include_system: bool) -> list[dict]:
    messages = []
    if include_system:
        messages.append({"role": "system", "content": [{"type": "text", "text": SFT_SYSTEM_PROMPT}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def _encode(processor, images: list[Image.Image], prompt: str, device: str, *, include_system: bool):
    texts = [
        processor.apply_chat_template(
            _messages(prompt, image, include_system=include_system),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for image in images
    ]
    return processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)


def _sample_group(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    group_size: int,
    device: str,
    include_system: bool,
) -> tuple[list[str], int]:
    encoded = _encode(processor, [image] * group_size, prompt, device, include_system=include_system)
    prompt_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=3,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    action_ids = {
        action: processor.tokenizer(action, add_special_tokens=False)["input_ids"][0] for action in ("A", "B")
    }
    sampled = []
    truncated = 0
    for token_ids in outputs[:, prompt_length:].cpu().tolist():
        if processor.tokenizer.eos_token_id not in token_ids:
            truncated += 1
        if len(token_ids) >= 2 and token_ids[1] == processor.tokenizer.eos_token_id:
            if token_ids[0] == action_ids["A"]:
                sampled.append("A")
                continue
            if token_ids[0] == action_ids["B"]:
                sampled.append("B")
                continue
        sampled.append("invalid")
    return sampled, truncated


def _action_a_probability(model, processor, image: Image.Image, prompt: str, device: str) -> float:
    encoded = _encode(processor, [image], prompt, device, include_system=True)
    action_a_id = processor.tokenizer("A", add_special_tokens=False)["input_ids"][0]
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)[action_a_id].item()


def _evaluate_rows(
    model,
    processor,
    frame: pd.DataFrame,
    *,
    prompt: str,
    target_key: str,
    num_images: int,
    group_size: int,
    device: str,
    include_system: bool,
) -> dict:
    frame = frame.iloc[: min(num_images, len(frame))]
    counts = Counter()
    mixed_groups = 0
    correct = 0
    truncated = 0
    total = 0
    combination_correct = Counter()
    combination_total = Counter()
    for row in frame.to_dict(orient="records"):
        actions, group_truncated = _sample_group(
            model,
            processor,
            _decode_image(row["image"]),
            prompt,
            group_size=group_size,
            device=device,
            include_system=include_system,
        )
        target = row[target_key]
        counts.update(actions)
        correct += sum(action == target for action in actions)
        mixed_groups += int("A" in actions and "B" in actions)
        truncated += group_truncated
        total += len(actions)
        combination = row.get("combination")
        if combination is None and isinstance(row.get("metadata"), dict):
            combination = row["metadata"].get("combination")
        if combination is not None:
            combination_correct[combination] += sum(action == target for action in actions)
            combination_total[combination] += len(actions)
    summary = {
        "num_images": len(frame),
        "num_samples": total,
        "action_counts": dict(counts),
        "action_a_rate": counts["A"] / total,
        "action_b_rate": counts["B"] / total,
        "valid_response_rate": (counts["A"] + counts["B"]) / total,
        "target_accuracy": correct / total,
        "mixed_group_rate": mixed_groups / len(frame),
        "truncated_rate": truncated / total,
    }
    if combination_total:
        summary["combination_accuracy"] = {
            combination: combination_correct[combination] / combination_total[combination]
            for combination in sorted(combination_total)
        }
    return summary


def evaluate(model, processor, args: argparse.Namespace) -> tuple[dict, list[str]]:
    sft_frame = pd.read_parquet(args.sft_eval_data)
    xor_frame = pd.read_parquet(args.xor_eval_data)
    sft_summary = _evaluate_rows(
        model,
        processor,
        sft_frame,
        prompt=SFT_USER_PROMPT,
        target_key="preferred_action",
        num_images=args.num_images,
        group_size=args.group_size,
        device=args.device,
        include_system=True,
    )
    xor_summary = _evaluate_rows(
        model,
        processor,
        xor_frame,
        prompt=XOR_USER_PROMPT,
        target_key="label",
        num_images=args.num_images,
        group_size=args.group_size,
        device=args.device,
        include_system=True,
    )

    counterfactual_shifts = []
    for row in sft_frame.iloc[: min(args.counterfactual_images, len(sft_frame))].to_dict(orient="records"):
        original = _decode_image(row["image"])
        flipped = Image.open(
            io.BytesIO(
                render_visual_xor_png(
                    seed=int(row["render_seed"]),
                    left_bit=1 - int(row["left_bit"]),
                    right_bit=int(row["right_bit"]),
                )
            )
        ).convert("RGB")
        original_probability = _action_a_probability(model, processor, original, SFT_USER_PROMPT, args.device)
        flipped_probability = _action_a_probability(model, processor, flipped, SFT_USER_PROMPT, args.device)
        counterfactual_shifts.append(abs(original_probability - flipped_probability))
    mean_shift = sum(counterfactual_shifts) / len(counterfactual_shifts)

    summary = {
        "sft_pretask": sft_summary,
        "xor_baseline": xor_summary,
        "mean_left_glyph_counterfactual_probability_shift": mean_shift,
    }
    errors = []
    if not 0.60 <= sft_summary["target_accuracy"] <= 0.85:
        errors.append(f"SFT preferred-action rate {sft_summary['target_accuracy']:.3f} is outside [0.60, 0.85]")
    if not 0.35 <= xor_summary["target_accuracy"] <= 0.65:
        errors.append(f"XOR accuracy {xor_summary['target_accuracy']:.3f} is outside [0.35, 0.65]")
    if xor_summary["valid_response_rate"] < 0.95:
        errors.append(f"valid A/B+EOS rate {xor_summary['valid_response_rate']:.3f} is below 0.95")
    for action in ("a", "b"):
        rate = xor_summary[f"action_{action}_rate"]
        if not 0.20 <= rate <= 0.80:
            errors.append(f"action {action.upper()} rate {rate:.3f} is outside [0.20, 0.80]")
    if xor_summary["mixed_group_rate"] < 0.90:
        errors.append(f"mixed XOR group rate {xor_summary['mixed_group_rate']:.3f} is below 0.90")
    if xor_summary["truncated_rate"] > 0.05:
        errors.append(f"truncation rate {xor_summary['truncated_rate']:.3f} is above 0.05")
    if mean_shift < 0.25:
        errors.append(f"left-glyph counterfactual probability shift {mean_shift:.3f} is below 0.25")
    return summary, errors


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.group_size <= 1 or args.counterfactual_images <= 0:
        raise ValueError("num-images and counterfactual-images must be positive; group-size must exceed one")
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(args.device)
    model.eval()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("Reloaded visual SFT checkpoint does not preserve tied embeddings")

    summary, errors = evaluate(model, processor, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("Visual SFT acceptance failed:\n- " + "\n- ".join(errors))
    print("Visual SFT acceptance passed")


if __name__ == "__main__":
    main()
