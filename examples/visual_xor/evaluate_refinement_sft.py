import argparse
import io
import json

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from examples.visual_xor.evaluate_sft import _action_a_probability, _decode_image, _evaluate_rows
from examples.visual_xor.task import XOR_USER_PROMPT, render_visual_xor_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the easy visual-XOR refinement SFT acceptance gate.")
    parser.add_argument("model")
    parser.add_argument("--xor-eval-data", required=True)
    parser.add_argument("--permuted-eval-data", required=True)
    parser.add_argument("--constant-eval-data", required=True)
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--counterfactual-images", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _render_counterfactual(row: dict, *, flip_left: bool, flip_right: bool) -> Image.Image:
    return Image.open(
        io.BytesIO(
            render_visual_xor_png(
                seed=int(row["render_seed"]),
                left_bit=int(row["metadata"]["left_bit"]) ^ int(flip_left),
                right_bit=int(row["metadata"]["right_bit"]) ^ int(flip_right),
            )
        )
    ).convert("RGB")


def _counterfactual_summary(model, processor, frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    left_directional_shifts = []
    right_directional_shifts = []
    both_absolute_shifts = []
    rows = frame.iloc[: min(args.counterfactual_images, len(frame))].to_dict(orient="records")
    for row in rows:
        original_probability = _action_a_probability(
            model,
            processor,
            _decode_image(row["image"]),
            XOR_USER_PROMPT,
            args.device,
        )
        left_probability = _action_a_probability(
            model,
            processor,
            _render_counterfactual(row, flip_left=True, flip_right=False),
            XOR_USER_PROMPT,
            args.device,
        )
        right_probability = _action_a_probability(
            model,
            processor,
            _render_counterfactual(row, flip_left=False, flip_right=True),
            XOR_USER_PROMPT,
            args.device,
        )
        both_probability = _action_a_probability(
            model,
            processor,
            _render_counterfactual(row, flip_left=True, flip_right=True),
            XOR_USER_PROMPT,
            args.device,
        )
        direction = 1.0 if row["label"] == "A" else -1.0
        left_directional_shifts.append(direction * (original_probability - left_probability))
        right_directional_shifts.append(direction * (original_probability - right_probability))
        both_absolute_shifts.append(abs(original_probability - both_probability))

    return {
        "mean_left_flip_directional_probability_shift": sum(left_directional_shifts) / len(rows),
        "mean_right_flip_directional_probability_shift": sum(right_directional_shifts) / len(rows),
        "mean_both_flip_absolute_probability_shift": sum(both_absolute_shifts) / len(rows),
    }


def validate_refinement_summary(summary: dict) -> list[str]:
    heldout = summary["xor_heldout"]
    errors = []
    if not 0.60 <= heldout["target_accuracy"] <= 0.80:
        errors.append(f"held-out sampled XOR accuracy {heldout['target_accuracy']:.3f} is outside [0.60, 0.80]")
    if heldout["valid_response_rate"] < 0.99:
        errors.append(f"valid A/B+EOS rate {heldout['valid_response_rate']:.3f} is below 0.99")
    for action in ("a", "b"):
        rate = heldout[f"action_{action}_rate"]
        if not 0.35 <= rate <= 0.65:
            errors.append(f"action {action.upper()} rate {rate:.3f} is outside [0.35, 0.65]")
    if heldout["mixed_group_rate"] < 0.90:
        errors.append(f"mixed held-out group rate {heldout['mixed_group_rate']:.3f} is below 0.90")
    if heldout["truncated_rate"] > 0.01:
        errors.append(f"truncation rate {heldout['truncated_rate']:.3f} is above 0.01")
    for combination in ("00", "01", "10", "11"):
        accuracy = heldout.get("combination_accuracy", {}).get(combination, 0.0)
        if accuracy <= 0.55:
            errors.append(f"combination {combination} accuracy {accuracy:.3f} is not above 0.55")

    for control_name in ("permuted_control", "constant_control"):
        control = summary[control_name]
        if not 0.35 <= control["target_accuracy"] <= 0.65:
            errors.append(
                f"{control_name.replace('_', ' ')} accuracy {control['target_accuracy']:.3f} is outside [0.35, 0.65]"
            )
        if control["valid_response_rate"] < 0.99:
            errors.append(
                f"{control_name.replace('_', ' ')} valid response rate "
                f"{control['valid_response_rate']:.3f} is below 0.99"
            )

    counterfactual = summary["counterfactual"]
    for side in ("left", "right"):
        shift = counterfactual[f"mean_{side}_flip_directional_probability_shift"]
        if shift < 0.15:
            errors.append(f"{side}-flip directional probability shift {shift:.3f} is below 0.15")
    both_shift = counterfactual["mean_both_flip_absolute_probability_shift"]
    if both_shift > 0.20:
        errors.append(f"both-flip absolute probability shift {both_shift:.3f} is above 0.20")
    return errors


def evaluate(model, processor, args: argparse.Namespace) -> tuple[dict, list[str]]:
    frames = {
        "xor_heldout": pd.read_parquet(args.xor_eval_data),
        "permuted_control": pd.read_parquet(args.permuted_eval_data),
        "constant_control": pd.read_parquet(args.constant_eval_data),
    }
    summary = {
        name: _evaluate_rows(
            model,
            processor,
            frame,
            prompt=XOR_USER_PROMPT,
            target_key="label",
            num_images=args.num_images,
            group_size=args.group_size,
            device=args.device,
            include_system=True,
        )
        for name, frame in frames.items()
    }
    summary["counterfactual"] = _counterfactual_summary(model, processor, frames["xor_heldout"], args)
    return summary, validate_refinement_summary(summary)


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
        raise RuntimeError("Reloaded visual XOR refinement SFT checkpoint does not preserve tied embeddings")

    summary, errors = evaluate(model, processor, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("Visual XOR refinement SFT acceptance failed:\n- " + "\n- ".join(errors))
    print("Visual XOR refinement SFT acceptance passed")


if __name__ == "__main__":
    main()
