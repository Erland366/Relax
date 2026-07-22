import copy
import hashlib
import io
import json
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


IMAGE_SIZE = 256
SFT_SYSTEM_PROMPT = "Answer with exactly one uppercase character: A or B. Do not explain."
SFT_USER_PROMPT = "Inspect the left glyph. Reply A for vertical bars or B for horizontal bars."
XOR_USER_PROMPT = (
    "Inspect both glyphs. Reply A if they have the same orientation or B if they have different orientations."
)

_COMBINATIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
_PALETTES = (
    ((245, 247, 250), (26, 35, 50)),
    ((25, 31, 42), (241, 245, 249)),
    ((250, 244, 232), (38, 61, 77)),
    ((232, 246, 242), (49, 46, 129)),
    ((245, 236, 248), (22, 76, 79)),
)


def _validate_bit(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def _sample_id(split: str, render_seed: int) -> str:
    digest = hashlib.sha256(f"visual-xor:{split}:{render_seed}".encode()).hexdigest()
    return digest[:24]


def _draw_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    bit: int,
    center_x: int,
    center_y: int,
    extent: int,
    thickness: int,
    gap: int,
    color: tuple[int, int, int],
) -> None:
    half_extent = extent // 2
    half_thickness = thickness // 2
    if bit == 0:
        for offset in (-gap, gap):
            draw.rounded_rectangle(
                (
                    center_x + offset - half_thickness,
                    center_y - half_extent,
                    center_x + offset + half_thickness,
                    center_y + half_extent,
                ),
                radius=max(2, thickness // 4),
                fill=color,
            )
    else:
        for offset in (-gap, gap):
            draw.rounded_rectangle(
                (
                    center_x - half_extent,
                    center_y + offset - half_thickness,
                    center_x + half_extent,
                    center_y + offset + half_thickness,
                ),
                radius=max(2, thickness // 4),
                fill=color,
            )


def render_visual_xor_png(*, seed: int, left_bit: int, right_bit: int) -> bytes:
    """Render two abstract binary glyphs without embedding labels or metadata."""
    _validate_bit("left_bit", left_bit)
    _validate_bit("right_bit", right_bit)
    rng = random.Random(seed)
    background, foreground = rng.choice(_PALETTES)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background)
    draw = ImageDraw.Draw(image)

    panel_y = 128 + rng.randint(-8, 8)
    panel_centers = (72 + rng.randint(-6, 6), 184 + rng.randint(-6, 6))
    extent = rng.randint(42, 51)
    thickness = rng.randint(13, 18)
    gap = rng.randint(20, 26)
    for bit, center_x in zip((left_bit, right_bit), panel_centers, strict=True):
        _draw_glyph(
            draw,
            bit=bit,
            center_x=center_x,
            center_y=panel_y + rng.randint(-5, 5),
            extent=extent + rng.randint(-3, 3),
            thickness=thickness,
            gap=gap,
            color=foreground,
        )

    # Label-independent nuisance marks stay near the border, outside both glyph regions.
    for _ in range(rng.randint(0, 4)):
        x = rng.choice((rng.randint(8, 32), rng.randint(224, 247)))
        y = rng.randint(12, 243)
        radius = rng.randint(1, 3)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=foreground)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _render_seed(base_seed: int, split: str, index: int) -> int:
    split_digest = hashlib.sha256(split.encode()).digest()
    split_offset = int.from_bytes(split_digest[:4], "big")
    return base_seed * 10_000_000 + split_offset + index


def _balanced_semantics(index: int) -> tuple[int, int]:
    return _COMBINATIONS[index % len(_COMBINATIONS)]


def build_sft_rows(
    *,
    num_examples: int,
    seed: int,
    split: str,
    preferred_action_probability: float = 0.7,
) -> list[dict]:
    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}")
    if not 0.0 < preferred_action_probability < 1.0:
        raise ValueError("preferred_action_probability must be strictly between 0 and 1")

    rows = []
    for index in range(num_examples):
        left_bit, right_bit = _balanced_semantics(index)
        render_seed = _render_seed(seed, split, index)
        preferred_action = "A" if left_bit == 0 else "B"
        noise_rng = random.Random(render_seed ^ 0x5F3759DF)
        target = (
            preferred_action
            if noise_rng.random() < preferred_action_probability
            else ("B" if left_bit == 0 else "A")
        )
        rows.append(
            {
                "sample_id": _sample_id(split, render_seed),
                "split": split,
                "render_seed": render_seed,
                "left_bit": left_bit,
                "right_bit": right_bit,
                "prompt": SFT_USER_PROMPT,
                "preferred_action": preferred_action,
                "target": target,
                "image": render_visual_xor_png(seed=render_seed, left_bit=left_bit, right_bit=right_bit),
            }
        )
    random.Random(seed ^ 0xA5A5A5A5).shuffle(rows)
    return rows


def build_xor_rows(*, num_examples: int, seed: int, split: str, shuffle: bool = True) -> list[dict]:
    if num_examples <= 0 or num_examples % len(_COMBINATIONS) != 0:
        raise ValueError(f"num_examples must be a positive multiple of 4, got {num_examples}")

    rows = []
    for index in range(num_examples):
        left_bit, right_bit = _balanced_semantics(index)
        render_seed = _render_seed(seed, split, index)
        combination = f"{left_bit}{right_bit}"
        label = "A" if left_bit == right_bit else "B"
        rows.append(
            {
                "prompt": f"<image>{XOR_USER_PROMPT}",
                "image": [render_visual_xor_png(seed=render_seed, left_bit=left_bit, right_bit=right_bit)],
                "label": label,
                "render_seed": render_seed,
                "metadata": {
                    "sample_id": _sample_id(split, render_seed),
                    "split": split,
                    "render_seed": render_seed,
                    "left_bit": left_bit,
                    "right_bit": right_bit,
                    "combination": combination,
                },
            }
        )
    if shuffle:
        random.Random(seed ^ 0xC3C3C3C3).shuffle(rows)
    return rows


def build_refinement_sft_rows(
    *,
    num_examples: int,
    seed: int,
    split: str,
    correct_action_probability: float = 0.7,
) -> list[dict]:
    """Build exact-context XOR SFT rows with balanced deterministic label noise."""
    if num_examples <= 0 or num_examples % len(_COMBINATIONS) != 0:
        raise ValueError(f"num_examples must be a positive multiple of 4, got {num_examples}")
    if not 0.0 < correct_action_probability < 1.0:
        raise ValueError("correct_action_probability must be strictly between 0 and 1")

    examples_per_combination = num_examples // len(_COMBINATIONS)
    requested_correct = examples_per_combination * correct_action_probability
    correct_per_combination = round(requested_correct)
    if not math.isclose(requested_correct, correct_per_combination, abs_tol=1e-9):
        raise ValueError(
            "num_examples and correct_action_probability must produce an exact integer number of correct targets "
            f"per combination, got {requested_correct}"
        )
    if correct_per_combination <= 0 or correct_per_combination >= examples_per_combination:
        raise ValueError(
            "correct_action_probability must leave both correct and flipped targets within every combination; "
            f"got {correct_per_combination}/{examples_per_combination} correct targets"
        )

    correctness_by_combination = []
    for combination_index in range(len(_COMBINATIONS)):
        correctness = [True] * correct_per_combination + [False] * (
            examples_per_combination - correct_per_combination
        )
        random.Random(seed ^ (0x9E3779B9 * (combination_index + 1))).shuffle(correctness)
        correctness_by_combination.append(correctness)

    rows = []
    for index in range(num_examples):
        combination_index = index % len(_COMBINATIONS)
        occurrence_index = index // len(_COMBINATIONS)
        left_bit, right_bit = _COMBINATIONS[combination_index]
        render_seed = _render_seed(seed, split, index)
        combination = f"{left_bit}{right_bit}"
        preferred_action = "A" if left_bit == right_bit else "B"
        target_is_correct = correctness_by_combination[combination_index][occurrence_index]
        target = preferred_action if target_is_correct else ("B" if preferred_action == "A" else "A")
        rows.append(
            {
                "sample_id": _sample_id(split, render_seed),
                "split": split,
                "render_seed": render_seed,
                "left_bit": left_bit,
                "right_bit": right_bit,
                "combination": combination,
                "prompt": XOR_USER_PROMPT,
                "preferred_action": preferred_action,
                "target": target,
                "target_is_correct": target_is_correct,
                "image": render_visual_xor_png(seed=render_seed, left_bit=left_bit, right_bit=right_bit),
            }
        )
    random.Random(seed ^ 0x6A09E667).shuffle(rows)
    return rows


def build_refinement_control_rows(rows: list[dict], *, control: str) -> list[dict]:
    """Replace eval images while retaining private labels and metadata for visual controls."""
    if not rows:
        raise ValueError("rows must not be empty")
    if control not in {"permuted", "constant"}:
        raise ValueError(f"control must be 'permuted' or 'constant', got {control!r}")

    if control == "permuted":
        replacement_images = [rows[(index + 1) % len(rows)]["image"] for index in range(len(rows))]
    else:
        replacement_images = [rows[0]["image"]] * len(rows)

    controlled_rows = []
    for row, replacement_image in zip(rows, replacement_images, strict=True):
        controlled = copy.deepcopy(row)
        controlled["image"] = copy.deepcopy(replacement_image)
        controlled["metadata"]["visual_control"] = control
        controlled_rows.append(controlled)
    return controlled_rows


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_refinement_dataset_bundle(
    output_dir: str | Path,
    *,
    sft_train_examples: int = 8000,
    sft_eval_examples: int = 1000,
    rl_train_examples: int = 64,
    rl_eval_examples: int = 128,
    correct_action_probability: float = 0.7,
    seed: int = 42,
) -> dict[str, Path]:
    """Write the easy visual-XOR refinement SFT, RL, held-out, and control datasets."""
    output_dir = Path(output_dir)
    paths = {
        "sft_train": output_dir / "refinement_sft_train.parquet",
        "sft_eval": output_dir / "refinement_sft_eval.parquet",
        "rl_train": output_dir / "refinement_rl_train.parquet",
        "rl_eval": output_dir / "refinement_rl_eval.parquet",
        "rl_eval_permuted": output_dir / "refinement_rl_eval_permuted.parquet",
        "rl_eval_constant": output_dir / "refinement_rl_eval_constant.parquet",
        "eval_config": output_dir / "refinement_eval_config.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing visual XOR refinement data: {existing}")

    sft_train_rows = build_refinement_sft_rows(
        num_examples=sft_train_examples,
        seed=seed,
        split="refinement_sft_train",
        correct_action_probability=correct_action_probability,
    )
    sft_eval_rows = build_refinement_sft_rows(
        num_examples=sft_eval_examples,
        seed=seed,
        split="refinement_sft_eval",
        correct_action_probability=correct_action_probability,
    )
    rl_train_rows = build_xor_rows(
        num_examples=rl_train_examples,
        seed=seed,
        split="refinement_rl_train",
        shuffle=False,
    )
    rl_eval_rows = build_xor_rows(
        num_examples=rl_eval_examples,
        seed=seed,
        split="refinement_rl_eval",
        shuffle=False,
    )
    rl_eval_permuted_rows = build_refinement_control_rows(rl_eval_rows, control="permuted")
    rl_eval_constant_rows = build_refinement_control_rows(rl_eval_rows, control="constant")

    _write_parquet(paths["sft_train"], sft_train_rows)
    _write_parquet(paths["sft_eval"], sft_eval_rows)
    _write_parquet(paths["rl_train"], rl_train_rows)
    _write_parquet(paths["rl_eval"], rl_eval_rows)
    _write_parquet(paths["rl_eval_permuted"], rl_eval_permuted_rows)
    _write_parquet(paths["rl_eval_constant"], rl_eval_constant_rows)

    eval_defaults = {
        "input_key": "prompt",
        "label_key": "label",
        "metadata_key": "metadata",
        "top_p": 1.0,
        "top_k": -1,
        "max_response_len": 3,
    }
    eval_datasets = [
        {
            "name": "visual_xor_heldout",
            "path": str(paths["rl_eval"].resolve()),
            "rm_type": "visual_xor",
            "n_samples_per_eval_prompt": 4,
            "temperature": 1.0,
        },
        {
            "name": "visual_xor_permuted_control",
            "path": str(paths["rl_eval_permuted"].resolve()),
            "rm_type": "visual_xor",
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.0,
        },
        {
            "name": "visual_xor_constant_control",
            "path": str(paths["rl_eval_constant"].resolve()),
            "rm_type": "visual_xor",
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.0,
        },
    ]
    _write_json(paths["eval_config"], {"eval": {"defaults": eval_defaults, "datasets": eval_datasets}})
    return paths


def write_dataset_bundle(
    output_dir: str | Path,
    *,
    sft_train_examples: int = 8192,
    sft_eval_examples: int = 1024,
    xor_train_examples: int = 4096,
    xor_eval_examples: int = 1024,
    preferred_action_probability: float = 0.7,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "sft_train": output_dir / "sft_train.parquet",
        "sft_eval": output_dir / "sft_eval.parquet",
        "xor_train": output_dir / "xor_train.parquet",
        "xor_eval": output_dir / "xor_eval.parquet",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing visual XOR data: {existing}")

    _write_parquet(
        paths["sft_train"],
        build_sft_rows(
            num_examples=sft_train_examples,
            seed=seed,
            split="sft_train",
            preferred_action_probability=preferred_action_probability,
        ),
    )
    _write_parquet(
        paths["sft_eval"],
        build_sft_rows(
            num_examples=sft_eval_examples,
            seed=seed,
            split="sft_eval",
            preferred_action_probability=preferred_action_probability,
        ),
    )
    _write_parquet(paths["xor_train"], build_xor_rows(num_examples=xor_train_examples, seed=seed, split="xor_train"))
    _write_parquet(paths["xor_eval"], build_xor_rows(num_examples=xor_eval_examples, seed=seed, split="xor_eval"))
    return paths
