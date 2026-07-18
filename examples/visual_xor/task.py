import hashlib
import io
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


def build_xor_rows(*, num_examples: int, seed: int, split: str) -> list[dict]:
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
    random.Random(seed ^ 0xC3C3C3C3).shuffle(rows)
    return rows


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


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
