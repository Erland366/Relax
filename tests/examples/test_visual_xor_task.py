import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from examples.visual_xor.task import (
    SFT_USER_PROMPT,
    XOR_USER_PROMPT,
    build_sft_rows,
    build_xor_rows,
    render_visual_xor_png,
    write_dataset_bundle,
)


def test_render_visual_xor_png_is_deterministic_and_decodable():
    first = render_visual_xor_png(seed=17, left_bit=0, right_bit=1)
    second = render_visual_xor_png(seed=17, left_bit=0, right_bit=1)
    counterfactual = render_visual_xor_png(seed=17, left_bit=1, right_bit=1)

    assert first == second
    assert first != counterfactual
    with Image.open(io.BytesIO(first)) as image:
        assert image.size == (256, 256)
        assert image.mode == "RGB"
        assert image.info == {}


def test_build_xor_rows_are_balanced_and_do_not_leak_answers():
    rows = build_xor_rows(num_examples=40, seed=123, split="train")

    assert len(rows) == 40
    assert {row["metadata"]["combination"] for row in rows} == {"00", "01", "10", "11"}
    combination_counts = {
        combination: sum(row["metadata"]["combination"] == combination for row in rows)
        for combination in ("00", "01", "10", "11")
    }
    assert combination_counts == {
        "00": 10,
        "01": 10,
        "10": 10,
        "11": 10,
    }
    assert sum(row["label"] == "A" for row in rows) == 20
    assert sum(row["label"] == "B" for row in rows) == 20
    assert all(row["prompt"] == f"<image>{XOR_USER_PROMPT}" for row in rows)
    assert all(row["metadata"]["combination"] not in row["prompt"] for row in rows)
    assert len({row["metadata"]["sample_id"] for row in rows}) == len(rows)


def test_sft_rows_use_noisy_left_glyph_targets_without_xor_prompt():
    rows = build_sft_rows(num_examples=1000, seed=456, split="train", preferred_action_probability=0.7)

    preferred = sum(row["target"] == row["preferred_action"] for row in rows) / len(rows)
    assert 0.65 <= preferred <= 0.75
    assert all(row["prompt"] == SFT_USER_PROMPT for row in rows)
    assert all(row["target"] in {"A", "B"} for row in rows)
    assert all(row["preferred_action"] == ("A" if row["left_bit"] == 0 else "B") for row in rows)
    assert all("same" not in row["prompt"].lower() and "different" not in row["prompt"].lower() for row in rows)


def test_dataset_bundle_uses_disjoint_seeds_and_relax_multimodal_schema(tmp_path):
    paths = write_dataset_bundle(
        tmp_path,
        sft_train_examples=8,
        sft_eval_examples=8,
        xor_train_examples=8,
        xor_eval_examples=8,
        seed=42,
    )

    assert set(paths) == {"sft_train", "sft_eval", "xor_train", "xor_eval"}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    seed_sets = {name: set(frame["render_seed"]) for name, frame in frames.items()}
    for name, seeds in seed_sets.items():
        assert all(seeds.isdisjoint(other) for other_name, other in seed_sets.items() if other_name != name)
    assert frames["xor_train"].iloc[0]["prompt"] == f"<image>{XOR_USER_PROMPT}"
    assert isinstance(frames["xor_train"].iloc[0]["image"], (list, np.ndarray))
    assert isinstance(frames["xor_train"].iloc[0]["image"][0], bytes)


@pytest.mark.parametrize("num_examples", [0, 2, 6])
def test_xor_rows_require_a_positive_multiple_of_four(num_examples):
    with pytest.raises(ValueError, match="positive multiple of 4"):
        build_xor_rows(num_examples=num_examples, seed=1, split="train")
