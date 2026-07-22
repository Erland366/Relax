import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from examples.visual_xor.task import (
    SFT_USER_PROMPT,
    XOR_USER_PROMPT,
    build_refinement_control_rows,
    build_refinement_sft_rows,
    build_sft_rows,
    build_xor_rows,
    render_visual_xor_png,
    write_dataset_bundle,
    write_refinement_dataset_bundle,
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


def test_refinement_sft_rows_use_exact_xor_context_and_balanced_noise():
    rows = build_refinement_sft_rows(
        num_examples=400,
        seed=456,
        split="refinement_sft_train",
        correct_action_probability=0.7,
    )

    for combination in ("00", "01", "10", "11"):
        combination_rows = [row for row in rows if row["combination"] == combination]
        assert len(combination_rows) == 100
        assert sum(row["target_is_correct"] for row in combination_rows) == 70
        assert sum(row["target"] == row["preferred_action"] for row in combination_rows) == 70
    assert all(row["prompt"] == XOR_USER_PROMPT for row in rows)
    assert all(row["preferred_action"] == ("A" if row["left_bit"] == row["right_bit"] else "B") for row in rows)


def test_ordered_xor_rows_keep_every_rollout_batch_balanced():
    rows = build_xor_rows(num_examples=64, seed=123, split="refinement_rl_train", shuffle=False)

    for start in range(0, len(rows), 4):
        assert [row["metadata"]["combination"] for row in rows[start : start + 4]] == ["00", "01", "10", "11"]
        assert [row["label"] for row in rows[start : start + 4]] == ["A", "B", "B", "A"]


def test_refinement_visual_controls_replace_only_images():
    rows = build_xor_rows(num_examples=8, seed=123, split="refinement_rl_eval", shuffle=False)
    permuted = build_refinement_control_rows(rows, control="permuted")
    constant = build_refinement_control_rows(rows, control="constant")

    for original, controlled in zip(rows, permuted, strict=True):
        assert controlled["label"] == original["label"]
        assert controlled["prompt"] == original["prompt"]
        assert controlled["metadata"]["combination"] == original["metadata"]["combination"]
        assert controlled["metadata"]["visual_control"] == "permuted"
    assert [row["image"] for row in permuted] == [rows[(index + 1) % len(rows)]["image"] for index in range(len(rows))]
    assert len({row["image"][0] for row in constant}) == 1
    assert all(row["metadata"]["visual_control"] == "constant" for row in constant)


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


def test_refinement_bundle_is_disjoint_balanced_and_configures_multimodal_controls(tmp_path):
    paths = write_refinement_dataset_bundle(
        tmp_path,
        sft_train_examples=40,
        sft_eval_examples=40,
        rl_train_examples=64,
        rl_eval_examples=128,
        correct_action_probability=0.7,
        seed=42,
    )

    assert set(paths) == {
        "sft_train",
        "sft_eval",
        "rl_train",
        "rl_eval",
        "rl_eval_permuted",
        "rl_eval_constant",
        "eval_config",
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items() if name != "eval_config"}
    assert len(frames["rl_train"]) == 64
    assert len(frames["rl_eval"]) == 128
    assert frames["rl_train"]["metadata"].map(lambda value: value["combination"]).value_counts().to_dict() == {
        "00": 16,
        "01": 16,
        "10": 16,
        "11": 16,
    }
    assert set(frames["sft_train"]["render_seed"]).isdisjoint(frames["sft_eval"]["render_seed"])
    assert set(frames["sft_train"]["render_seed"]).isdisjoint(frames["rl_train"]["render_seed"])
    assert set(frames["rl_train"]["render_seed"]).isdisjoint(frames["rl_eval"]["render_seed"])

    config = json.loads(paths["eval_config"].read_text())
    datasets = {dataset["name"]: dataset for dataset in config["eval"]["datasets"]}
    assert datasets["visual_xor_heldout"]["n_samples_per_eval_prompt"] == 4
    assert datasets["visual_xor_heldout"]["temperature"] == 1.0
    assert datasets["visual_xor_permuted_control"]["temperature"] == 0.0
    assert datasets["visual_xor_constant_control"]["temperature"] == 0.0
    assert all(Path(dataset["path"]).is_absolute() for dataset in datasets.values())


@pytest.mark.parametrize("num_examples", [0, 2, 6])
def test_xor_rows_require_a_positive_multiple_of_four(num_examples):
    with pytest.raises(ValueError, match="positive multiple of 4"):
        build_xor_rows(num_examples=num_examples, seed=1, split="train")


@pytest.mark.parametrize("num_examples", [0, 2, 6])
def test_refinement_sft_rows_require_a_positive_multiple_of_four(num_examples):
    with pytest.raises(ValueError, match="positive multiple of 4"):
        build_refinement_sft_rows(num_examples=num_examples, seed=1, split="train")


def test_refinement_sft_rows_require_exact_per_combination_noise_counts():
    with pytest.raises(ValueError, match="exact integer number"):
        build_refinement_sft_rows(num_examples=64, seed=1, split="train", correct_action_probability=0.7)
