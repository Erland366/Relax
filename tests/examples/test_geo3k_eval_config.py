# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
from PIL import Image

from examples.geo3k.build_eval_config import build_eval_config
from examples.geo3k.prepare_data import _encode_image


def test_build_geo3k_eval_config_is_deterministic_and_uses_heldout_reward(tmp_path):
    test_data = tmp_path / "test.parquet"
    test_data.write_bytes(b"parquet-placeholder")

    config = build_eval_config(test_data)

    defaults = config["eval"]["defaults"]
    dataset = config["eval"]["datasets"][0]
    assert defaults["label_key"] == "reward_model"
    assert dataset["path"] == str(test_data.resolve())
    assert dataset["rm_type"] == "geo3k"
    assert dataset["temperature"] == 0.0
    assert dataset["n_samples_per_eval_prompt"] == 1


def test_build_geo3k_eval_config_rejects_missing_test_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Geo3K test parquet"):
        build_eval_config(tmp_path / "missing.parquet")


def test_encode_geo3k_image_produces_one_png_byte_string():
    encoded = _encode_image(Image.new("RGB", (32, 16), "white"))

    assert len(encoded) == 1
    assert encoded[0].startswith(b"\x89PNG")
