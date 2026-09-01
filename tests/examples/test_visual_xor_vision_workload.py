# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import io

import pandas as pd
import pytest
from PIL import Image

from examples.visual_xor.build_vision_workload import _image_bytes, _resize_png, _validate_balanced_group


def _png(size=32):
    output = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(output, format="PNG")
    return output.getvalue()


def test_resize_png_preserves_single_image_container_and_changes_resolution():
    resized = _resize_png([_png()], 64)

    assert len(resized) == 1
    assert _image_bytes(resized).startswith(b"\x89PNG")
    with Image.open(io.BytesIO(resized[0])) as image:
        assert image.size == (64, 64)


def test_validate_balanced_group_accepts_equal_actions():
    _validate_balanced_group(pd.DataFrame({"label": ["A", "B", "A", "B"]}), name="test")


def test_validate_balanced_group_rejects_action_imbalance():
    with pytest.raises(ValueError, match="not A/B balanced"):
        _validate_balanced_group(pd.DataFrame({"label": ["A", "A", "B"]}), name="test")
