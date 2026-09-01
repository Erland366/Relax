# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert downloaded hiyouga/geometry3k Parquet splits into Relax rows."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image


def _encode_image(image: Any) -> list[bytes]:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Geo3K image must decode to PIL.Image.Image, got {type(image).__name__}")
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return [output.getvalue()]


def _convert_dataset(dataset, output_path: Path, *, split_name: str, source_name: str) -> int:
    required = {"images", "problem", "answer"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Geo3K {split_name} split is missing column(s): {missing}")
    rows = []
    for index, sample in enumerate(dataset):
        images = sample["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"Geo3K {split_name} row {index} must contain exactly one image")
        prompt = str(sample["problem"])
        if "<image>" not in prompt:
            raise ValueError(f"Geo3K {split_name} row {index} prompt is missing the <image> placeholder")
        rows.append(
            {
                "prompt": prompt,
                "images": _encode_image(images[0]),
                "reward_model": {"ground_truth": str(sample["answer"])},
                "metadata": {"source": source_name, "split": split_name, "index": index},
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def _convert_split(source_path: Path, output_path: Path, *, split_name: str) -> int:
    dataset = load_dataset("parquet", data_files={split_name: str(source_path)}, split=split_name)
    return _convert_dataset(dataset, output_path, split_name=split_name, source_name=str(source_path.resolve()))


def prepare_data(
    input_dir: str | Path | None,
    output_dir: str | Path,
    *,
    dataset_id: str | None = None,
) -> dict[str, int]:
    if (input_dir is None) == (dataset_id is None):
        raise ValueError("Provide exactly one of input_dir or dataset_id")
    output_root = Path(output_dir)
    if dataset_id is not None:
        dataset_dict = load_dataset(dataset_id)
        missing_splits = sorted({"train", "validation", "test"} - set(dataset_dict))
        if missing_splits:
            raise ValueError(f"Geo3K dataset {dataset_id!r} is missing split(s): {missing_splits}")
        return {
            split: _convert_dataset(
                dataset_dict[split],
                output_root / f"{split}.parquet",
                split_name=split,
                source_name=dataset_id,
            )
            for split in ("train", "validation", "test")
        }

    source_root = Path(input_dir)
    split_files = {
        "train": source_root / "train.parquet",
        "validation": source_root / "validation.parquet",
        "test": source_root / "test.parquet",
    }
    missing = [str(path) for path in split_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Downloaded Geo3K split file(s) are missing: {missing}")
    return {
        split: _convert_split(source, output_root / f"{split}.parquet", split_name=split)
        for split, source in split_files.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir")
    source.add_argument("--dataset-id")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    counts = prepare_data(args.input_dir, args.output_dir, dataset_id=args.dataset_id)
    print("Converted Geo3K splits: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
