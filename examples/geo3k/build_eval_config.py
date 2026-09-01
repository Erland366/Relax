# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a deterministic Relax evaluation config for a local Geo3K test split."""

import argparse
import json
from pathlib import Path


def build_eval_config(test_data: str | Path) -> dict:
    test_path = Path(test_data).resolve()
    if not test_path.is_file():
        raise FileNotFoundError(f"Geo3K test parquet does not exist: {test_path}")
    return {
        "eval": {
            "defaults": {
                "input_key": "prompt",
                "label_key": "reward_model",
                "top_p": 1.0,
                "top_k": -1,
                "max_response_len": 2048,
            },
            "datasets": [
                {
                    "name": "geo3k_heldout",
                    "path": str(test_path),
                    "rm_type": "geo3k",
                    "n_samples_per_eval_prompt": 1,
                    "temperature": 0.0,
                }
            ],
        }
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_eval_config(args.test_data), indent=2) + "\n")


if __name__ == "__main__":
    main()
