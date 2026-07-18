import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from examples.visual_xor.train_sft import (
    DEFAULT_METADATA_REPO,
    DEFAULT_METADATA_REVISION,
    METADATA_ALLOW_PATTERNS,
    validate_metadata_allowlist,
    validate_metadata_only_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL processor metadata without model weights.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_METADATA_REPO)
    parser.add_argument("--revision", default=DEFAULT_METADATA_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    validate_metadata_allowlist(METADATA_ALLOW_PATTERNS)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=list(METADATA_ALLOW_PATTERNS),
    )
    validate_metadata_only_directory(output_dir)
    print(f"Prepared metadata-only Qwen3-VL processor at {output_dir}")


if __name__ == "__main__":
    main()
