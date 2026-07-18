import argparse

from examples.visual_xor.task import write_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic multimodal SFT and visual XOR datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-train-examples", type=int, default=8192)
    parser.add_argument("--sft-eval-examples", type=int, default=1024)
    parser.add_argument("--xor-train-examples", type=int, default=4096)
    parser.add_argument("--xor-eval-examples", type=int, default=1024)
    parser.add_argument("--preferred-action-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_dataset_bundle(
        args.output_dir,
        sft_train_examples=args.sft_train_examples,
        sft_eval_examples=args.sft_eval_examples,
        xor_train_examples=args.xor_train_examples,
        xor_eval_examples=args.xor_eval_examples,
        preferred_action_probability=args.preferred_action_probability,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
