import argparse

from examples.visual_xor.task import write_refinement_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the easy visual-XOR refinement datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-train-examples", type=int, default=8000)
    parser.add_argument("--sft-eval-examples", type=int, default=1000)
    parser.add_argument("--rl-train-examples", type=int, default=64)
    parser.add_argument("--rl-eval-examples", type=int, default=128)
    parser.add_argument("--correct-action-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_refinement_dataset_bundle(
        args.output_dir,
        sft_train_examples=args.sft_train_examples,
        sft_eval_examples=args.sft_eval_examples,
        rl_train_examples=args.rl_train_examples,
        rl_eval_examples=args.rl_eval_examples,
        correct_action_probability=args.correct_action_probability,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
