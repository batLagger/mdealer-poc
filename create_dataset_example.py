"""Create a saved single-hand catalog using generate_training_set."""

import argparse
from pathlib import Path

import torch

from bridge_generator import generate_training_set, verify_generated_hands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-hands", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("demo_output/datasets/hands.pt")
    )
    args = parser.parse_args()

    if args.num_hands < 1:
        parser.error("--num-hands must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Dataset already exists: {output}; choose another --output")

    torch.manual_seed(args.seed)
    hands = generate_training_set(args.num_hands).to(torch.float32)
    verify_generated_hands(hands)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hands, output)
    print(f"Saved {len(hands)} hands: {output}")
    print(f"Tensor shape: {tuple(hands.shape)}; dtype: {hands.dtype}")
