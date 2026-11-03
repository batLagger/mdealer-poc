"""Command-line example of the existing train_ltc_fuzzy function."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from hand_emb import train_ltc_fuzzy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-hands", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output/models"))
    args = parser.parse_args()

    if args.num_hands < 2:
        parser.error("--num-hands must be at least 2 for training and validation")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    output_dir = args.output_dir.resolve()
    model_path = output_dir / "ltc_fuzzy_model.pth"
    if model_path.exists():
        parser.error(f"Model already exists: {model_path}; choose another --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # The existing trainer saves to a fixed filename in the working directory.
    original_dir = Path.cwd()
    try:
        os.chdir(output_dir)
        train_ltc_fuzzy(
            num_hands=args.num_hands,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    finally:
        os.chdir(original_dir)
    print(f"Saved fuzzy model: {model_path}")
