"""Print hands ranked by the existing hands_similarity_fuzzy function."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from bridge_generator import print_hand, verify_generated_hands
from hand_emb import hands_similarity_fuzzy, load_ltc_fuzzy_model
from hand_pair_emb import create_query_pair, decode_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand", required=True, help="13 cards in S.H.D.C order, e.g. AKQ.J53.9.KQJ985"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("demo_output/models/ltc_fuzzy_model.pth")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("demo_output/datasets/hands.pt")
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--ltc-param", type=float, default=1.0)
    parser.add_argument("--shape-param", type=float, default=1.0)
    parser.add_argument("--sorted-shape-param", type=float, default=1.0)
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be positive")
    weights = (args.ltc_param, args.shape_param, args.sorted_shape_param)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        parser.error("Similarity weights must be finite and nonnegative")
    if not any(weight > 0 for weight in weights):
        parser.error("At least one similarity weight must be positive")
    suits = args.hand.split(".")
    if (
        len(suits) != 4
        or any(rank not in "23456789TJQKA" for suit in suits for rank in suit)
        or any(len(set(suit)) != len(suit) for suit in suits)
        or sum(map(len, suits)) != 13
    ):
        parser.error(
            "--hand must contain exactly 13 distinct cards in S.H.D.C order; "
            "use uppercase ranks 23456789TJQKA and an empty suit for a void"
        )
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if not args.dataset.is_file():
        parser.error(f"Dataset file not found: {args.dataset}")

    query, _ = decode_pair(create_query_pair(args.hand, "..."))
    verify_generated_hands(query.unsqueeze(0))
    hands = torch.load(args.dataset, map_location="cpu", weights_only=True)
    if not isinstance(hands, torch.Tensor):
        parser.error("--dataset must contain a single tensor, not a checkpoint or mapping")
    verify_generated_hands(hands)
    if len(hands) == 0:
        parser.error("--dataset must contain at least one hand")
    model = load_ltc_fuzzy_model(str(args.model), device="cpu")
    indices, scores = hands_similarity_fuzzy(
        query,
        hands,
        model,
        ltc_param=args.ltc_param,
        shape_param=args.shape_param,
        sorted_shape_param=args.sorted_shape_param,
    )

    print("Query (suit rows: S, H, D, C):")
    print_hand(query)
    count = min(args.top, len(hands))
    print(f"Top {count} of {len(hands)} hands; dataset indices are zero-based.")
    print("Scores are catalog-relative normalized similarities, not probabilities.")
    for rank, (index, score) in enumerate(zip(indices[:count], scores[:count]), start=1):
        print(f"Rank {rank} | dataset index {index.item()} | similarity {score.item():.6f}")
        print_hand(hands[index.item()])
