#hand2_embedding
# LTC and shape embeddings for 2 partnership hands (North and South)

from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from bridge_generator import calc_losers, print_hand

NORTH = 1
SOUTH = 2
LTC_PAIR_BASE = 24.0  # expected tricks = 24 - (LTC_N + LTC_S)


# -----------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------

def encode_pair(north: torch.Tensor, south: torch.Tensor) -> torch.Tensor:
    """Combine two binary [4,13] hands into a single [4,13] tensor (0 = none, 1 = North, 2 = South)."""
    if north.shape != (4, 13) or south.shape != (4, 13):
        raise ValueError(f"Expected two [4,13] hands, got {tuple(north.shape)} and {tuple(south.shape)}")
    north, south = north.to(torch.float32), south.to(torch.float32)
    for name, hand in (("North", north), ("South", south)):
        if not torch.all((hand == 0) | (hand == 1)):
            raise ValueError(f"{name} hand must be binary encoded")
        if int(hand.sum().item()) != 13:
            raise ValueError(f"{name} hand must contain exactly 13 cards")
    if torch.any((north * south) > 0):
        raise ValueError("North and South hands share at least one card")
    return north * NORTH + south * SOUTH


def decode_pair(pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a [4,13] or [N,4,13] pair tensor into binary (north, south) tensors."""
    return (pair == NORTH).to(torch.float32), (pair == SOUTH).to(torch.float32)


def verify_pairs(pairs: torch.Tensor) -> None:
    """Raise ValueError unless pairs is [N,4,13] with values in {0,1,2} and 13 cards each for N and S."""
    if pairs.ndim != 3 or tuple(pairs.shape[1:]) != (4, 13):
        raise ValueError(f"Expected 3D tensor [N,4,13], got shape {tuple(pairs.shape)}")
    if not torch.all((pairs == 0) | (pairs == NORTH) | (pairs == SOUTH)):
        raise ValueError("Pairs must be encoded with values in {0, 1, 2}")
    north, south = decode_pair(pairs)
    if not torch.all(north.sum(dim=(1, 2)) == 13) or not torch.all(south.sum(dim=(1, 2)) == 13):
        raise ValueError("North and South must each contain exactly 13 cards")


def generate_random_pair() -> torch.Tensor:
    """Deal one random North/South pair from a shuffled deck as a [4,13] 0/1/2 tensor."""
    pair = torch.zeros(52, dtype=torch.float32)
    cards = torch.randperm(52)
    pair[cards[:13]] = NORTH
    pair[cards[13:26]] = SOUTH
    return pair.reshape(4, 13)


def generate_pair_training_set(num_pairs: int) -> torch.Tensor:
    """Return [num_pairs,4,13] random North/South pairs."""
    return torch.stack([generate_random_pair() for _ in range(num_pairs)])


def pair_labels(pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (tricks, ltc_north, ltc_south) for a [N,4,13] pair batch, with tricks = 24 - (LTC_N + LTC_S)."""
    north, south = decode_pair(pairs)
    ltc_n = torch.tensor([calc_losers(h)[0] for h in north], dtype=torch.float32)
    ltc_s = torch.tensor([calc_losers(h)[0] for h in south], dtype=torch.float32)
    return LTC_PAIR_BASE - (ltc_n + ltc_s), ltc_n, ltc_s


def print_pair(pair: torch.Tensor, print_hcp: bool = False) -> None:
    """Print the North and South hands of a [4,13] pair with their LTC and expected tricks."""
    north, south = decode_pair(pair)
    ltc_n, ltc_s = calc_losers(north)[0], calc_losers(south)[0]
    print("North:")
    print_hand(north, print_hcp=print_hcp)
    print("South:")
    print_hand(south, print_hcp=print_hcp)
    print(f"LTC N: {ltc_n:.2f} | LTC S: {ltc_s:.2f} | expected tricks: {LTC_PAIR_BASE - (ltc_n + ltc_s):.2f}")




def _suit_str(suit: torch.Tensor, void: str = "----") -> str:
    RANKS = "23456789TJQKA"
    cards = "".join(RANKS[i] for i in range(12, -1, -1) if suit[i] > 0)
    return cards or void


def print_pair2(pairs: torch.Tensor, gap: int = 16) -> None:
    """Print North and South side by side: one row per suit (S, H, D, C), South starting at column `gap`.
    Accepts a [4,13] pair or a [N,4,13] batch.
    """
    pairs = pairs.unsqueeze(0) if pairs.ndim == 2 else pairs
    for idx, pair in enumerate(pairs):
        if idx > 0:
            print()
        north, south = decode_pair(pair)
        ltc_n, ltc_s = calc_losers(north)[0], calc_losers(south)[0]
        print(f"===========") #{'North':<{gap}}South")
        for suit in range(4):
            print(f"{_suit_str(north[suit]):<{gap}}{_suit_str(south[suit])}")
        print()
        print(f"{f'LTC {ltc_n:.2f}':<{gap}}LTC {ltc_s:.2f}  | Exp tricks: {LTC_PAIR_BASE - (ltc_n + ltc_s):.2f}")
        print("=======")
        #print(f"Expected tricks: {LTC_PAIR_BASE - (ltc_n + ltc_s):.2f}")


def create_query_pair(query_s: str, query_n: str) -> torch.Tensor:
    """ Input strings (query_s) AQJ4.KT.J7532.94 
        query_n too. KQJ5432..QT42.43 (void in hearts)
        Ouput: Tensor [4x13] with query_s as 1s and query_n as 2s
    """
    pair = torch.zeros((4, 13), dtype=torch.float32)
    for i, suit in enumerate(query_s.split(".")):
        for r in suit:
            pair[i, "23456789TJQKA".index(r)] = 1.
    for i, suit in enumerate(query_n.split(".")):
        for r in suit:
            pair[i, "23456789TJQKA".index(r)] = 2.
    return pair


# -----------------------------------------------------------
# Model
# -----------------------------------------------------------

class PairOutputs(NamedTuple):
    main: torch.Tensor                  # (B, 1) predicted tricks
    aux_tricks: torch.Tensor            # (B, 1) tricks predicted from the embedding only
    aux_ltc_north: torch.Tensor         # (B, 1)
    aux_ltc_south: torch.Tensor         # (B, 1)
    aux_shape_north: torch.Tensor       # (B, 4) suit lengths
    aux_shape_south: torch.Tensor       # (B, 4)
    aux_sorted_shape_north: torch.Tensor  # (B, 4) sorted suit lengths / 13
    aux_sorted_shape_south: torch.Tensor  # (B, 4)
    aux_fit_shape: torch.Tensor         # (B, 4) N+S suit lengths
    aux_fit_sorted_shape: torch.Tensor  # (B, 4) sorted N+S suit lengths / 13




class HandLTCfuzzyPair(nn.Module):
    """North/South pair version of HandLTCfuzzy.

    Input is a [B,4,13] tensor with 0 = none, 1 = North, 2 = South. Per-hand branches share
    weights between North and South; pair-level branches model the combined (fit) shape.
    Embedding layout:
        [card_N | card_S | shape_N | shape_S | sorted_N | sorted_S | fit_shape | fit_sorted]
    """

    def __init__(self, hidden: int = 64, embedding_dim: int = 16, sorted_shape_dim: int = 8):
        super().__init__()
        # shared per-hand embedding layers
        self.card_emb         = nn.Linear(52, embedding_dim // 2)
        self.shape_emb        = nn.Linear(4, embedding_dim // 2)
        self.sorted_shape_emb = nn.Linear(4, sorted_shape_dim)
        # pair-level (combined N+S) shape embedding layers
        self.fit_shape_emb        = nn.Linear(4, embedding_dim // 2)
        self.fit_sorted_shape_emb = nn.Linear(4, sorted_shape_dim)
        # shared per-hand trunk
        self.full_branch = nn.Sequential(
            nn.Linear(52, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.shape_branch = nn.Sequential(
            nn.Linear(4, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        self.loser_branch = nn.Sequential(
            nn.Linear(12, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        # fit interaction between N and S suit lengths
        self.fit_branch = nn.Sequential(
            nn.Linear(8, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        hand_emb_dim = embedding_dim + sorted_shape_dim               # card + shape + sorted, per hand
        emb_dim      = 2 * hand_emb_dim + embedding_dim // 2 + sorted_shape_dim
        trunk_dim    = hidden // 2 + hidden // 4 + hidden // 4
        head_in      = 2 * trunk_dim + hidden // 4 + emb_dim
        self.embedding_size = emb_dim

        self.head                      = nn.Linear(head_in, 1)
        self.aux_tricks_head           = nn.Linear(emb_dim, 1)
        self.hand_ltc_aux_head         = nn.Linear(hand_emb_dim, 1)
        self.shape_aux_head            = nn.Linear(embedding_dim // 2, 4)
        self.sorted_shape_aux_head     = nn.Linear(sorted_shape_dim, 4)
        self.fit_shape_aux_head        = nn.Linear(embedding_dim // 2, 4)
        self.fit_sorted_shape_aux_head = nn.Linear(sorted_shape_dim, 4)

    @staticmethod
    def _hand_features(hand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat           = hand.reshape(hand.size(0), -1)             # (B, 52)
        suit_lengths   = hand.sum(dim=2)                            # (B, 4)
        top3           = hand[:, :, -3:].reshape(hand.size(0), 12)  # (B, 12) Q/K/A per suit
        sorted_lengths = suit_lengths.sort(dim=1, descending=True).values
        return flat, suit_lengths, top3, sorted_lengths

    def _hand_embedding(self, flat, suit_lengths, sorted_lengths):
        return self.card_emb(flat), self.shape_emb(suit_lengths), self.sorted_shape_emb(sorted_lengths)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the [B, embedding_size] pair embedding."""
        return self._forward_parts(x)[0]

    def _forward_parts(self, x: torch.Tensor):
        north, south = decode_pair(x)
        n_flat, n_len, n_top3, n_sorted = self._hand_features(north)
        s_flat, s_len, s_top3, s_sorted = self._hand_features(south)
        fit_len    = n_len + s_len
        fit_sorted = fit_len.sort(dim=1, descending=True).values

        card_n, shape_n, sorted_n = self._hand_embedding(n_flat, n_len, n_sorted)
        card_s, shape_s, sorted_s = self._hand_embedding(s_flat, s_len, s_sorted)
        fit_shape_out  = self.fit_shape_emb(fit_len)
        fit_sorted_out = self.fit_sorted_shape_emb(fit_sorted)
        emb = torch.cat([card_n, card_s, shape_n, shape_s, sorted_n, sorted_s,
                         fit_shape_out, fit_sorted_out], dim=1)
        parts = dict(card_n=card_n, card_s=card_s, shape_n=shape_n, shape_s=shape_s,
                     sorted_n=sorted_n, sorted_s=sorted_s,
                     fit_shape=fit_shape_out, fit_sorted=fit_sorted_out,
                     trunk_n=(n_flat, n_len, n_top3), trunk_s=(s_flat, s_len, s_top3))
        return emb, parts

    def forward(self, x: torch.Tensor) -> PairOutputs:
        # x: (B, 4, 13) with values in {0, 1, 2}
        emb, p = self._forward_parts(x)
        n_flat, n_len, n_top3 = p["trunk_n"]
        s_flat, s_len, s_top3 = p["trunk_s"]
        trunk_n = torch.cat([self.full_branch(n_flat), self.shape_branch(n_len), self.loser_branch(n_top3)], dim=1)
        trunk_s = torch.cat([self.full_branch(s_flat), self.shape_branch(s_len), self.loser_branch(s_top3)], dim=1)
        fit = self.fit_branch(torch.cat([n_len, s_len], dim=1))

        hand_emb_n = torch.cat([p["card_n"], p["shape_n"], p["sorted_n"]], dim=1)
        hand_emb_s = torch.cat([p["card_s"], p["shape_s"], p["sorted_s"]], dim=1)
        return PairOutputs(
            main=self.head(torch.cat([trunk_n, trunk_s, fit, emb], dim=1)),
            aux_tricks=self.aux_tricks_head(emb),
            aux_ltc_north=self.hand_ltc_aux_head(hand_emb_n),
            aux_ltc_south=self.hand_ltc_aux_head(hand_emb_s),
            aux_shape_north=self.shape_aux_head(p["shape_n"]),
            aux_shape_south=self.shape_aux_head(p["shape_s"]),
            aux_sorted_shape_north=self.sorted_shape_aux_head(p["sorted_n"]),
            aux_sorted_shape_south=self.sorted_shape_aux_head(p["sorted_s"]),
            aux_fit_shape=self.fit_shape_aux_head(p["fit_shape"]),
            aux_fit_sorted_shape=self.fit_sorted_shape_aux_head(p["fit_sorted"]),
        )

    def embedding_slices(self) -> dict[str, slice]:
        """Return slices of the embedding for each similarity component (derived from layer sizes)."""
        card  = self.card_emb.out_features
        shape = self.shape_emb.out_features
        srt   = self.sorted_shape_emb.out_features
        fit   = self.fit_shape_emb.out_features
        fit_s = self.fit_sorted_shape_emb.out_features
        bounds, start = {}, 0
        for name, size in (("card", 2 * card), ("shape", 2 * shape), ("sorted_shape", 2 * srt),
                           ("fit_shape", fit), ("fit_sorted_shape", fit_s)):
            bounds[name] = slice(start, start + size)
            start += size
        return bounds


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

def train_ltc_fuzzy_pair(
    num_pairs: int = 300_000,
    epochs: int = 14,
    batch_size: int = 256,
    hidden: int = 64,
    embedding_dim: int = 16,
    sorted_shape_dim: int = 8,
    lr: float = 1e-3,
    train_split: float = 0.8,
    aux_weight: float = 0.5,
    hand_ltc_aux_weight: float = 0.3,
    shape_aux_weight: float = 0.3,
    sorted_shape_aux_weight: float = 0.3,
    fit_shape_aux_weight: float = 0.3,
    fit_sorted_shape_aux_weight: float = 0.3,
    save_path: str | None = "ltc_fuzzy_pair_model.pth",
    plot: bool = True,
    device=None) -> HandLTCfuzzyPair:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    print("Generating pairs and computing trick labels...")
    pairs = generate_pair_training_set(num_pairs)
    tricks, ltc_n, ltc_s = pair_labels(pairs)

    dataset = TensorDataset(pairs, tricks, ltc_n, ltc_s)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(51)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    model     = HandLTCfuzzyPair(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_maes, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for pairs_b, tricks_b, ltc_n_b, ltc_s_b in train_loader:
            pairs_b, tricks_b = pairs_b.to(device), tricks_b.to(device)
            ltc_n_b, ltc_s_b = ltc_n_b.to(device), ltc_s_b.to(device)
            optimizer.zero_grad()
            out = model(pairs_b)

            north, south = decode_pair(pairs_b)
            len_n, len_s = north.sum(dim=2), south.sum(dim=2)  # (B, 4) true suit lengths
            fit_len = len_n + len_s
            sorted_n   = len_n.sort(dim=1, descending=True).values / 13.0
            sorted_s   = len_s.sort(dim=1, descending=True).values / 13.0
            fit_sorted = fit_len.sort(dim=1, descending=True).values / 13.0

            loss = (criterion(out.main.squeeze(1), tricks_b)
                    + aux_weight * criterion(out.aux_tricks.squeeze(1), tricks_b)
                    + hand_ltc_aux_weight * (criterion(out.aux_ltc_north.squeeze(1), ltc_n_b)
                                             + criterion(out.aux_ltc_south.squeeze(1), ltc_s_b))
                    + shape_aux_weight * (criterion(out.aux_shape_north, len_n)
                                          + criterion(out.aux_shape_south, len_s))
                    + sorted_shape_aux_weight * (criterion(out.aux_sorted_shape_north, sorted_n)
                                                 + criterion(out.aux_sorted_shape_south, sorted_s))
                    + fit_shape_aux_weight * criterion(out.aux_fit_shape, fit_len)
                    + fit_sorted_shape_aux_weight * criterion(out.aux_fit_sorted_shape, fit_sorted))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        abs_errors, within_half = [], []
        with torch.no_grad():
            for pairs_b, tricks_b, _, _ in val_loader:
                pairs_b, tricks_b = pairs_b.to(device), tricks_b.to(device)
                preds = model(pairs_b).main.squeeze(1)
                err = (preds - tricks_b).abs()
                abs_errors.append(err)
                within_half.append((err <= 0.5).float())

        train_loss = running_loss / len(train_loader)
        val_mae    = torch.cat(abs_errors).mean().item()
        val_acc    = torch.cat(within_half).mean().item()  # % within ±0.5 tricks
        train_losses.append(train_loss)
        val_maes.append(val_mae)
        val_accs.append(val_acc)
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val MAE {val_mae:.3f} | val acc (±0.5) {val_acc:.3f}")

    if plot:
        epoch_numbers = range(1, epochs + 1)
        figure, (loss_ax, mae_ax, acc_ax) = plt.subplots(1, 3, figsize=(15, 4))
        loss_ax.plot(epoch_numbers, train_losses, marker="o")
        loss_ax.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
        loss_ax.grid(alpha=0.3)
        mae_ax.plot(epoch_numbers, val_maes, marker="o", color="tab:orange")
        mae_ax.set(title="Validation MAE (tricks)", xlabel="Epoch", ylabel="MAE")
        mae_ax.grid(alpha=0.3)
        acc_ax.plot(epoch_numbers, val_accs, marker="o", color="tab:green")
        acc_ax.set(title="Accuracy (pred within ±0.5)", xlabel="Epoch", ylabel="Accuracy")
        acc_ax.set_ylim(0, 1)
        acc_ax.grid(alpha=0.3)
        figure.tight_layout()
        plt.show()

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    return model


# -----------------------------------------------------------
# Inference
# -----------------------------------------------------------

def load_ltc_fuzzy_pair_model(path: str = "ltc_fuzzy_pair_model.pth", hidden: int = 64, embedding_dim: int = 16,
                              sorted_shape_dim: int = 8, device=None) -> HandLTCfuzzyPair:
    """Instantiate HandLTCfuzzyPair and load weights from a saved state-dict file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandLTCfuzzyPair(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_ltc_embedding_fuzzy_pair(model: HandLTCfuzzyPair, pairs: torch.Tensor, device=None) -> torch.Tensor:
    """Return the [N, embedding_size] pair embeddings for a [N,4,13] batch."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    pairs = pairs.to(torch.float32).to(device)
    with torch.no_grad():
        emb = model.embed(pairs)
    return emb.cpu()


def predict_ltc_pair(model: HandLTCfuzzyPair, pair: torch.Tensor) -> torch.Tensor:
    """Return predicted expected tricks for a [4,13] pair or [N,4,13] batch."""
    device = next(model.parameters()).device
    x = pair.to(torch.float32).to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds = model(x).main.squeeze(1).cpu()
    return preds if pair.ndim == 3 else preds[0]


def hands_similarity_fuzzy_pair(
    query: torch.Tensor,
    pair_set: torch.Tensor,
    model: HandLTCfuzzyPair,
    ltc_param: float = 1.0,               # weight on card_N|card_S cosine
    shape_param: float = 1.0,             # weight on exact per-hand shapes
    sorted_shape_param: float = 1.0,      # weight on sorted per-hand shapes
    fit_shape_param: float = 1.0,         # weight on combined N+S suit lengths
    fit_sorted_shape_param: float = 1.0,  # weight on sorted combined N+S suit lengths
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sorted_indices, scores) of pair_set ranked by weighted cosine similarity to query.
    N/S order matters: (N,S) and (S,N) are different pairs.
    """
    q_emb = get_ltc_embedding_fuzzy_pair(model, query.unsqueeze(0))  # (1, D)
    s_emb = get_ltc_embedding_fuzzy_pair(model, pair_set)            # (N, D)

    weights = {"card": ltc_param, "shape": shape_param, "sorted_shape": sorted_shape_param,
               "fit_shape": fit_shape_param, "fit_sorted_shape": fit_sorted_shape_param}
    scores = torch.zeros(s_emb.size(0))
    for name, sl in model.embedding_slices().items():
        if weights[name] != 0:
            scores += weights[name] * torch.nn.functional.cosine_similarity(q_emb[:, sl], s_emb[:, sl])

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    sorted_idx = scores.argsort(descending=True)
    return sorted_idx, scores[sorted_idx]


if __name__ == "__main__":
    model = train_ltc_fuzzy_pair()
    # model = load_ltc_fuzzy_pair_model("ltc_fuzzy_pair_model.pth")
    pairs = generate_pair_training_set(400_000)
    query = pairs[0]
    print("Query pair:")
    print_pair(query)

    #calculate expected tricks for the pairs batch, using the ACTUAL calculation (24-ltc(north)-ltc(south))
    norths, souths = decode_pair(pairs)
    ltc_norths, ltc_souths = [calc_losers(n)[0] for n in norths], [calc_losers(s)[0] for s in souths]
    expected_tricks = [LTC_PAIR_BASE - (ltc_n + ltc_s) for ltc_n, ltc_s in zip(ltc_norths, ltc_souths)]

    #get sorted indexes of expected tricks (lambda)
    sorted_expected_idx_lambda = sorted(range(len(expected_tricks)), key=lambda i: expected_tricks[i], reverse=True)
    
    #get sorted indexes of expected tricks (torch argsort descending)
    sorted_expected_idx = torch.tensor(expected_tricks).argsort(descending=True)

    gap = 14
    for ltc_n, ltc_s in zip(ltc_norths, ltc_souths):
        print(f"{f'LTC {ltc_n:.2f}':<{gap}}LTC {ltc_s:.2f}  | Exp tricks: {LTC_PAIR_BASE - (ltc_n + ltc_s):.2f}")

     
    expected_tricks = LTC_PAIR_BASE - predict_ltc_pair(model, pairs[:, 0]) - predict_ltc_pair(model, pairs[:, 1])
    #LTC_PAIR_BASE - (ltc_n + ltc_s)

    sorted_idx, scores = hands_similarity_fuzzy_pair(
        query, pairs, model, ltc_param=1, shape_param=0.2, sorted_shape_param=0.5,
        fit_shape_param=0.2, fit_sorted_shape_param=0.7)
    preds = predict_ltc_pair(model, pairs[sorted_idx[:10]])
    for rank, i in enumerate(sorted_idx[:10]):
        print_pair(pairs[i])
        print(f"Predicted tricks: {preds[rank]:.2f} | score: {scores[rank]:.3f}\n")
