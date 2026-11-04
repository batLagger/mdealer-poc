import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import random
import numpy as np
from bridge_generator import closest_hand 
from bridge_generator import generate_training_set, print_hand, calc_losers, hcp as calc_hcp
from hand_pair_emb import create_query_pair, print_pair2

## hand embedding of bridge hands


class HandLTCfuzzy(nn.Module):
    """HandLTC variant with a dedicated, slice-local sorted-shape embedding subspace."""

    def __init__(self, hidden: int = 64, embedding_dim: int = 16, sorted_shape_dim: int = 8):
        super().__init__()
        self.card_emb         = nn.Linear(52, embedding_dim // 2)   # pure card structure, no scalar leak
        self.shape_emb        = nn.Linear(4, embedding_dim // 2)    # exact (suit-identity) shape subspace
        self.sorted_shape_emb = nn.Linear(4, sorted_shape_dim)      # sorted (suit-agnostic) shape subspace
        # full card layout
        self.full_branch = nn.Sequential(
            nn.Linear(52, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # suit lengths detect shape (void/singleton penalise loser count differently)
        self.shape_branch = nn.Sequential(
            nn.Linear(4, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        # Q/K/A per suit — the 3 ranks that directly determine losers
        self.loser_branch = nn.Sequential(
            nn.Linear(12, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        emb_dim = embedding_dim + sorted_shape_dim
        head_in = hidden // 2 + hidden // 4 + hidden // 4 + emb_dim
        self.head                  = nn.Linear(head_in, 1)
        self.aux_head              = nn.Linear(emb_dim, 1)             # predicts LTC from the full embedding
        self.shape_aux_head        = nn.Linear(embedding_dim // 2, 4)  # predicts suit lengths from shape_emb only
        self.sorted_shape_aux_head = nn.Linear(sorted_shape_dim, 4)    # predicts sorted suit lengths from sorted_shape_emb only

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 4, 13)
        flat         = x.reshape(x.size(0), -1)             # (B, 52)
        suit_lengths = x.sum(dim=2)                         # (B, 4)
        top3         = x[:, :, -3:].reshape(x.size(0), 12)  # (B, 12) — Q/K/A per suit
        full = self.full_branch(flat)
        shp  = self.shape_branch(suit_lengths)
        lose = self.loser_branch(top3)

        sorted_lengths = suit_lengths.sort(dim=1, descending=True).values  # (B, 4) suit-agnostic shape

        card_out         = self.card_emb(flat)
        shape_out        = self.shape_emb(suit_lengths)
        sorted_shape_out = self.sorted_shape_emb(sorted_lengths)
        emb = torch.cat([card_out, shape_out, sorted_shape_out], dim=1)

        main_out             = self.head(torch.cat([full, shp, lose, emb], dim=1))  # (B, 1)
        aux_ltc_out          = self.aux_head(emb)                                    # (B, 1)
        aux_shape_out        = self.shape_aux_head(shape_out)                        # (B, 4)
        aux_sorted_shape_out = self.sorted_shape_aux_head(sorted_shape_out)          # (B, 4)
        return main_out, aux_ltc_out, aux_shape_out, aux_sorted_shape_out


def train_ltc_fuzzy(
    num_hands: int = 500_000,
    epochs: int = 14,
    batch_size: int = 256,
    hidden: int = 64,
    embedding_dim: int = 16,
    sorted_shape_dim: int = 8,
    lr: float = 1e-3,
    train_split: float = 0.8,
    aux_weight: float = 0.5,
    shape_aux_weight: float = 0.3,
    sorted_shape_aux_weight: float = 0.3,
    device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    print("Generating hands and computing loser labels...")
    hands = generate_training_set(num_hands).to(torch.float32)
    labels = torch.tensor([calc_losers(h)[0] for h in hands], dtype=torch.float32)

    dataset = TensorDataset(hands, labels)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(51)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    model     = HandLTCfuzzy(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_maes, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for hands_b, labels_b in train_loader:
            hands_b, labels_b = hands_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            main_out, aux_ltc_out, aux_shape_out, aux_sorted_shape_out = model(hands_b)
            shape_target        = hands_b.sum(dim=2)  # (B, 4) true suit lengths
            sorted_shape_target = shape_target.sort(dim=1, descending=True).values / 13.0  # (B, 4) normalised, desc
            loss = (criterion(main_out.squeeze(1), labels_b) +
                    aux_weight * criterion(aux_ltc_out.squeeze(1), labels_b)
                    + shape_aux_weight * criterion(aux_shape_out, shape_target)
                    + sorted_shape_aux_weight * criterion(aux_sorted_shape_out, sorted_shape_target))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        abs_errors, within_half = [], []
        with torch.no_grad():
            for hands_b, labels_b in val_loader:
                hands_b, labels_b = hands_b.to(device), labels_b.to(device)
                preds = model(hands_b)[0].squeeze(1)  # main head only
                err = (preds - labels_b).abs()
                abs_errors.append(err)
                within_half.append((err <= 0.5).float())

        train_loss = running_loss / len(train_loader)
        val_mae    = torch.cat(abs_errors).mean().item()
        val_acc    = torch.cat(within_half).mean().item()  # % within ±0.5 losers
        train_losses.append(train_loss)
        val_maes.append(val_mae)
        val_accs.append(val_acc)
        print(f"Epoch {epoch:3d} | MSE {train_loss:.4f} | val MAE {val_mae:.3f} | val acc (±0.5) {val_acc:.3f}")

    #optional plot (disable the flag Agg if running in headless mode)
    epoch_numbers = range(1, epochs + 1)
    figure, (loss_ax, mae_ax, acc_ax) = plt.subplots(1, 3, figsize=(15, 4))
    loss_ax.plot(epoch_numbers, train_losses, marker="o")
    loss_ax.set(title="Training MSE Loss", xlabel="Epoch", ylabel="MSE")
    loss_ax.grid(alpha=0.3)
    mae_ax.plot(epoch_numbers, val_maes, marker="o", color="tab:orange")
    mae_ax.set(title="Validation MAE (losers)", xlabel="Epoch", ylabel="MAE")
    mae_ax.grid(alpha=0.3)
    acc_ax.plot(epoch_numbers, val_accs, marker="o", color="tab:green")
    acc_ax.set(title="Accuracy (pred within ±0.5)", xlabel="Epoch", ylabel="Accuracy")
    acc_ax.set_ylim(0, 1)
    acc_ax.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "ltc_fuzzy_model.pth")
    print("Model saved to ltc_fuzzy_model.pth")
    return model



def load_ltc_fuzzy_model(path: str = "ltc_fuzzy_model.pth", hidden: int = 64, embedding_dim: int = 16,
                          sorted_shape_dim: int = 8, device=None) -> HandLTCfuzzy:
    """Instantiate HandLTCfuzzy and load weights from a saved state-dict file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandLTCfuzzy(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_ltc_fuzzy(model: HandLTCfuzzy, hand: torch.Tensor) -> torch.Tensor:
    """Return predicted expected tricks for a [4,13] pair or [N,4,13] batch."""
    device = next(model.parameters()).device
    x = hand.to(torch.float32).to(device)
    if x.ndim == 2: #if x is a single hand, add a batch dimension up front
        x = x.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds = model(x)[0].squeeze(1).cpu()  # main head only
    return preds if hand.ndim == 3 else preds[0]


def get_ltc_embedding_fuzzy(model: HandLTCfuzzy, hands: torch.Tensor, device=None) -> torch.Tensor:
    """Return the three-stream (card / exact-shape / sorted-shape) embedding activations for a batch of hands."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hands = hands.to(torch.float32).to(device)
    with torch.no_grad():
        flat           = hands.reshape(hands.size(0), -1)
        suit_lengths   = hands.sum(dim=2)
        sorted_lengths = suit_lengths.sort(dim=1, descending=True).values
        emb = torch.cat([
            model.card_emb(flat),
            model.shape_emb(suit_lengths),
            model.sorted_shape_emb(sorted_lengths),
        ], dim=1)

    return emb.cpu()



def hand_from_indices(indices, device):
    out = torch.zeros((4, 13), dtype=torch.float32, device=device)
    for idx in indices:
        suit = idx // 13
        rank = idx % 13
        out[suit, rank] = 1.0
    return out



def hands_similarity_fuzzy(
    query: torch.Tensor,
    hand_set: torch.Tensor,
    model: HandLTCfuzzy,
    ltc_param: float = 1.0,           # weight on card_emb cosine (0 = ignore cards)
    shape_param: float = 1.0,         # weight on exact shape_emb cosine (0 = ignore exact shape)
    sorted_shape_param: float = 1.0,  # weight on sorted_shape_emb cosine (0 = ignore sorted shape)
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sorted_indices, scores) ranked by weighted cosine similarity across the
    card/LTC, exact-shape and sorted-shape embedding slices.
    """
    q_emb = get_ltc_embedding_fuzzy(model, query.unsqueeze(0))  # (1, D)
    s_emb = get_ltc_embedding_fuzzy(model, hand_set)            # (N, D)

    # slice boundaries come from each branch's own output size, not a hardcoded fraction
    card_dim  = model.card_emb.out_features
    shape_dim = model.shape_emb.out_features
    card_slice         = slice(0, card_dim)
    shape_slice        = slice(card_dim, card_dim + shape_dim)
    sorted_shape_slice = slice(card_dim + shape_dim, None)

    card_scores         = torch.nn.functional.cosine_similarity(q_emb[:, card_slice], s_emb[:, card_slice])
    shape_scores        = torch.nn.functional.cosine_similarity(q_emb[:, shape_slice], s_emb[:, shape_slice])
    sorted_shape_scores = torch.nn.functional.cosine_similarity(q_emb[:, sorted_shape_slice], s_emb[:, sorted_shape_slice])

    scores = ltc_param * card_scores + shape_param * shape_scores + sorted_shape_param * sorted_shape_scores
    #normalize scores to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    sorted_idx = scores.argsort(descending=True)

    return sorted_idx, scores[sorted_idx]


def generate_smart_hands(
    query_s: torch.Tensor,
    query_n: torch.Tensor,
    hand_set: torch.Tensor,
    model: HandLTCfuzzy,
    ltc_param: float = 1.0,
    shape_param: float = 0.04,
    sorted_shape_param: float = 0.05,
    num_pairs: int = 10
) -> torch.Tensor:
    """Return a pair of hands from hand_set that are most similar to the query hand."""
    
    #get most similar hands for query_s
    sorted_idx_s, scores = hands_similarity_fuzzy(query_s, hand_set, model, ltc_param, shape_param, sorted_shape_param)
    #get most similar hands for query_n
    sorted_idx_n, scores_n = hands_similarity_fuzzy(query_n, hand_set, model, ltc_param, shape_param, sorted_shape_param)
    
    #get the top K pairs using sorted_idx_s and sorted_idx_n, create for each with closest_hand 
    #return tensor of [K,4,13]
    closest_pairs = torch.zeros((num_pairs, 4, 13), dtype=hand_set.dtype, device=hand_set.device)

    for h in range(1, num_pairs + 1):
        #get closest pair for hands sorted_idx_n[h] and hands sorted_idx_s[h]
        closest_pair = closest_hand(hand_set[sorted_idx_n[h]], hand_set[sorted_idx_s[h]])
        closest_pairs[h-1] = closest_pair

    return closest_pairs
