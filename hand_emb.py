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

#from hand_emb import load_ltc_model
## hand embedding of bridge hands
"""
## Example Hand: ♠AKQ ♥J53 ♦9 ♣KQJ985

### Current Binary Representation (4x13)
```
         2  3  4  5  6  7  8  9  T  J  Q  K  A
Spades:  0  0  0  0  0  0  0  0  0  0  1  1  1  ← cards present
Hearts:  0  1  0  1  0  0  0  0  0  1  0  0  0
Diamonds:0  0  0  0  0  0  0  1  0  0  0  0  0
Clubs:   0  0  0  1  0  0  1  1  0  1  1  1  0
```
"""


class HandLTC(nn.Module):
    """Three-branch MLP + embedding for Losing Trick Count regression (single float output)."""

    def __init__(self, hidden: int = 64, embedding_dim: int = 16):
        super().__init__()
        self.card_emb  = nn.Linear(53, embedding_dim // 2)   # card structure + sorted shape
        self.shape_emb = nn.Linear(4, embedding_dim // 2)    # shape subspace
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
        head_in = hidden // 2 + hidden // 4 + hidden // 4 + embedding_dim
        self.head           = nn.Linear(head_in, 1)
        self.aux_head       = nn.Linear(embedding_dim, 1)  # predicts LTC directly from embedding
        self.shape_aux_head = nn.Linear(embedding_dim, 4)  # predicts suit lengths from embedding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 4, 13)
        flat         = x.reshape(x.size(0), -1)           # (B, 52)
        suit_lengths = x.sum(dim=2)                       # (B, 4)
        top3         = x[:, :, -3:].reshape(x.size(0), 12)  # (B, 12) — Q/K/A per suit
        full = self.full_branch(flat)
        shp  = self.shape_branch(suit_lengths)
        lose = self.loser_branch(top3)

        sorted_shape = (suit_lengths.std(dim=1, unbiased=False, keepdim=True))  # (B, 1) shape-flatness scalar
        emb  = torch.cat([self.card_emb( torch.cat([flat, sorted_shape], dim=1) ) , self.shape_emb(suit_lengths)], dim=1)

        #sorted_shape = suit_lengths.sort(dim=1, descending=True).values / 13.0  # normalised, desc
        main_out       = self.head(torch.cat([full, shp, lose, emb], dim=1))  # (B, 1)
        aux_ltc_out    = self.aux_head(emb)                                    # (B, 1)
        aux_shape_out  = self.shape_aux_head(emb)                              # (B, 4)
        return main_out, aux_ltc_out, aux_shape_out


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


class HandLTCfuzzy2(nn.Module):
    """HandLTCfuzzy variant with Dropout regularization in full_branch.

    Identical to HandLTCfuzzy except full_branch applies nn.Dropout(dropout_rate)
    after each ReLU. Unlike HandLTCfuzzy (pure Linear+ReLU, no train/eval-sensitive
    layers), this model's outputs genuinely differ between .train() and .eval()
    mode, so callers must call .eval() before inference to disable dropout.
    """

    def __init__(self, hidden: int = 64, embedding_dim: int = 16, sorted_shape_dim: int = 8,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.card_emb         = nn.Linear(52, embedding_dim // 2)   # pure card structure, no scalar leak
        self.shape_emb        = nn.Linear(4, embedding_dim // 2)    # exact (suit-identity) shape subspace
        self.sorted_shape_emb = nn.Linear(4, sorted_shape_dim)      # sorted (suit-agnostic) shape subspace
        # full card layout, with dropout regularization
        self.full_branch = nn.Sequential(
            nn.Linear(52, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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


def train_ltc(
    num_hands: int = 300_000,
    epochs: int = 10,
    batch_size: int = 256,
    hidden: int = 64,
    embedding_dim: int = 16,
    lr: float = 1e-3,
    train_split: float = 0.8,
    aux_weight: float = 0.5,
    shape_aux_weight: float = 0.3,
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

    model     = HandLTC(hidden=hidden, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_maes, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for hands_b, labels_b in train_loader:
            hands_b, labels_b = hands_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            main_out, aux_ltc_out, aux_shape_out = model(hands_b)
            shape_target = hands_b.sum(dim=2)  # (B, 4) true suit lengths
            loss = (criterion(main_out.squeeze(1), labels_b) +
                    aux_weight * criterion(aux_ltc_out.squeeze(1), labels_b)
                    + shape_aux_weight * criterion(aux_shape_out, shape_target))
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

    torch.save(model.state_dict(), "ltc_model.pth")
    print("Model saved to ltc_model.pth")
    return model


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


def train_ltc_fuzzy2(
    num_hands: int = 500_000,
    epochs: int = 14,
    batch_size: int = 256,
    hidden: int = 64,
    embedding_dim: int = 16,
    sorted_shape_dim: int = 8,
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    train_split: float = 0.8,
    aux_weight: float = 0.5,
    shape_aux_weight: float = 0.3,
    sorted_shape_aux_weight: float = 0.3,
    device=None):
    """Train HandLTCfuzzy2 (dropout-regularized full_branch). Mirrors train_ltc_fuzzy."""
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

    model     = HandLTCfuzzy2(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim,
                               dropout_rate=dropout_rate).to(device)
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
                within_half.append((err <= 0.4).float())

        train_loss = running_loss / len(train_loader)
        val_mae    = torch.cat(abs_errors).mean().item()
        val_acc    = torch.cat(within_half).mean().item()  # % within ±0.4 losers
        train_losses.append(train_loss)
        val_maes.append(val_mae)
        val_accs.append(val_acc)
        print(f"Epoch {epoch:3d} | MSE {train_loss:.4f} | val MAE {val_mae:.3f} | val acc (±0.4) {val_acc:.3f}")

    epoch_numbers = range(1, epochs + 1)
    figure, (loss_ax, mae_ax, acc_ax) = plt.subplots(1, 3, figsize=(15, 4))
    loss_ax.plot(epoch_numbers, train_losses, marker="o")
    loss_ax.set(title="Training MSE Loss", xlabel="Epoch", ylabel="MSE")
    loss_ax.grid(alpha=0.3)
    mae_ax.plot(epoch_numbers, val_maes, marker="o", color="tab:orange")
    mae_ax.set(title="Validation MAE (losers)", xlabel="Epoch", ylabel="MAE")
    mae_ax.grid(alpha=0.3)
    acc_ax.plot(epoch_numbers, val_accs, marker="o", color="tab:green")
    acc_ax.set(title="Accuracy (pred within ±0.4)", xlabel="Epoch", ylabel="Accuracy")
    acc_ax.set_ylim(0, 1)
    acc_ax.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "ltc_fuzzy2_model.pth")
    print("Model saved to ltc_fuzzy2_model.pth")
    return model



def predict_ltc(model: HandLTC, hand: torch.Tensor) -> torch.Tensor:
    """Return predicted LTC for a [4,13] hand or [N,4,13] batch."""
    device = next(model.parameters()).device
    x = hand.to(torch.float32).to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds = model(x)[0].squeeze(1).cpu()  # main head only
    return preds if hand.ndim == 3 else preds[0]


def load_ltc_model(path: str = "ltc_model.pth", hidden: int = 64, embedding_dim: int = 16, device=None) -> HandLTC:
    """Instantiate HandLTC and load weights from a saved state-dict file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandLTC(hidden=hidden, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_ltc_embedding(model: HandLTC, hands: torch.Tensor, device=None) -> torch.Tensor:
    """Return the two-stream embedding activations for a batch of hands."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hands = hands.to(torch.float32).to(device)
    with torch.no_grad():
        flat         = hands.reshape(hands.size(0), -1)
        suit_lengths = hands.sum(dim=2)
        #concatenate flat (N,52) and sorted_shape (N,1) before feeding into card_emb
        sorted_shape = (suit_lengths.std(dim=1, unbiased=False, keepdim=True) / 13.0)  # (N, 1) shape-flatness scalar
        emb = torch.cat([model.card_emb(torch.cat([flat, sorted_shape], dim=1)), model.shape_emb(suit_lengths)], dim=1)

    return emb.cpu()


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


def load_ltc_fuzzy2_model(path: str = "ltc_fuzzy2_model.pth", hidden: int = 64, embedding_dim: int = 16,
                           sorted_shape_dim: int = 8, dropout_rate: float = 0.2, device=None) -> HandLTCfuzzy2:
    """Instantiate HandLTCfuzzy2 and load weights from a saved state-dict file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandLTCfuzzy2(hidden=hidden, embedding_dim=embedding_dim, sorted_shape_dim=sorted_shape_dim,
                           dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_ltc_fuzzy2(model: HandLTCfuzzy2, hand: torch.Tensor) -> torch.Tensor:
    """Return predicted expected tricks for a [4,13] pair or [N,4,13] batch.

    Calls model.eval() first so Dropout in full_branch is disabled for inference.
    """
    device = next(model.parameters()).device
    x = hand.to(torch.float32).to(device)
    if x.ndim == 2: #if x is a single hand, add a batch dimension up front
        x = x.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds = model(x)[0].squeeze(1).cpu()  # main head only
    return preds if hand.ndim == 3 else preds[0]

def get_ltc_embedding_fuzzy2(model: HandLTCfuzzy2, hands: torch.Tensor, device=None) -> torch.Tensor:
    """Return the three-stream (card / exact-shape / sorted-shape) embedding activations for a batch of hands.

    Unaffected by dropout_rate: card_emb/shape_emb/sorted_shape_emb are read directly, not full_branch.
    """
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


def hands_similarity0(query: torch.Tensor, hand_set: torch.Tensor, model: HandLTC) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sorted_indices, scores) of hand_set ranked by cosine similarity to query, descending."""
    q_emb = get_ltc_embedding(model, query.unsqueeze(0))           # (1, emb_dim)
    s_emb = get_ltc_embedding(model, hand_set)                     # (N, emb_dim)
    scores = torch.nn.functional.cosine_similarity(q_emb, s_emb)  # (N,)
    sorted_idx = scores.argsort(descending=True)
    return sorted_idx, scores[sorted_idx]


def hands_similarity(
    query: torch.Tensor,
    hand_set: torch.Tensor,
    model: HandLTC,
    fuzzy: str = "none",   # "none" | "multislice" | "topk", "ltc"
    n_slices: int = 8,     # multislice: number of random subsets to average
    slice_size: int = 8,   # multislice: embedding dims per subset
    top_k_dims: int = 8,   # topk: keep K highest-variance dims only
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sorted_indices, scores) of hand_set ranked by cosine similarity to query, descending."""
    q_emb = get_ltc_embedding(model, query.unsqueeze(0))  # (1, D)
    s_emb = get_ltc_embedding(model, hand_set)            # (N, D)
    emb_dim = q_emb.shape[1]
    if fuzzy == "ltc":
        # select dims most used to predict LTC
        dims = model.aux_head.weight.detach().cpu().abs().squeeze(0).argsort(descending=True)[:top_k_dims]
        scores = torch.nn.functional.cosine_similarity(q_emb[:, dims], s_emb[:, dims])

    elif fuzzy == "multislice":
        rng = torch.Generator().manual_seed(seed)
        scores = torch.zeros(len(s_emb))
        for _ in range(n_slices):
            dims = torch.randperm(emb_dim, generator=rng)[:slice_size]
            scores += torch.nn.functional.cosine_similarity(q_emb[:, dims], s_emb[:, dims])
        scores /= n_slices
    elif fuzzy == "topk":
        # select dims with highest variance across the hand set
        dims = s_emb.var(dim=0).argsort(descending=True)[:top_k_dims]
        scores = torch.nn.functional.cosine_similarity(q_emb[:, dims], s_emb[:, dims])
    else:
        scores = torch.nn.functional.cosine_similarity(q_emb, s_emb)

    sorted_idx = scores.argsort(descending=True)
    return sorted_idx, scores[sorted_idx]


def hand_from_indices(indices, device):
    out = torch.zeros((4, 13), dtype=torch.float32, device=device)
    for idx in indices:
        suit = idx // 13
        rank = idx % 13
        out[suit, rank] = 1.0
    return out



def convert_hand_indices_to_tensor(indices, device) -> torch.Tensor:
    return hand_from_indices(indices, device)



def hands_similarity_new(
    query: torch.Tensor,
    hand_set: torch.Tensor,
    model: HandLTC,
    shape_param: float = 1.0,  # weight on shape_emb cosine (0 = ignore shape)
    ltc_param: float = 1.0,    # weight on card_emb cosine  (0 = ignore cards)
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
    """Return combined ranking and LTC-only similarity scores.

    The combined score uses cosine similarity of the card and shape embedding
    halves. The LTC-only score compares the scalar predictions from
    ``model.aux_head``; higher values mean closer predicted LTC values.
    """
    q_emb = get_ltc_embedding(model, query.unsqueeze(0))  # (1, D)
    s_emb = get_ltc_embedding(model, hand_set)            # (N, D)
    half = q_emb.shape[1] // 2
    just_ltc = q_emb.shape[1] // 3
    tiny_ltc = q_emb.shape[1] // 5
    card_scores  = torch.nn.functional.cosine_similarity(q_emb[:, :tiny_ltc], s_emb[:, :tiny_ltc])
    shape_scores = torch.nn.functional.cosine_similarity(q_emb[:, half:], s_emb[:, half:])
    #unsorted_shape_scores = torch.nn.functional.cosine_similarity(q_emb[])
    scores = ltc_param * card_scores + shape_param * shape_scores
    #normalize scores to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    sorted_idx = scores.argsort(descending=True)

    return sorted_idx, scores[sorted_idx]


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


def decode_hand(emb: torch.Tensor, hand_set: torch.Tensor, model: HandLTC, n: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the n hands from hand_set whose embeddings are closest to the given embedding vector."""
    s_emb  = get_ltc_embedding(model, hand_set)                          # (M, D)
    q_emb  = emb.unsqueeze(0).to(torch.float32)                          # (1, D)
    scores = torch.nn.functional.cosine_similarity(q_emb, s_emb)        # (M,)
    top_idx = scores.argsort(descending=True)[:n]
    return hand_set[top_idx], scores[top_idx]


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


def create_similar_hands(query, query2,hands, ltc_model2):
    ###############
    # new tests ###
    ###############
    #train_ltc_fuzzy()
    #ltc_model2 = load_ltc_fuzzy2_model('ltc_fuzzy2_model.pth')
    ltc_values = [calc_losers(hands[i])[0] for i in range(len(hands))]
    pred_ltc = [predict_ltc_fuzzy2(ltc_model2, hands[i].unsqueeze(0)).item() for i in range(len(hands))]
    sorted_idx = torch.tensor(pred_ltc).argsort(descending=False)
    for i in range(20):
        print_hand(hands[sorted_idx[i]])
        print(f"LTC value: {ltc_values[sorted_idx[i]]} | Predicted LTC: {pred_ltc[sorted_idx[i]]:.2f}")

    hands_set = generate_smart_hands(query_s = query.unsqueeze(0), query_n = query2.unsqueeze(0), hand_set = hands, model = ltc_model2, ltc_param = 1.0, shape_param = 0.04, sorted_shape_param = 0.05, num_pairs = 10)
    print_pair2(hands_set)


def similar_feature_hands(
    hand: torch.Tensor,
    model: HandLTC,
    hands: torch.Tensor,
    ltc_weight: float = 1.0,
    shape_weight: float = 1.0,
    top_n: int = 10,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top_n hands whose embeddings best match `hand`'s embedding after
    fuzzing (adding noise to) whichever half — card/LTC or shape — has the lower weight.
    """
    q_emb = get_ltc_embedding(model, hand.unsqueeze(0))  # (1, D)
    s_emb = get_ltc_embedding(model, hands)              # (N, D)
    half  = q_emb.shape[1] // 2
    ltc_dims, shape_dims = slice(0, half), slice(half, q_emb.shape[1])

    total  = ltc_weight + shape_weight
    w_ltc  = ltc_weight / total
    w_shape = 1.0 - w_ltc
    noise_dims = shape_dims if w_shape < w_ltc else ltc_dims
    lower_w    = min(w_ltc, w_shape)
    std        = noise_scale * (1.0 - 2.0 * lower_w)  # 0 at equal weights, noise_scale at lower_w=0

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(q_emb[:, noise_dims].shape, generator=generator) * std
    q_emb_noised = q_emb.clone()
    q_emb_noised[:, noise_dims] += noise

    scores = torch.nn.functional.cosine_similarity(q_emb_noised, s_emb)  # (N,)
    top_idx = scores.argsort(descending=True)[:top_n]
    return hands[top_idx], scores[top_idx]


def plot_hand_tsne(
    query: torch.Tensor,
    hand_set: torch.Tensor,
    model: HandLTC,
    n_sample: int = 500,
) -> None:
    """t-SNE scatter of hand embeddings: left=t-SNE space, right=HCP vs shape, both colored by HCP."""
    from sklearn.manifold import TSNE

    if len(hand_set) > n_sample:
        idx = torch.randperm(len(hand_set))[:n_sample]
        hand_set = hand_set[idx]

    # prepend query so index 0 = query throughout
    all_hands = torch.cat([query.unsqueeze(0).to(torch.float32), hand_set.to(torch.float32)])
    all_emb   = get_ltc_embedding(model, all_hands).numpy()  # (N+1, emb_dim)

    hcp_vals   = np.array([calc_hcp(h) for h in all_hands])
    shape_vals = np.array([int(h.sum(dim=1).max().item()) for h in all_hands])  # longest suit

    sim_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(all_emb[:1]), torch.tensor(all_emb[1:])
    ).numpy()  # (N,)

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=min(30, len(all_emb) - 1)).fit_transform(all_emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # t-SNE subplot — embedding clusters
    sc1 = ax1.scatter(coords[1:, 0], coords[1:, 1], c=hcp_vals[1:],
                      cmap="plasma", alpha=0.6, s=18)
    ax1.scatter(coords[0, 0], coords[0, 1], c="red", marker="*", s=250,
                zorder=5, label="Query")
    fig.colorbar(sc1, ax=ax1, label="HCP")
    ax1.set(title="t-SNE (embedding space)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax1.legend()

    # HCP vs longest-suit subplot — point size encodes similarity to query
    pt_sizes = sim_scores * 60 + 5
    sc2 = ax2.scatter(hcp_vals[1:], shape_vals[1:], c=hcp_vals[1:],
                      cmap="plasma", alpha=0.55, s=pt_sizes)
    ax2.scatter(hcp_vals[0], shape_vals[0], c="red", marker="*", s=250,
                zorder=5, label="Query")
    fig.colorbar(sc2, ax=ax2, label="HCP")
    ax2.set(title="HCP vs Longest Suit  (size = similarity)",
            xlabel="HCP", ylabel="Longest Suit")
    ax2.legend()

    fig.tight_layout()
    plt.show()

#ltc_model = load_ltc_model("ltc_model.pth")
#hands = generate_training_set(400000).to(torch.float32)
#plot_hand_tsne(hands[0], hands[1:], ltc_model)


def predict(model, n = 100, device=None):
    """Generate n random hands and return (hands, predicted_labels, probabilities)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hands = generate_training_set(n).to(torch.float32).to(device)
    with torch.no_grad():
        logits = model(hands).squeeze(1)          # (N,)
        probs  = torch.sigmoid(logits)            # (N,)
        preds  = (logits > 0).long()              # (N,) — 1 = predicted 1NT
    return hands.cpu(), preds.cpu(), probs.cpu()

#not used
def get_embedding(model, hands, device=None):
    """Get embedding for a batch of hands."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hands = hands.to(torch.float32).to(device)
    with torch.no_grad():
        
        suit_lengths = hands.sum(dim=2)                        # (B, 4)
        high_cards   = hands[:, :, :].reshape(hands.size(0), -1) #not just x[:,:,9:] for hi cards (B, 16)  J Q K A

        emb = model.embedding(high_cards)
    return emb.cpu()


def embedding_shape_factors():
    # shape_dims: embedding dimensions most used to predict suit lengths
    shape_dims = ltc_model.shape_aux_head.weight.detach().norm(dim=0).argsort(descending=True)

    # ltc_dims: embedding dimensions most used to predict LTC
    ltc_dims = ltc_model.aux_head.weight.detach().abs().squeeze(0).argsort(descending=True)

    print("Top shape dims:", shape_dims[:6].tolist())
    print("Top LTC dims:  ", ltc_dims[:6].tolist())


"""
ltc_model = load_ltc_model("ltc_model.pth")
preds = predict(ltc_model, hands)  # use the generic predict function
indices, scores = hands_similarity(hand, hands, ltc_model)
print_hand(hands[indices[:10]])  
print(scores[:10])

"""

if __name__ == "__main__":
    ltc_model = train_ltc()
    
    """
    ltc_model = load_ltc_model("ltc_model.pth") 
    hands = generate_training_set(500_000).to(torch.float32)
    test_hands = generate_training_set(1000).to(torch.float32)

    preds = predict_ltc(ltc_model, test_hands)
    for i in range(15): #len(test_hands)
        true_ltc = calc_losers(test_hands[i])[0]
        print_hand(test_hands[i])
        print(f"Predicted LTC: {preds[i].item():.2f} | True LTC: {true_ltc:.2f}\n")

    losers_vector =[calc_losers(hnd)[0] for hnd in hands]
    losers_sorted = np.argsort(losers_vector, descending=False)
    for i in range(20):    
        print_hand(hands[losers_sorted[i]])
        print(f"number of losers: {losers_vector[losers_sorted[i]]}")

    hand = hands[losers_sorted[4343]]
    print_hand(hand)
    
    # standard
    indices, scores = hands_similarity0(hand, hands, ltc_model)
    
    # fuzzy: average over 8 random 8-dim subsets (finds broadly similar hands)
    indices, scores = hands_similarity(hand, hands, ltc_model, fuzzy="multislice", n_slices=7, slice_size=3)
    
    # topk: compare only on the 6 most discriminative embedding dims
    indices, scores = hands_similarity(hand, hands, ltc_model, fuzzy="topk", top_k_dims=3)

    # ltc: take only ltc dims without shape considerations
    indices, scores = hands_similarity(hand, hands, ltc_model, fuzzy="ltc", top_k_dims=10)

    # pure shape similarity (ignore cards)
    indices, scores = hands_similarity_new(hand, hands, ltc_model, shape_param=10, ltc_param=2)

    # pure card/LTC similarity (ignore shape)
    indices, scores = hands_similarity_new(hand, hands, ltc_model, shape_param=0, ltc_param=10)

    # balanced
    indices, scores = hands_similarity_new(hand, hands, ltc_model, shape_param=10, ltc_param=0)

    most_similar_indices = indices[:24]
    
    ltc_values = [calc_losers(hands[i])[0] for i in most_similar_indices]
    pred_ltc = [predict_ltc(model, hands[i].unsqueeze(0)).item() for i in most_similar_indices]
    for i in range(len(most_similar_indices)):
        print_hand(hands[most_similar_indices[i]])
        print(f"LTC value: {ltc_values[i]} | Predicted LTC: {pred_ltc[i]:.2f} | score: {scores[most_similar_indices[i]]:.2f}")
    
    sorted_ltc_only = ltc_only.argsort(descending=True)
    ltc_only_vec = sorted_ltc_only[:30]
    ltc_values = [calc_losers(hands[i])[0] for i in ltc_only_vec]
    pred_ltc = [predict_ltc(ltc_model, hands[i].unsqueeze(0)).item() for i in ltc_only_vec]
    for i in range(len(ltc_only_vec)):
        print_hand(hands[ltc_only_vec[i]])
        print(f"LTC value: {ltc_values[i]} | Predicted LTC: {pred_ltc[i]:.2f} | score: {scores[most_similar_indices[i]]:.2f}")

        
    plot_hand_tsne(hand, hands, ltc_model)

    sorted_idx2, ltc_scores2 = hands_similarity_fuzzy(query, hands, ltc_model2, ltc_param=1, shape_param=0, sorted_shape_param=0.3)
    sorted_idx, ltc_scores = hands_similarity_fuzzy(query2, hands, model, ltc_param=1, shape_param=0, sorted_shape_param=0.3)
    most_similar_indices = sorted_idx[:8]
    most_similar_indices2 = sorted_idx2[:8]
    #create a pair out of hands in most_similar_indices and most_similar_indices2
    paired_hands = [(hands[most_similar_indices[i]], hands[most_similar_indices2[i]]) for i in range(min(len(most_similar_indices), len(most_similar_indices2)))]
    
    for i in range(len(most_similar_indices2)):
        print_hand(hands[most_similar_indices2[i]])
        print(f"LTC value: {ltc_values[most_similar_indices2[i]]} | Predicted LTC: {predict_ltc_fuzzy2(ltc_model2, hands[most_similar_indices2[i]].unsqueeze(0)).item():.2f} | score: {ltc_scores2[i]:.2f}")

    """
