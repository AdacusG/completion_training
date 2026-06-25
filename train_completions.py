#!/usr/bin/env python3
"""
Train a GPT-style decoder-only transformer on character-level completion sequences.

Data format: each line is "<pad><input>=<output>\n", tokenised and stored as
uint16 token IDs in train.bin / val.bin alongside a meta.pkl vocabulary file.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_dir: Path = Path("1-Char/data")
    out_dir: Path = Path("out_1char")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    embedding_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4

    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 50  # batches between progress prints


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

@dataclass
class Vocabulary:
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

    # Special token IDs resolved after loading
    pad_id: int = field(init=False)
    equal_id: int = field(init=False)
    eot_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.pad_id = self.stoi["_"]
        self.equal_id = self.stoi["="]
        self.eot_id = self.stoi["\n"]

    @classmethod
    def from_pickle(cls, path: Path) -> "Vocabulary":
        if not path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {path}. Run prepare.py first."
            )
        with path.open("rb") as fh:
            meta = pickle.load(fh)
        return cls(
            vocab_size=meta["vocab_size"],
            stoi=meta["stoi"],
            itos=meta["itos"],
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Loads tokenised completion sequences from a binary file of uint16 token IDs.

    Each line has the form:  [pad] [input chars] [=] [output chars] [\n]
    The model is trained with a causal LM objective, but loss is only computed
    on the output portion (tokens after and including the '=' position).
    """

    def __init__(self, bin_path: Path, vocab: Vocabulary) -> None:
        raw_tokens = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
        self.sequences = self._split_into_lines(raw_tokens, vocab.eot_id)
        self.max_len = max(len(s) for s in self.sequences)
        self.vocab = vocab

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_lines(
        tokens: np.ndarray, eot_id: int
    ) -> list[torch.Tensor]:
        """Split a flat token array into per-line tensors (inclusive of \n)."""
        eot_indices = np.where(tokens == eot_id)[0]
        sequences: list[torch.Tensor] = []
        start = 0
        for end in eot_indices:
            sequences.append(torch.tensor(tokens[start : end + 1]))
            start = end + 1
        return sequences

    def _build_targets(
        self, seq: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the target tensor for one sequence.

        Tokens before (and not including) '=' are masked with -100 so the
        loss function ignores them.  Tokens from '=' onward are the shifted
        ground-truth labels.
        """
        y = torch.full_like(x, fill_value=-100)
        shifted = seq[1:].clone()

        eq_positions = (x == self.vocab.equal_id).nonzero(as_tuple=True)[0]
        if len(eq_positions) == 0:
            raise ValueError(
                "Sequence is missing the '=' delimiter. "
                "Check your data preparation step."
            )

        eq_pos = eq_positions[0].item()
        y[eq_pos:] = shifted[eq_pos:]
        return y

    def _pad_to_length(
        self, tensor: torch.Tensor, target_len: int, pad_value: int
    ) -> torch.Tensor:
        """Right-pad a 1-D tensor to `target_len` with `pad_value`."""
        shortfall = target_len - len(tensor)
        if shortfall <= 0:
            return tensor
        padding = torch.full((shortfall,), pad_value, dtype=torch.long)
        return torch.cat([tensor, padding])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        target_len = self.max_len - 1   # max input length after dropping last token

        x = seq[:-1].clone()            # input: drop the trailing \n
        y = self._build_targets(seq, x)

        x = self._pad_to_length(x, target_len, self.vocab.pad_id)
        y = self._pad_to_length(y, target_len, pad_value=-100)

        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CompletionTransformer(nn.Module):
    """
    Decoder-only transformer (GPT-style) for character-level sequence completion.

    Uses a causal attention mask so each position can only attend to earlier
    positions, preventing the model from "seeing" the answer during training.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _b, t = idx.size()
        pos = torch.arange(t, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.token_embedding(idx) + self.position_embedding(pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            t, device=idx.device
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        return self.lm_head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Training & validation steps
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    grad_clip: float,
    device: str,
    log_interval: int,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one full pass (train or eval) and return the mean loss."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = criterion(logits.view(-1, vocab_size), Y.view(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        f"Batch {batch_idx}/{len(loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("1-Char/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("out_1char"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.device = args.device
    cfg.embedding_dim = args.embedding_dim
    cfg.n_heads = args.n_heads
    cfg.n_layers = args.n_layers
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    return cfg


def main() -> None:
    cfg = parse_args()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # --- Vocabulary ---
    vocab = Vocabulary.from_pickle(cfg.data_dir / "meta.pkl")

    # --- Datasets & loaders ---
    train_dataset = SequenceDataset(cfg.data_dir / "train.bin", vocab)
    val_dataset = SequenceDataset(cfg.data_dir / "val.bin", vocab)

    seq_len = train_dataset.max_len
    print(f"Sequence length : {seq_len} tokens")
    print(f"Batch shape     : ({cfg.batch_size}, {seq_len - 1})")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    # --- Model, optimiser, loss ---
    model = CompletionTransformer(
        vocab_size=vocab.vocab_size,
        seq_len=seq_len,
        d_model=cfg.embedding_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(cfg.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Training on device: {cfg.device}")

    # --- Training loop ---
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            vocab_size=vocab.vocab_size,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
            device=cfg.device,
            log_interval=cfg.log_interval,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            vocab_size=vocab.vocab_size,
            criterion=criterion,
            optimizer=None,         # signals eval mode
            grad_clip=cfg.grad_clip,
            device=cfg.device,
            log_interval=cfg.log_interval,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )

        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"{'=' * 60}\n")

    # --- Persist weights ---
    weights_path = cfg.out_dir / "completion_model.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")


if __name__ == "__main__":
    main()