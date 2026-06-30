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
import sys
import time
import csv

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
    """

    def __init__(self, bin_path: Path, vocab: Vocabulary) -> None:
        raw_tokens = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
        self.sequences = self._split_into_lines(raw_tokens, vocab.eot_id)
        self.max_len = max(len(s) for s in self.sequences)
        self.vocab = vocab

    @staticmethod
    def _split_into_lines(
        tokens: np.ndarray, eot_id: int
    ) -> list[torch.Tensor]:
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
        y = torch.full_like(x, fill_value=-100)
        shifted = seq[1:].clone()

        eq_positions = (x == self.vocab.equal_id).nonzero(as_tuple=True)[0]
        if len(eq_positions) == 0:
            raise ValueError("Sequence is missing the '=' delimiter.")

        eq_pos = eq_positions[0].item()
        y[eq_pos:] = shifted[eq_pos:]
        return y

    def _pad_to_length(
        self, tensor: torch.Tensor, target_len: int, pad_value: int
    ) -> torch.Tensor:
        shortfall = target_len - len(tensor)
        if shortfall <= 0:
            return tensor
        padding = torch.full((shortfall,), pad_value, dtype=torch.long)
        return torch.cat([tensor, padding])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        target_len = self.max_len - 1   

        x = seq[:-1].clone()            
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
    parser.add_argument("config_file", type=str, nargs="?", default=None,
                        help="Optional path to a nanoGPT-style python config file")
    parser.add_argument("--data-dir", type=Path, default=Path("1-Char/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("out_1char"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = TrainConfig()
    
    if args.config_file and args.config_file.endswith('.py'):
        print(f"Overriding config using nanoGPT-style file: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_code = f.read()
        
        local_namespace = {}
        exec(config_code, {}, local_namespace)
        
        for key, value in local_namespace.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                
        if isinstance(cfg.data_dir, str):
            cfg.data_dir = Path(cfg.data_dir)
        if isinstance(cfg.out_dir, str):
            cfg.out_dir = Path(cfg.out_dir)
    
    if "--data-dir" in sys.argv: cfg.data_dir = args.data_dir
    if "--out-dir" in sys.argv: cfg.out_dir = args.out_dir
    if "--device" in sys.argv: cfg.device = args.device
    if "--embedding-dim" in sys.argv: cfg.embedding_dim = args.embedding_dim
    if "--n-heads" in sys.argv: cfg.n_heads = args.n_heads
    if "--n-layers" in sys.argv: cfg.n_layers = args.n_layers
    if "--batch-size" in sys.argv: cfg.batch_size = args.batch_size
    if "--epochs" in sys.argv: cfg.epochs = args.epochs
    if "--lr" in sys.argv: cfg.lr = args.lr
        
    return cfg


def main() -> None:
    cfg = parse_args()
    
    # Track total runtime
    total_start_time = time.perf_counter()

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

    # Calculate exact parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Model Parameters: {param_count:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Training on device: {cfg.device}")

    # Metrics collections for charting
    history_epochs = []
    history_train_loss = []
    history_val_loss = []

    # CSV setup
    csv_path = cfg.out_dir / "metrics.csv"
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Training Time (sec)", "Param Count"])

    # --- Training loop ---
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.perf_counter()
        
        train_loss = run_epoch(
            model=model, loader=train_loader, vocab_size=vocab.vocab_size,
            criterion=criterion, optimizer=optimizer, grad_clip=cfg.grad_clip,
            device=cfg.device, log_interval=cfg.log_interval, epoch=epoch, total_epochs=cfg.epochs,
        )
        
        epoch_time = time.perf_counter() - start_time
        
        val_loss = run_epoch(
            model=model, loader=val_loader, vocab_size=vocab.vocab_size,
            criterion=criterion, optimizer=None, grad_clip=cfg.grad_clip,
            device=cfg.device, log_interval=cfg.log_interval, epoch=epoch, total_epochs=cfg.epochs,
        )

        # Track history data for every epoch to ensure smooth curves
        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch} SUMMARY | Time: {epoch_time:.2f}s")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"{'=' * 60}\n")

        # Write data row ONLY on 10-epoch intervals (or the absolute last epoch)
        if epoch % 10 == 0 or epoch == cfg.epochs:
            csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{epoch_time:.2f}", param_count])
            csv_file.flush() # force write to disk safely

    csv_file.close()
    print(f"Metrics table log updated successfully at: {csv_path}")

    # --- Generate Loss Curve Plot ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history_epochs, history_train_loss, label="Train Loss", color="blue", linewidth=2)
        plt.plot(history_epochs, history_val_loss, label="Val Loss", color="orange", linewidth=2)
        plt.title("Loss Over Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        
        plot_path = cfg.out_dir / "loss_chart.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Loss plot chart saved successfully at: {plot_path}")
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot layout creation.")

    # --- Persist weights ---
    weights_path = cfg.out_dir / "completion_model.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    # --- Final Statistics ---
    total_runtime = time.perf_counter() - total_start_time
    final_train_loss = history_train_loss[-1]
    final_val_loss = history_val_loss[-1]
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total Runtime    : {total_runtime:.2f} seconds")
    print(f"  Final Train Loss : {final_train_loss:.4f}")
    print(f"  Final Val Loss   : {final_val_loss:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()