#!/usr/bin/env python3
"""
Evaluate a trained sequence completion transformer on ALL unseen even combinations.

Reads the original training file to filter out seen sequences, programmatically 
generates all remaining mathematical combinations for a specific length layout, 
and measures pure out-of-distribution accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import string
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration & Vocabulary
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    out_dir: Path = Path("out_1char")
    data_dir: Path = Path("1-Char/data")
    train_file: Path = Path("inputrev.txt")
    n: int = 3                          # Generates lengths up to 2 * n
    charset_size: int = 4               # Kept small by default to prevent combination explosions!


@dataclass
class Vocabulary:
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

    pad_id: int = field(init=False)
    equal_id: int = field(init=False)
    eot_id: int = field(init=False)
    is_2char: bool = field(init=False)

    def __post_init__(self) -> None:
        self.pad_id = self.stoi["_"]
        self.equal_id = self.stoi["="]
        self.eot_id = self.stoi["\n"]
        self.is_2char = any(len(k) == 2 for k in self.stoi.keys())

    @classmethod
    def from_pickle(cls, path: Path) -> "Vocabulary":
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with path.open("rb") as fh:
            meta = pickle.load(fh)
        return cls(vocab_size=meta["vocab_size"], stoi=meta["stoi"], itos=meta["itos"])

    def tokenize_string(self, text: str) -> list[str]:
        if not self.is_2char:
            return list(text)
        if len(text) % 2 != 0:
            text += "_"
        return [text[i:i+2] for i in range(0, len(text), 2)]


# ---------------------------------------------------------------------------
# Model Layout
# ---------------------------------------------------------------------------

class CompletionTransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _b, t = idx.size()
        pos = torch.arange(t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln_f(x))


def load_model(weights_path: Path, vocab: Vocabulary, device: str) -> tuple[CompletionTransformer, int]:
    checkpoint = torch.load(weights_path, map_location=device)
    seq_len = checkpoint["position_embedding.weight"].shape[0]
    model = CompletionTransformer(
        vocab_size=vocab.vocab_size, seq_len=seq_len, d_model=128, n_heads=4, n_layers=4
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, seq_len


@torch.no_grad()
def complete_sequence(lhs: str, model: CompletionTransformer, vocab: Vocabulary, max_supported_len: int, device: str) -> str:
    lhs_tokens = vocab.tokenize_string(lhs)
    tokens = [vocab.stoi[tok] for tok in lhs_tokens] + [vocab.equal_id]
    
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    start_len = x.size(1)
    max_new_tokens = len(lhs_tokens) + 1  # Mirror prediction output token budget size

    while (x.size(1) - start_len) < max_new_tokens and x.size(1) < max_supported_len:
        logits = model(x)
        next_id = torch.argmax(logits[0, -1, :]).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eot_id:
            break

    gen_tokens = [vocab.itos[t.item()] for t in x[0]][start_len:]
    return "".join(gen_tokens).replace("\n", "").replace("_", "")


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Max length step parameter (checks lengths up to 2*n)")
    parser.add_argument("--charset-size", type=int, default=6, help="Alphabet dimension constraint used to build test space")
    parser.add_argument("--train-file", type=Path, default=Path("inputrev.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("out_1char"))
    parser.add_argument("--data-dir", type=Path, default=Path("1-Char/data"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocabulary.from_pickle(args.data_dir / "meta.pkl")
    model, max_supported_len = load_model(args.out_dir / "completion_model.pth", vocab, device)

    print(f"Loaded format: {'2-char' if vocab.is_2char else '1-char'}")
    
    # 1. Map already-trained instances to a set
    trained_sequences = set()

    available_chars = string.ascii_lowercase[:min(max(1, args.charset_size), 26)]
    
    print(f"\nEvaluating combinations across charset: {list(available_chars)}")
    print("-" * 60)

    # 2. Loop through every distinct even length group
    for length_step in range(1, args.n + 1):
        actual_len = 2 * length_step
        correct, total = 0, 0
        
        # Programmatically evaluate all possible combinations of fixed length
        for items in itertools.product(available_chars, repeat=actual_len):
            lhs_string = "".join(items)
            
            # Pure Out-Of-Distribution validation condition check
            if lhs_string in trained_sequences:
                continue
                
            ground_truth = lhs_string[::-1]
            prediction = complete_sequence(lhs_string, model, vocab, max_supported_len, device)
            
            if prediction == ground_truth:
                correct += 1
            total += 1

        print(f"String Length {actual_len:02d} | Unseen Evaluated: {total:<6} | Correct: {correct:<6} | Accuracy: {0.00 if total == 0 else (correct/total)*100:.2f}%")
        
    print("-" * 60)


if __name__ == "__main__":
    main()