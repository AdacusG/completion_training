#!/usr/bin/env python3
"""
Evaluate the accuracy of a trained sequence completion transformer.

Loads a saved checkpoint from train_completions.py, runs greedy autoregressive
decoding on each line of an evaluation file, and reports overall accuracy.

Usage:
    python generate.py                          # uses defaults
    python generate.py path/to/eval.txt
    python generate.py --out-dir out_1char --data-dir 1-Char/data
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    out_dir: Path = Path("out_1char")
    data_dir: Path = Path("1-Char/data")
    eval_file: Path | None = None       # None → resolved to data_dir/input.txt
    max_new_tokens: int = 10


# ---------------------------------------------------------------------------
# Vocabulary
# (Mirrors train_completions.py — keep in sync if you change token symbols.)
# ---------------------------------------------------------------------------

@dataclass
class Vocabulary:
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

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
# Model
# (Must match the architecture used in train_completions.py exactly.)
# ---------------------------------------------------------------------------

class CompletionTransformer(nn.Module):
    """
    Decoder-only transformer (GPT-style) for character-level sequence completion.

    seq_len is inferred from the saved checkpoint's positional embedding matrix,
    so the model automatically supports whatever context length it was trained on.
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
# Inference
# ---------------------------------------------------------------------------

def load_model(
    weights_path: Path,
    vocab: Vocabulary,
    d_model: int,
    n_heads: int,
    n_layers: int,
    device: str,
) -> tuple[CompletionTransformer, int]:
    """
    Load saved weights and reconstruct the model.

    The maximum sequence length is read directly from the checkpoint's
    positional embedding shape so it matches training exactly.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    seq_len = checkpoint["position_embedding.weight"].shape[0]

    model = CompletionTransformer(
        vocab_size=vocab.vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, seq_len


def _normalise_prompt(raw_prompt: str) -> str:
    """Ensure the prompt ends with '=' and contains no RHS content."""
    if "=" in raw_prompt:
        return raw_prompt.split("=")[0] + "="
    return raw_prompt + "="


@torch.no_grad()
def complete_sequence(
    prompt: str,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int,
    max_supported_len: int,
    device: str,
) -> str:
    """
    Greedily decode tokens one at a time until EOT or the length budget is hit.

    Returns only the RHS (right-hand side of '='), stripped of padding and
    newline characters.
    """
    prompt = _normalise_prompt(prompt)
    tokens = [vocab.stoi[c] for c in prompt]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    start_len = x.size(1)

    while (x.size(1) - start_len) < max_new_tokens and x.size(1) < max_supported_len:
        logits = model(x)
        next_id = torch.argmax(logits[0, -1, :]).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eot_id:
            break

    chars = [vocab.itos[t.item()] for t in x[0]]
    if "=" in chars:
        rhs = chars[chars.index("=") + 1:]
        return "".join(rhs).replace("\n", "").replace("_", "")
    return "".join(chars)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    eval_file: Path,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int,
    max_supported_len: int,
    device: str,
) -> None:
    """Run greedy completion on every line in `eval_file` and report accuracy."""
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    print(f"Evaluating: {eval_file}\n")

    correct = 0
    total = 0

    with eval_file.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip().replace("_", "")
            if not line:
                continue
            if "=" not in line:
                print(f"  [SKIP] Line {line_num}: missing '=' — '{raw_line.strip()}'")
                continue

            lhs, ground_truth = line.split("=", maxsplit=1)
            prompt = lhs + "="
            prediction = complete_sequence(
                prompt=prompt,
                model=model,
                vocab=vocab,
                max_new_tokens=len(ground_truth) + 1,
                max_supported_len=max_supported_len,
                device=device,
            )

            if prediction == ground_truth:
                correct += 1
            else:
                print(
                    f"  [MISS] Prompt: {prompt:<6}  "
                    f"Expected: {ground_truth:<5}  Got: {prediction:<5}"
                )
            total += 1

    print(f"\n{'=' * 40}")
    print(f"ACCURACY REPORT — {eval_file.name}")
    print(f"  Lines processed : {total}")
    if total > 0:
        print(f"  Correct         : {correct}")
        print(f"  Accuracy        : {correct / total * 100:.2f}%")
    else:
        print("  No valid sequences were found.")
    print("=" * 40)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_file",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the evaluation file (default: <data-dir>/input.txt)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out_1char"))
    parser.add_argument("--data-dir", type=Path, default=Path("1-Char/data"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Upper bound on generated tokens per prompt (default: 10)",
    )
    args = parser.parse_args()

    cfg = EvalConfig()
    cfg.out_dir = args.out_dir
    cfg.data_dir = args.data_dir
    cfg.eval_file = args.eval_file
    cfg.max_new_tokens = args.max_new_tokens
    return cfg


def main() -> None:
    cfg = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_file = cfg.eval_file or (cfg.data_dir / "input.txt")

    vocab = Vocabulary.from_pickle(cfg.data_dir / "meta.pkl")
    model, max_supported_len = load_model(
        weights_path=cfg.out_dir / "completion_model.pth",
        vocab=vocab,
        d_model=128,
        n_heads=4,
        n_layers=4,
        device=device,
    )
    print(f"Checkpoint loaded  (seq_len={max_supported_len}, device={device})\n")

    evaluate(
        eval_file=eval_file,
        model=model,
        vocab=vocab,
        max_new_tokens=cfg.max_new_tokens,
        max_supported_len=max_supported_len,
        device=device,
    )


if __name__ == "__main__":
    main()