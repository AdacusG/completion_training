#!/usr/bin/env python3
"""
Prepare a 2-character level dataset for train_completions.py.

Reads a plain-text file where each line is one sequence of format {input}={output},
splits it 90/10 into train and validation sets, builds a custom 2-character 
vocabulary (treating '=' and '\n' as separate tokens), and writes 
train.bin, val.bin, and meta.pkl to an output directory.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrepareConfig:
    input_file: Path = Path("input.txt")
    out_dir: Path = Path("data")
    train_split: float = 0.9
    pad_token: str = "_"        # Used to pad odd-length inputs/outputs


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_lines(path: Path) -> list[str]:
    """Read all lines from a text file, stripping trailing whitespace but keeping structure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Check the path and try again."
        )
    with path.open(encoding="utf-8") as fh:
        # Strip trailing spaces/newlines so we can manually handle '\n' token safely
        return [line.strip() for line in fh if line.strip()]


def tokenize_line(line: str, pad_token: str) -> list[str]:
    """
    Splits a line into custom 2-character tokens, keeping '=' and '\\n' separate.
    Pads odd-lengthed input or output parts with pad_token.
    """
    if '=' not in line:
        raise ValueError(f"Line missing '=' separator: {line}")
        
    inp, out = line.split('=', 1)
    tokens = []
    
    # Process Input Side
    if len(inp) % 2 != 0:
        inp += pad_token
    for i in range(0, len(inp), 2):
        tokens.append(inp[i:i+2])
        
    # Separator
    tokens.append('=')
    
    # Process Output Side
    if len(out) % 2 != 0:
        out += pad_token
    for i in range(0, len(out), 2):
        tokens.append(out[i:i+2])
        
    # End of line token
    tokens.append('\n')
    
    return tokens


def build_vocab(tokenized_lines: list[list[str]], pad_token: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build token-to-index and index-to-character lookup tables from tokenized lists.
    """
    # Gather all unique tokens across all sequences
    unique_tokens = set()
    for tokens in tokenized_lines:
        unique_tokens.update(tokens)
        
    # Sort them for consistency, pushing special tokens or structural ones wherever you like
    sorted_tokens = sorted(list(unique_tokens))
    
    # Ensure pad token as an independent string isn't floating around weirdly 
    # if it's already embedded inside 2-character tokens (like 'a_').
    if pad_token in sorted_tokens:
        sorted_tokens.remove(pad_token)
    sorted_tokens.append(pad_token)

    stoi = {tok: i for i, tok in enumerate(sorted_tokens)}
    itos = {i: tok for i, tok in enumerate(sorted_tokens)}
    return stoi, itos


def encode(tokenized_lines: list[list[str]], stoi: dict[str, int]) -> np.ndarray:
    """Encode custom tokens into a continuous uint16 array of token IDs."""
    flat_ids = []
    for tokens in tokenized_lines:
        for token in tokens:
            flat_ids.append(stoi[token])
    return np.array(flat_ids, dtype=np.uint16)


def save_outputs(
    out_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    """Write train.bin, val.bin, and meta.pkl to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}
    with (out_dir / "meta.pkl").open("wb") as fh:
        pickle.dump(meta, fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("input.txt"),
        help="Path to the input text file (default: input.txt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write train.bin, val.bin, meta.pkl (default: data/)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of lines used for training (default: 0.9)",
    )
    args = parser.parse_args()

    cfg = PrepareConfig()
    cfg.input_file = args.input_file
    cfg.out_dir = args.out_dir
    cfg.train_split = args.train_split
    return cfg


def main() -> None:
    cfg = parse_args()

    lines = load_lines(cfg.input_file)
    print(f"Lines loaded    : {len(lines)}")
    
    # Tokenize every line individual first
    tokenized_lines = [tokenize_line(line, cfg.pad_token) for line in lines]

    # Split dataset based on lines
    split = int(len(tokenized_lines) * cfg.train_split)
    train_lines = tokenized_lines[:split]
    val_lines = tokenized_lines[split:]

    # Build vocabulary using all tokens
    stoi, itos = build_vocab(tokenized_lines, cfg.pad_token)
    print(f"Vocabulary size : {len(stoi)}")

    # --- ADD THIS TO DUMP THE VOCAB ---
    import pprint
    print("\n--- Vocabulary Mapping (stoi) ---")
    pprint.pprint(stoi)
    print("----------------------------------\n")
    # ----------------------------------
    
    # Flatten and convert tokens to integer IDs
    train_ids = encode(train_lines, stoi)
    val_ids = encode(val_lines, stoi)
    print(f"Train tokens    : {len(train_ids)}")
    print(f"Val tokens      : {len(val_ids)}")

    save_outputs(cfg.out_dir, train_ids, val_ids, stoi, itos)
    print(f"\nFiles written to '{cfg.out_dir}/'")


if __name__ == "__main__":
    main()