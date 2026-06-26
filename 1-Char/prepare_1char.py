#!/usr/bin/env python3
"""
Prepare a character-level dataset for train_completions.py.

Reads a plain-text file where each line is one sequence, splits it 90/10 into
train and validation sets, builds a character vocabulary (plus a pad token '_'),
and writes train.bin, val.bin, and meta.pkl to an output directory.

Usage:
    python prepare_1char.py                          # reads input.txt, writes data/
    python prepare_1char.py path/to/input.txt
    python prepare_1char.py input.txt --out-dir 1-Char/data
    python prepare_1char.py input.txt --train-split 0.95
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
    pad_token: str = "_"        # Must match the pad token expected by the trainer


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_lines(path: Path) -> list[str]:
    """Read all lines from a text file, preserving newline characters."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Check the path and try again."
        )
    with path.open(encoding="utf-8") as fh:
        return fh.readlines()


def build_vocab(text: str, pad_token: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build character-to-index and index-to-character lookup tables.

    The pad token is appended after sorting so its ID is always at the end
    of the vocabulary, keeping it stable across datasets that share the same
    character set.
    """
    chars = sorted(set(text))
    if pad_token in chars:
        chars.remove(pad_token)
    chars.append(pad_token)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Encode a string as a uint16 array of token IDs."""
    return np.array([stoi[c] for c in text], dtype=np.uint16)


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

    split = int(len(lines) * cfg.train_split)
    train_str = "".join(lines[:split])
    val_str = "".join(lines[split:])
    full_str = train_str + val_str

    stoi, itos = build_vocab(full_str, cfg.pad_token)
    print(f"Vocabulary size : {len(stoi)}")

    train_ids = encode(train_str, stoi)
    val_ids = encode(val_str, stoi)
    print(f"Train tokens    : {len(train_ids)}")
    print(f"Val tokens      : {len(val_ids)}")

    save_outputs(cfg.out_dir, train_ids, val_ids, stoi, itos)
    print(f"\nFiles written to '{cfg.out_dir}/'")


if __name__ == "__main__":
    main()