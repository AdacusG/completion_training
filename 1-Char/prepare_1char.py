#! /usr/bin/env python3
import os
import pickle
import numpy as np
import sys

def main():
    input_file_path = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    if not os.path.exists(input_file_path):
        print(f"Input file '{input_file_path}' not found. Make sure path is correct.")
        return
    
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        
    print("Total lines in dataset:", len(data))
    
    # Split the data list clean at 90%
    # Can be done better with dataloader
    num_train_lines = int(len(data) * 0.9)
    train_lines = data[:num_train_lines]
    val_lines = data[num_train_lines:]
    
    # Join them back into continuous strings to extract tokens
    train_data_str = "".join(train_lines)
    val_data_str = "".join(val_lines)
    full_data_str = train_data_str + val_data_str
    
    # Generate vocab using the full text pool
    chars = sorted(list(set(full_data_str)))
    # Add the padding character '_' to the vocabulary for use in train_completions.py
    chars.append('_')
    vocab_size = len(chars)
    print("Vocab size:", vocab_size)
    
    # Create lookup dictionaries
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Split into train and validation sets
    train_data = np.array([stoi[c] for c in train_data_str], dtype=np.uint16)
    val_data = np.array([stoi[c] for c in val_data_str], dtype=np.uint16)

    print("Train data length:", len(train_data))
    print("Validation data length:", len(val_data))

    # Save the data and metadata
    os.makedirs("data", exist_ok=True)
    train_data.tofile("data/train.bin")
    val_data.tofile("data/val.bin")
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open("data/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Data preparation complete. Files saved in 'data/' directory.")

if __name__ == "__main__":
    main()