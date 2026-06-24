#!/usr/bin/env python3
import os
import pickle
import numpy as np
import argparse

def tokenize_line_with_exceptions(line):
    """Splits a string into 2-character pairs, but keeps '=' and '\n' 

    as individual 1-character tokens.
    """
    tokens = []
    i = 0
    n = len(line)
    
    while i < n:
        # If the current character is an exception, isolate it
        if line[i] in ('=', '\n'):
            tokens.append(line[i])
            i += 1
        # If the NEXT character is an exception, we can only take 1 character 
        # for the current token to prevent grouping them together
        elif i + 1 < n and line[i+1] in ('=', '\n'):
            tokens.append(line[i])
            i += 1
        # Otherwise, safely grab a 2-character pair
        elif i + 1 < n:
            tokens.append(line[i:i+2])
            i += 2
        # Handle trailing odd character at the end of a line if necessary
        else:
            tokens.append(line[i])
            i += 1
            
    return tokens

def main():
    parser = argparse.ArgumentParser(description="Prepare text dataset with protected 2-char tokenization.")
    parser.add_argument(
        "input_file_path", 
        nargs="?", 
        default="input.txt", 
        help="Path to the input text file (default: input.txt)"
    )
    args = parser.parse_args()

    input_file_path = args.input_file_path
    if not os.path.exists(input_file_path):
        print(f"Input file '{input_file_path}' not found. Make sure path is correct.")
        return
    
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        
    print("Total lines in dataset:", len(data))
    
    # Split the data list clean at 90%
    num_train_lines = int(len(data) * 0.9)
    train_lines = data[:num_train_lines]
    val_lines = data[num_train_lines:]
    
    # Tokenize lines individually using our conditional exception rules
    print("Tokenizing data (protecting '=' and '\\n')...")
    train_tokens_nested = [tokenize_line_with_exceptions(line) for line in train_lines]
    val_tokens_nested = [tokenize_line_with_exceptions(line) for line in val_lines]
    
    # Flatten token lists
    train_tokens = [tok for line in train_tokens_nested for tok in line]
    val_tokens = [tok for line in val_tokens_nested for tok in line]
    full_tokens = train_tokens + val_tokens
    
    # Generate vocab using unique tokens found
    unique_tokens = sorted(list(set(full_tokens)))
    
    # Explicitly ensure our special tokens are in the vocabulary list
    for special_tok in ['_', '=', '\n']:
        if special_tok not in unique_tokens:
            unique_tokens.append(special_tok)
        
    vocab_size = len(unique_tokens)
    print("Protected Vocab size:", vocab_size)
    
    # Create lookup dictionaries
    stoi = {tok: i for i, tok in enumerate(unique_tokens)}
    itos = {i: tok for i, tok in enumerate(unique_tokens)}

    # Map tokens to their corresponding integer IDs
    train_data = np.array([stoi[tok] for tok in train_tokens], dtype=np.uint16)
    val_data = np.array([stoi[tok] for tok in val_tokens], dtype=np.uint16)

    print("Train tokens count:", len(train_data))
    print("Validation tokens count:", len(val_data))

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