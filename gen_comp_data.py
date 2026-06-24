#!/usr/bin/env python3
import random
import string
import argparse

def generate_all_unique_strings(n, charset):
    """Generates every possible unique string combination up to length n."""
    # Start with an empty list and recursively build strings up to length n
    pool = []
    def build(current_str):
        if len(current_str) > 0:
            pool.append(current_str)
        if len(current_str) == n:
            return
        for char in charset:
            build(current_str + char)
            
    build("")
    return pool

def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic 1-to-1 mapping dataset."
    )
    
    # Define required positional arguments
    parser.add_argument(
        "n", 
        type=int, 
        help="Max string length"
    )
    parser.add_argument(
        "raw_charset", 
        type=str, 
        help="An integer size for ascii_lowercase slice, or a literal string of characters to use"
    )
    parser.add_argument(
        "num_lines", 
        type=int, 
        help="Number of dataset lines to generate"
    )
    
    args = parser.parse_args()
    
    n = args.n
    raw_charset = args.raw_charset
    num_lines = args.num_lines
    
    # Parse the charset logic
    if raw_charset.isdigit():
        count = int(raw_charset)
        charset = string.ascii_lowercase[:min(count, 26)]
    else:
        charset = raw_charset

    # 1. Generate all possible unique strings for 'a'
    # NOTE: If n or charset is huge, you might want to randomly sample unique strings instead
    all_possible_inputs = generate_all_unique_strings(n, charset)
    
    # 2. Shuffle them to create a completely unique, randomized 1-to-1 mapping pool for 'b'
    random.seed(42) # Keeping seed for deterministic dictionary generation
    all_possible_outputs = all_possible_inputs.copy()
    random.shuffle(all_possible_outputs)
    
    # 3. Create the master 1-to-1 mapping lookup table
    mapping_dict = dict(zip(all_possible_inputs, all_possible_outputs))
    
    # 4. Generate the dataset by sampling lines from our strict dictionary mapping
    # We reset the seed or let it roll if you want variety in what lines are picked
    with open("input.txt", "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            a = random.choice(all_possible_inputs)
            b = mapping_dict[a]  # 1-to-1 deterministic answer
            
            # a_padded = "_" * (n - len(a)) + a
            # b_padded = b + "_" * (n - len(b))
            
            f.write(f"{a}={b}\n")

    print(f"Successfully generated {num_lines} lines in input.txt")

if __name__ == "__main__":
    main()