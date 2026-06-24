#!/usr/bin/env python3
import random
import string
import argparse

def generate_capital_dataset(n, charset_size, num_lines):
    # Ensure charset size doesn't exceed lowercase alphabet length
    charset_size = min(max(1, charset_size), 26)
    available_chars = string.ascii_lowercase[:charset_size]
    
    output_filename = "inputcapital.txt"
    
    print(f"Generating dataset...")
    print(f"Max string length (n) : {n}")
    print(f"Using charset [{charset_size}]: {available_chars}")
    print(f"Writing {num_lines} lines to {output_filename}...\n")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            # Pick a random length for the input string between 1 and n
            line_len = random.randint(1, n)
            
            # Generate random characters from our specific charset slice
            input_chars = [random.choice(available_chars) for _ in range(line_len)]
            input_str = "".join(input_chars)
            
            # Compute the capitalized ground truth
            capitalized_str = input_str.upper()
            
            # Format exactly matching your completion training scheme
            f.write(f"{input_str}={capitalized_str}\n")
            
    print(f"Dataset successfully created!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a capitalization dataset for sequence completion models."
    )
    
    # Define optional positional arguments with defaults using nargs="?"
    parser.add_argument(
        "n", 
        type=int, 
        nargs="?", 
        default=3, 
        help="Max string length (default: 3)"
    )
    parser.add_argument(
        "charset_size", 
        type=int, 
        nargs="?", 
        default=26, 
        help="Size of the alphabet charset to use (default: 26)"
    )
    parser.add_argument(
        "num_lines", 
        type=int, 
        nargs="?", 
        default=1000, 
        help="Number of dataset lines to generate (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Check if the user ran the script without passing arguments to mirror original console output
    import sys
    if len(sys.argv) < 2:
        print(f"No parameters passed. Using defaults: n={args.n}, charset={args.charset_size}, num_lines={args.num_lines}")

    generate_capital_dataset(args.n, args.charset_size, args.num_lines)