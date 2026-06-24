#!/usr/bin/env python3
import random
import string
import argparse

def generate_reverse_dataset(n, charset_size, num_lines):
    # Ensure charset size doesn't exceed lowercase alphabet length
    charset_size = min(max(1, charset_size), 26)
    available_chars = string.ascii_lowercase[:charset_size]
    
    output_filename = "inputrev.txt"
    
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
            
            # Compute the reversed ground truth
            reversed_str = input_str[::-1]
            
            # Format exactly matching your completion training scheme
            f.write(f"{input_str}={reversed_str}\n")
            
    print(f"Dataset successfully created!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a reverse string dataset for sequence completion models."
    )
    
    # Define positional parameters. Using type=int automatically validation handles type checking.
    # nargs="?" allows them to be optional so fallback defaults work if no arguments are passed.
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
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Determine if any arguments were provided to mimic the original print statement logic
    # argparse assigns defaults if args aren't passed, so we check if the user omitted them.
    import sys
    if len(sys.argv) < 2:
        print(f"No parameters passed. Using defaults: n={args.n}, charset={args.charset_size}, num_lines={args.num_lines}")

    generate_reverse_dataset(args.n, args.charset_size, args.num_lines)