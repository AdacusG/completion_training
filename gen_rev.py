#!/usr/bin/env python3
import random
import string
import sys

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
    # Check if correct arguments are provided via terminal
    if len(sys.argv) == 4:
        try:
            n = int(sys.argv[1])
            charset = int(sys.argv[2])
            num_lines = int(sys.argv[3])
        except ValueError:
            print("Error: All arguments must be integers.")
            print("Usage: python gen_rev.py <n> <charset_size> <num_lines>")
            sys.exit(1)
    else:
        # Fallback default values if run without CLI arguments
        print("No parameters passed. Using defaults: n=3, charset=26, num_lines=1000")
        n = 3
        charset = 26
        num_lines = 1000

    generate_reverse_dataset(n, charset, num_lines)