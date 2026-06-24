#!/usr/bin/env python3
import random
import string
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate a modular addition dataset."
    )
    
    # Define required positional arguments
    parser.add_argument(
        "V", 
        type=int, 
        help="The modulo base value (upper bound for random integers)"
    )
    parser.add_argument(
        "N", 
        type=int, 
        help="Number of addition lines to generate"
    )
    
    args = parser.parse_args()
    
    V = args.V
    N = args.N
    
    with open("inputadd.txt", "w", encoding="utf-8") as f:
        for i in range(N):
            a = random.randint(0, V-1)
            b = random.randint(0, V-1)
            f.write(f"{a}+{b}={(a+b) % V}\n")

if __name__ == "__main__":
    main()