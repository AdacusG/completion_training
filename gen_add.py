#!/usr/bin/env python3
import random
import string
import sys

def main():
    V = int(sys.argv[1])
    N = int(sys.argv[2])
    with open("inputadd.txt", "w", encoding="utf-8") as f:
        for i in range(N):
            a = random.randint(0, V-1)
            b = random.randint(0, V-1)
            f.write(f"{a}+{b}={(a+b) % V}\n")

if __name__ == "__main__":
    main()