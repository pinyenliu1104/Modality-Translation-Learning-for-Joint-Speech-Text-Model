"""
This file is used for converting output of generate_unit.py to unit data for training
"""

import os
import argparse
from tqdm import tqdm
import numpy as np


def writefile(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    args = parser.parse_args()
    
    tgt_lines = []

    with open(args.input, 'r') as f: 
        for line in tqdm(f):
            if line.startswith("D-"):
                # only keep the number after 0.0
                print(line)
                line = line.split("0.0	")[1]
                tgt_lines.append(line)

    writefile(f"{args.output}.km", tgt_lines)

if __name__ == "__main__":
    main()

