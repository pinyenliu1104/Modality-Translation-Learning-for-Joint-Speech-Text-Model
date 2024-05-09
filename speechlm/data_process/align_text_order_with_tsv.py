# align the line order of the text file with the tsv file

import argparse
import numpy as np
import sys
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(
        description="align the line order of the text file with the tsv file"
    )
    parser.add_argument(
        "--input-text",
        "-i",
        help="input text file",
        required=True,
    )
    parser.add_argument(
        "--input-tsv",
        "-t",
        help="input tsv file",
        required=True,
    )
    parser.add_argument(
        "--output-text",
        "-o",
        help="output text file",
        required=True,
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.input_tsv, "r") as ftsv, open(args.input_text, "r") as ftext, open(args.output_text, "w", encoding="utf-8") as fout:
        tsv_lines = ftsv.readlines()
        text_lines = ftext.readlines()
        i=0

        for tsv_line in tqdm(tsv_lines):
            # don't read the first line of the tsv file
            if tsv_line.startswith("/home_new"):
                continue
            tsv_id = tsv_line.split("/")[2]
            tsv_id = tsv_id.split(".")[0]
            text_line = text_lines[i]
            fout.write(f'{tsv_id} {text_line}')
            i+=1

if __name__ == "__main__":
    main()
