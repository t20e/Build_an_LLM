"""
- Run this script before main.py
It:
    1. Checks if a trained Tokenizer exists, if not, train it.
    2. Builds the binary dataset from the raw text dataset.
"""

import EasyJupyter
from data_prep.prepare_pretrained import prepare_pretrain_data
import argparse
import os
from llama_configs import Llama3_scaled_down
from utils.misc import print_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3 Prepare Dataset CLI.")
    parser.add_argument(
        "--d",
        choices=[
            "pretrain",
            "sft",
            "dpo",
        ],
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
    )

    args = parser.parse_args()
    print_args(args)

    match args.d:
        case "pretrain":
            cfg = Llama3_scaled_down()
            is_overfit = False

            if args.overfit:
                cfg.token_budget = 5_000
                is_overfit = True

            prepare_pretrain_data(cfg, is_overfit)
        case "sft":
            pass
        case "dpo":
            pass

    os._exit(0)
