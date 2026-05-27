"""
- Run this script before main.py
It:
    1. Checks if a trained Tokenizer exists, if not, train it.
    2. Builds the binary dataset from the raw text dataset.
"""

import easyjupyter
from prep.prepare_pretrained import prepare_pretrain_data
import argparse
import os
from configs import Llama3_1_405B, Scaled_down_text
from utils.misc import print_args
from model.bpe_tokenizer import BPETokenizer
from datasets import load_dataset

import warnings
import os

# Silence HuggingFace multiprocessing resource tracker's final exit warning
os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker"
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")


MODELS_SUITE = {
    "Scaled_down_text": Scaled_down_text,
    "3.1_405B": Llama3_1_405B,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3 Prepare Dataset CLI.")
    parser.add_argument(
        "--stage",
        choices=[
            "pretrain",
            "sft",
            "dpo",
        ],
    )

    parser.add_argument("--model", type=str, default="Scaled_down_text")

    parser.add_argument(
        "--overfit",
        action="store_true",
    )
    parser.add_argument(
        "--tokenizer_only",
        action="store_true",
        help="Only train the tokenizer, do not build the binary dataset.",
    )

    args = parser.parse_args()
    print_args(args)

    if args.model not in MODELS_SUITE:
        raise ValueError(f"Model {args.model} not recognized!")

    cfg = MODELS_SUITE[args.model].initialize(is_overfit=args.overfit)

    bpeTokenizer = BPETokenizer(cfg=cfg)
    success, tokenizer = bpeTokenizer.load_tokenizer()

    if not success:
        if args.stage != "pretrain":
            raise ValueError(
                f"\nTokenizer file not found! You must use the tokenizer that you trained in pre-training, for all further stages!"
            )

        ds_stream = load_dataset(
            cfg.text_only.pretrain.initial_stage.dataset.name,
            cfg.text_only.pretrain.initial_stage.dataset.config,
            split="train",
            streaming=True,
        )
        samples = [
            sample["text"]
            for sample in ds_stream.take(
                cfg.text_only.pretrain.initial_stage.tokenizer_training_samples
            )
        ]
        tokenizer = bpeTokenizer.train(samples)

    match args.stage:
        case "pretrain":
            if not args.tokenizer_only:
                prepare_pretrain_data(cfg, tokenizer, args.overfit)
        case "sft":
            # TODO Can only load the tokenizer, not train it.
            pass
        case "dpo":
            # TODO Can only load the tokenizer, not train it.
            pass
    os._exit(0)
