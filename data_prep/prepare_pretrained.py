"""
--- IMPORTANT ---

Pipeline:
    1. Downloads a portion of the dataset that fits in RAM buffer.
    2. Tokenizes whats in RAM in parallel across all CPU cores, and appends it to the binary file.
    3. Repeats
    - Now the dataloader is simply just a memory-map to the binary file, it will only load what it needs at any given time.
    - NOTE: I found this to be the best options, its only downside is that you cannot use a different tokenizer with the same binary file. To use another tokenizer, you must re-download the dataset and tokenize it again. Here are some other alternatives that aren't so great:
        1. Download the whole dataset, saving it as a Parquet file, and tokenize from the Parquet file into the binary file. This isn't a good idea for massive datasets (~1TB+), because it will leave you with two files: one Parquet and one binary file. For example, the entire FineWeb dataset is 54.8 TB, you would need 109.6 TB of storage to hold both files. For this case, cloud preprocessing across a cluster of CPUs is the only option!
        2. Another option is to (stream -> tokenize a small amount of documents on the fly -> append to binary file and repeat). The two major downsides are:
            1. That you cannot use another tokenizer to load the dataset from that binary file afterwards, you must use the same tokenizer that was used to tokenize it.
            2. The stream will need to wait for the CPU to finish tokenizing whatever it is currently working on, and you will not be able to utilize all the CPU cores.
"""

import EasyJupyter
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from model.tokenizer import BPETokenizer
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import json
from model.data_loader import get_raw_ds
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_configs import BaseConfig


def prepare_pretrain_data(cfg: BaseConfig, is_overfit=False):
    """
    Builds the binary dataset from the a dataset for the pre-training phase.

    Pipeline:
        1. Downloads a portion of the dataset that fits in RAM buffer.
        2. Tokenizes whats in RAM in parallel across all CPU cores, and appends it to the binary file.
        3. Repeats

    Args:
        is_overfit: Whether to create a smaller dataset for overfit testing.
    """

    training_samples = 5000 if is_overfit else cfg.tokenizer_training_samples

    bpe_tokenizer = BPETokenizer(cfg)
    success, tokenizer = bpe_tokenizer.load_tokenizer(training_samples)

    if not success:
        print("\nTokenizer not found, training one...")
        raw_ds = get_raw_ds(
            cfg.pre_training_ds["initial_and_lc_stage"]["name"],
            cfg.pre_training_ds["initial_and_lc_stage"]["config"],
        )
        tokenizer = bpe_tokenizer.train_tokenizer(
            raw_ds, training_samples=training_samples
        )

    doc_end_id = tokenizer.token_to_id(cfg.special_tokens["doc_end_token"]["token"])

    for stage_name, ds_info in cfg.pre_training_ds.items():
        # Stages: 1. initial and long-context, 2. annealing, both get different datasets.

        if is_overfit and stage_name != "initial_and_lc_stage":
            # We only need the first file for overfitting.
            continue

        print(f"\n--- Preparing {stage_name} | Streaming & Tokenizing...")

        prefix = "overfit_" if is_overfit else ""
        output_file = cfg.DATA_DIR / f"{prefix}{stage_name}.bin"

        # Calculate how many tokens are in this stage, e.g., if the annealing stage is 3% of the 2B token_budget, we only need ~60M tokens from the dataset that is used for the annealing stage.
        perc = cfg.pre_training_stages_steps.get(
            f"{stage_name.replace('_stage', '')}_perc", 1.0
        )
        if stage_name == "initial_and_lc_stage":
            # Combine the initial and long-context stages to use the same dataset.
            perc = (
                cfg.pre_training_stages_steps["initial_perc"]
                + cfg.pre_training_stages_steps["long_context_perc"]
            )

        stage_budget = int(cfg.token_budget * perc)
        print(f"\nTarget tokens for this file: {stage_budget:,}")

        ds_stream = get_raw_ds(ds_info["name"], ds_info["config"])
        token_count = 0
        batch_size = 1000  # Number of docs to tokenize at once per CPU cycle.
        text_batch = []

        def tokenize_unit(text):
            """Parallelize tokenization to all CPU cores."""
            ids = tokenizer.encode(text).ids
            ids.append(doc_end_id)
            return ids

        with open(output_file, "wb") as f:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # ThreadPoolExecutor allows all available CPU cores to work on a batch.
                pbar = tqdm(total=stage_budget, unit="tokens", desc="Progress")

                for sample in ds_stream:
                    text_batch.append(sample["text"])

                    # When the RAM buffer is full, tokenize the entire batch in parallel.
                    if len(text_batch) >= batch_size:
                        results = list(executor.map(tokenize_unit, text_batch))

                        for ids in results:
                            remaining = stage_budget - token_count
                            if remaining <= 0:
                                break

                            # If the last document tokens puts us over the budget, slice it to be exact.
                            to_write = ids[:remaining]
                            np.array(to_write, dtype=cfg.dtype).tofile(f)
                            token_count += len(to_write)
                            pbar.update(len(to_write))

                        # Reset the RAM buffer
                        text_batch = []

                    if token_count >= stage_budget:
                        break
                pbar.close()

        # Save the metadata file.
        meta_path = output_file.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "total_tokens": token_count,
                    "vocab_size": cfg.vocab_size,
                },
                f,
                indent=4,
            )

        print(f"\nFinished tokenizing {stage_name} dataset to {output_file}")
