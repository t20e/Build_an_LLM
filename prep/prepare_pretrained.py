"""
--- IMPORTANT ---

Pipeline:
    1. Downloads: Stream a specified number of documents (max_docs) form the HuggingFace dataset, and save them to a .parquet file. `token_budget` has no effect on `max_docs`. For training, we either run out of documents, or hit the `token_budget`.
    2. Tokenizes from that .parquet file in parallel across all CPU cores, and save it as a binary file.
        - Now the dataloader is simply just a memory-map to the binary file, it will only load what it needs at any given time.
    - NOTE: This was one of the bigger headaches when building this project. I tried a few different options, but stuck with the above pipeline. The main issue with this method is when you want massive datasets (~1TB+), this method will leave you with two files: one Parquet and one binary file. For example, the entire FineWeb dataset is 54.8 TB, you would need 109.6 TB of storage to hold both files.
        - Alternatives:
            1. Stream to RAM -> tokenize whats in RAM -> append to binary file -> repeat.
                - Issues:
                    1. If you want to update the hyperparameters of the tokenizer, you will need to re-download the dataset and tokenize it again.
                    2. HuggingFace stream makes this difficult to implement, maybe look into https://github.com/Lightning-AI/litdata
            2. Stream a small amount of documents at a time -> tokenize those documents on the fly -> append to binary file -> repeat.
                - Issues:
                    1. If you want to update the hyperparameters of the tokenizer, you will need to re-download the dataset and tokenize it again.
                    2. The stream will need to wait for the CPU to finish tokenizing whatever it is currently working on. So for most of it, the stream will be idling.
                    3. You will not be able to utilize all the CPU cores. It is a lot slower than the other two options.
"""

import easyjupyter
import numpy as np
from tqdm import tqdm
import gc
from model.bpe_tokenizer import BPETokenizer
from datasets import Dataset, load_dataset
import os
import json
from model.data_loader import get_raw_ds
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_configs import BaseConfig


def prepare_pretrain_data(cfg: BaseConfig, is_overfit=False):
    """
    Builds the binary dataset from the a dataset for the pre-training phase.

    1. Downloads: Stream a specified number of documents (max_docs) form the HuggingFace dataset, and save them to a .parquet file. `token_budget` has no effect on `max_docs`. For training, we either run out of documents, or hit the `token_budget`.
    2. Tokenizes from that .parquet file in parallel across all CPU cores, and save it as a binary file.

    Args:
        is_overfit: Whether to create a smaller dataset for overfit testing.
    """

    max_docs = cfg.overfit_max_docs if is_overfit else cfg.max_docs

    # The pre-training has three stages: The initial and long-context stage get the same dataset. The annealing stage gets a higher quality dataset
    for stage_name, ds_info in cfg.pre_training_ds.items():
        if is_overfit and stage_name != "initial_and_lc_stage":
            # We only need the first file for overfitting.
            continue

        print(f"\n--- Preparing {stage_name}")

        perc = cfg.pre_training_stages_steps.get(f"{stage_name.replace('_stage', '')}_perc", 1.0)
        if stage_name == "initial_and_lc_stage":
            # Combine the initial and long-context stages to use the same dataset.
            perc = (
                cfg.pre_training_stages_steps["initial_perc"]
                + cfg.pre_training_stages_steps["long_context_perc"]
            )
        
        stage_docs = max_docs if is_overfit else int(max_docs * perc)
        print(f"\nTargeting: {stage_docs:,} documents for this stage.")

        prefix = "overfit_" if is_overfit else ""
        raw_parquet_file = cfg.DATA_DIR / f"{prefix}{stage_name}.parquet"
        output_file = cfg.DATA_DIR / f"{prefix}{stage_name}.bin"
        meta_path = output_file.with_suffix(".json")

        # =======================================================================
        # DOWNLOAD THE RAW DATASET TO PARQUET FILE
        # =======================================================================
        if not raw_parquet_file.exists():
            print(f"\nDownloading {stage_docs:,} documents...")

            def raw_data_generator():
                ds_stream = get_raw_ds(ds_info["name"], ds_info["config"])
                for i, sample in enumerate(ds_stream):
                    if i >= stage_docs:
                        break
                    yield {"text": sample["text"]}

            raw_ds = Dataset.from_generator(raw_data_generator)
            raw_ds.to_parquet(str(raw_parquet_file))
            print(f"Saved {len(raw_ds):,} documents to parquet file")
            del raw_ds
        else:
            print(f"\nFound existing {raw_parquet_file.name}. Skipping download.")

        # Load the local dataset
        local_ds = load_dataset(
            "parquet", data_files=str(raw_parquet_file), split="train"
        )

        # =======================================================================
        # TRAIN A TOKENIZER
        # =======================================================================
        bpe_tokenizer = BPETokenizer(cfg)
        success, tokenizer = bpe_tokenizer.load_tokenizer()

        if not success and stage_name == "initial_and_lc_stage":
            # The dataset for the initial and long-context stages is the larger of the two (90+%), so its enough for the tokenizer.
            print(f"\nTokenizer not found, training new tokenizer...")
            tokenizer = bpe_tokenizer.train_tokenizer(local_ds)
        elif stage_name == "annealing_stage" and not success:
            # When processing annealing stage and the tokenizer hasn't been trained yet.
            raise FileNotFoundError(
                f"Tokenizer not found, re-run with the initial_and_lc_stage dataset downloaded."
            )

        doc_end_id = tokenizer.token_to_id(cfg.special_tokens["doc_end_token"]["token"])

        # =======================================================================
        # PARALLELIZE TOKENIZATION & WRITE TO BINARY FILE
        # =======================================================================
        def tokenize_batch(samples):
            """Parallelize tokenization to all CPU cores."""
            outputs = []
            for text in samples["text"]:
                ids = tokenizer.encode(text).ids
                ids.append(doc_end_id)
                outputs.append(ids)
            return {"tokens": outputs}

        tokenized_ds = local_ds.map(
            tokenize_batch,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=local_ds.column_names,
            desc="Mapping tokenization",
        )
        print(f"\nWriting tokenized array of sequential binary layout; {output_file}...")
        token_count = 0
        with open(output_file, "wb") as f:
            for sample in tqdm(tokenized_ds, desc="Writing tokens"):
                tokens = sample["tokens"]
                np.array(tokens, dtype=cfg.dtype).tofile(f)
                token_count += len(tokens)

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "total_tokens": token_count,
                    "vocab_size": cfg.vocab_size,
                    "documents_processed": len(tokenized_ds),
                },
                f,
                indent=4,
            )
        print(f"\nFinished tokenizing {stage_name} dataset to {output_file}")