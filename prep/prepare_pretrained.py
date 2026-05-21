import easyjupyter
import numpy as np
from tqdm import tqdm
import gc
from model.bpe_tokenizer import BPETokenizer
from datasets import Dataset, load_dataset
import os
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import ConfigTemplate


def prepare_pretrain_data(cfg: ConfigTemplate, tokenizer, is_overfit=False):
    """
    Builds the binary dataset from the a dataset for the pre-training phase.

    Pipeline:
        - Stream + parallel tokenize utilizing all CPU cores -> append tokens to binary file -> and repeat until it reaches the tokens need for a stage.

    Args:
        is_overfit: If true, adds "overfit_" to the beginning of the binary file name.
    """

    ds_info = {
        "initial_stage": cfg.text_only.pretrain.initial_stage.dataset,
        "lc_stage": cfg.text_only.pretrain.lc_stage.dataset,
        "annealing_stage": cfg.text_only.pretrain.annealing_stage.dataset,
    }
    doc_end_id = tokenizer.token_to_id(cfg.special_tokens["doc_end_token"]["token"])

    # The pre-training has three stages: The initial and long-context stage get the same dataset. The annealing stage gets a higher quality dataset
    for stage_name in list(ds_info):
        print(f"\n--- Preparing {stage_name} ---")
        stage_cfg = getattr(cfg.text_only.pretrain, stage_name)
        hf_batch_size = getattr(stage_cfg, "streaming_batch_size", 100)
        print("streaming_batch_size:", hf_batch_size)

        tokens_target = stage_cfg.tokens

        print(f"\nTargeting: {tokens_target:,} tokens for this stage.")

        prefix = "overfit_" if is_overfit else ""
        bin_file = cfg.data_dir / f"{prefix}{stage_name}.bin"
        meta_path = bin_file.with_suffix(".json")

        ds_stream = load_dataset(
            stage_cfg.dataset.name,
            stage_cfg.dataset.config,
            split="train",
            streaming=True,
        )

        def tokenize_batch(docs):
            encodings = tokenizer.encode_batch(docs["text"])
            # Append the doc_end_id to each encoded sequence
            ids = [enc.ids + [doc_end_id] for enc in encodings]
            return {"ids": ids}

        tokenized_stream = ds_stream.map(
            tokenize_batch,
            batched=True,
            batch_size=hf_batch_size,
            remove_columns=list(
                # Drops the 'text', etc... columns to save RAM
                ds_stream.features.keys()
            ),
        )

        # The stream continues until it reaches the tokens needed for a stage.
        total_tokens_written = 0
        hit_token_cap = False

        with open(bin_file, "wb") as f:
            for doc in tqdm(tokenized_stream, desc="Writing to Binary"):

                doc_ids = doc["ids"]
                tokens_needed = tokens_target - total_tokens_written

                # Cap it exactly at the tokens target
                if len(doc_ids) >= tokens_needed:
                    doc_ids = doc_ids[:tokens_needed]
                    hit_token_cap = True

                # Write to disk
                np.array(doc_ids, dtype=cfg.dtype).tofile(f)
                total_tokens_written += len(doc_ids)

                if hit_token_cap:
                    print(
                        f"\nTokenized {total_tokens_written:,} tokens for this stage."
                    )
                    break

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "total_tokens": tokens_target,
                    "vocab_size": cfg.vocab_size,
                    "num_tokens_written_to_binary_file": total_tokens_written,
                },
                f,
                indent=4,
            )
        print(f"\nFinished tokenizing {stage_name} dataset to {bin_file}")
