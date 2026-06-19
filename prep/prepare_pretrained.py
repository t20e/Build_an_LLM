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
    Builds the binary datasets (train and validation partitions) for the pre-training phase.

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
        hf_batch_size = getattr(stage_cfg, "streaming_batch_size")
        tokens_target = stage_cfg.tokens
        val_tokens_target = stage_cfg.val_tokens

        print("streaming_batch_size:", hf_batch_size)
        print(f"Targeting: {tokens_target:,} train tokens | {val_tokens_target:,} validation tokens. tokens for this stage.")

        prefix = "overfit_" if is_overfit else ""
        bin_file = cfg.data_dir / f"{prefix}{stage_name}.bin"
        val_bin_file = cfg.data_dir / f"{prefix}{stage_name}_val.bin"
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
        total_train_written = 0
        total_val_written = 0
        hit_token_cap = False

        # Open both the train and val binary files concurrently
        with open(bin_file, "wb") as f_train, open(val_bin_file, "wb") as f_val:
            for doc in tqdm(tokenized_stream, desc="Processing Stream"):
                doc_ids = doc["ids"]

                # Fill validation bin first
                if total_val_written < val_tokens_target:
                    val_needed =  val_tokens_target - total_val_written
                    if len(doc_ids) >= val_needed:
                        # Slice exactly what we need to complete validation
                        val_slice = doc_ids[:val_needed]
                        np.array(val_slice, dtype=cfg.dataset_dtype_str).tofile(f_val)
                        total_val_written += len(val_slice)
                        continue
                    else:
                        np.array(doc_ids, dtype=cfg.dataset_dtype_str).tofile(f_val)
                        total_val_written += len(doc_ids)

                # Validation partition complete, now fill the train binary
                else:
                    train_needed = tokens_target - total_train_written

                    if len(doc_ids) >= train_needed:
                        doc_ids = doc_ids[:train_needed]
                        np.array(doc_ids, dtype=cfg.dataset_dtype_str).tofile(f_train)
                        total_train_written += len(doc_ids)
                        break
                    else:
                        np.array(doc_ids, dtype=cfg.dataset_dtype_str).tofile(f_train)
                        total_train_written += len(doc_ids)

            # Force the OS to immediately write the tokens to hard drive
            f_train.flush()
            os.fsync(f_train.fileno())
            f_val.flush()
            os.fsync(f_val.fileno())

            print(
                f"\nTokenized token counts for this stage."
                f"    Total train tokens: {total_train_written:,}"
                f"    Total val tokens: {total_val_written:,}"
            )


        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "vocab_size": cfg.vocab_size,
                    "train_tokens_target": tokens_target,
                    "total_train_written": total_train_written,
                    "val_tokens_target": val_tokens_target,
                    "total_val_written": total_val_written,
                },
                f,
                indent=4,
            )
        print(f"\nFinished processing stage: {stage_name}\n -> Train: {bin_file}\n -> Val: {val_bin_file}")
