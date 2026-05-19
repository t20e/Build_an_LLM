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
        "initial_and_lc": cfg.text_only.pretrain.initial_stage.dataset,
        "annealing": cfg.text_only.pretrain.annealing_stage.dataset,
    }

    # The pre-training has three stages: The initial and long-context stage get the same dataset. The annealing stage gets a higher quality dataset
    for stage_name in ["initial_and_lc", "annealing"]:
        print(f"\n--- Preparing {stage_name} stage ---")
        print(f"chunk_size: {cfg.chunk_size}")

        if stage_name == "initial_and_lc":
            # Combine the initial and long-context token counts so they come from the same dataset.
            tokens = (
                cfg.text_only.pretrain.initial_stage.tokens
                + cfg.text_only.pretrain.lc_stage.tokens
            )
        else:
            tokens = cfg.text_only.pretrain.annealing_stage.tokens

        # The stream continues until it reaches the tokens needed for a stage.
        total_tokens_written = 0

        print(f"\nTargeting: {tokens:,} tokens for this stage.")

        prefix = "overfit_" if is_overfit else ""
        bin_file = cfg.data_dir / f"{prefix}{stage_name}.bin"
        meta_path = bin_file.with_suffix(".json")

        ds_stream = load_dataset(
            ds_info[stage_name].name,
            ds_info[stage_name].config,
            split="train",
            streaming=True,
        )
        doc_end_id = tokenizer.token_to_id(cfg.special_tokens["doc_end_token"]["token"])

        batch_text_buffer = []
        """NOTE: There will always be more tokens saved to the binary file than is needed for training, this is because we overshoot using the chunk_size."""

        num_docs = 0

        with open(bin_file, "wb") as f:
            for sample in tqdm(ds_stream, desc="Streaming Dataset"):
                batch_text_buffer.append(sample["text"])
                num_docs += 1

                if len(batch_text_buffer) >= cfg.chunk_size:
                    # Parallel tokenization over CPU cores
                    encoded_batches = tokenizer.encode_batch(batch_text_buffer)
                    hit_token_cap = False

                    # Write to binary file
                    for encoding in encoded_batches:
                        ids = encoding.ids
                        ids.append(doc_end_id)
                        np.array(ids, dtype=cfg.dtype).tofile(f)
                        total_tokens_written += len(ids)

                        if total_tokens_written >= tokens:
                            hit_token_cap = True
                            break

                    batch_text_buffer = []  # Reset the buffer

                    if hit_token_cap:
                        print(
                            f"\nTokenized {total_tokens_written:,} tokens for this stage. (Fine if overshoot)"
                        )
                        break

            # Any remaining texts in the buffer is discarded

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stage": stage_name,
                    "total_tokens": tokens,
                    "vocab_size": cfg.vocab_size,
                    "num_tokens_written_to_binary_file": total_tokens_written,
                },
                f,
                indent=4,
            )
        print(f"\nFinished tokenizing {stage_name} dataset to {bin_file}")
