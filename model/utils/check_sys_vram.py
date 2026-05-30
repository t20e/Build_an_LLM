"""
Use HuggingFace's @find_executable_batch_size to dynamically set the micro_batch_size to an adequate size for the current system.
"""

import easyjupyter
from accelerate.utils import find_executable_batch_size
import torch

from model.utils.masking import create_causal_doc_mask
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.optim as optim
    from model.model_text_only import TextOnlyModel
    from configs import ConfigTemplate


def get_max_micro_batch_size(
    cfg: ConfigTemplate,
    model: TextOnlyModel,
    target_seq_len: int,
    starting_batch_size: int=4_096,
):
    """
    Use HuggingFace's `@find_executable_batch_size` to dynamically set the micro_batch_size to an adequate size for the current system.
        1. Probes hardware VRAM limits by attempting forward/backward passes.
        2. Automatically halves the micro_batch_size upon encountering an OOM (Out of Memory) error.

    Args:
        cfg: The configuration object containing model details.
        starting_batch_size: The initial micro_batch_size that will be attempting, if OOM error occurs, starting_batch_size will be halved, and repeat, until a the best max micro_batch_size is found for the current system.
    """
    print("\n\nAttempting to find the best micro_batch_size for the current system VRAM hardware")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def forward_backward_probe(batch_size):
        print(
            f"Probing hardware VRAM capacity with micro_batch_size = {batch_size} with seq_len = {target_seq_len}..."
        )

        # Create dummy tensors matching the target sequence length
        x_dummy = torch.randint(
            0, cfg.vocab_size, (batch_size, target_seq_len), device=cfg.device
        )
        y_dummy = torch.randint(
            0, cfg.vocab_size, (batch_size * target_seq_len,), device=cfg.device
        )

        mask_dummy = create_causal_doc_mask(cfg, x_dummy)

        # NOTE: My M1 Mac does not support bfloat16, but later M3 onward support bfloat16
        autocast_dtype = torch.float16 if cfg.device.type == "mps" else torch.bfloat16

        # Dummy forward & backward pass
        with torch.autocast(device_type=cfg.device.type, dtype=autocast_dtype):
            logits, _ = model(x_dummy, mask_dummy)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), y_dummy
            )
        loss.backward()

        # Memory clean-up
        model.zero_grad(set_to_none=True)
        if cfg.device.type == "mps":
            torch.mps.empty_cache()
        elif cfg.device.type == "cuda":
            torch.cuda.empty_cache()

        return batch_size

    was_training = model.training
    model.train()

    best_batch_size = forward_backward_probe()
    if not was_training:
        model.eval()
    print(f"The best and max micro batch size for the current system is {best_batch_size}.")
    return best_batch_size
