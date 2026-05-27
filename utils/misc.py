import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import BaseConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_num_workers() -> int:
    # 🚨 NOTE: For NVIDIA GPUs the Golden Rule is: num_worker = 4 * num_GPU | On Mac Silicone even though I have a 32 core GPU, it is still only one GPU, best to set num_workers = 0.
    if torch.cuda.is_available():
        return torch.cuda.device_count() * 4
    else:
        # MPS (Mac Silicon GPUs) and if you are just using CPU instead of a GPU
        return 0


def print_args(args):
    print("\n\n--------- Arguments ----------")
    for arg, val in vars(args).items():
        print(f"{arg}: {val}")
    print("--------- End Arguments ----------\n")


# TODO finish and correct the print attributes
def print_config(cfg: BaseConfig) -> str:
    return f"""\n\n{'='*20} Config {'='*20}
            [Model Architecture]
                - model_name               = {cfg.model_name}
                - d_model                  = {cfg.d_model}
                - num_kv_heads             = {cfg.num_kv_heads}
                - d_ff                     = {cfg.d_ff}
                - num_layers               = {cfg.num_layers}
                - attn_heads               = {cfg.attn_heads}
                - rms_norm_eps             = {cfg.rms_norm_eps}
                - rope_theta               = {cfg.rope_theta}
                - vocab_size               = {cfg.vocab_size}
            \n
            [Active Run]
                - num_workers              = {cfg.num_workers}
                - device                   = {cfg.device}
            \n
            [Stages Configs]
                [Pre-Training Stage]
                    - token_budget         = {None}
                    - peak_lr              = {None}
                    \n

                    > Initial Stage:
                        - tokens           = {cfg.text_only.pretrain.initial_stage.tokens}
                        - warmup_steps     = {cfg.text_only.pretrain.initial_stage.warmup_steps}
                        - dataset          = {cfg.text_only.pretrain.initial_stage.dataset.name} | {cfg.text_only.pretrain.initial_stage.dataset.config}
                    \n

                    > Long-Context Stage:
                        - tokens           = {cfg.text_only.pretrain.lc_stage.tokens}
                        - seq_len      = {cfg.text_only.pretrain.lc_stage.seq_len}
                        - grad_accum_steps = {None}
                        - dataset          = {cfg.text_only.pretrain.lc_stage.dataset.name} | {cfg.text_only.pretrain.lc_stage.dataset.config}
                    \n

                    > Annealing Stage:
                        - tokens           = {cfg.text_only.pretrain.annealing_stage.tokens}
                        - seq_len      = {cfg.text_only.pretrain.annealing_stage.seq_len}
                        - grad_accum_steps = {None}
                        - dataset          = {cfg.text_only.pretrain.annealing_stage.dataset.name} | {cfg.text_only.pretrain.annealing_stage.dataset.config}
            \n\n
            """
