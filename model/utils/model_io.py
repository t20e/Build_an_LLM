"""Loading, and saving model checkpoints."""

from typing import TYPE_CHECKING
from pathlib import Path
import torch

if TYPE_CHECKING:
    from llama_configs import BaseConfig
    from model.decoder import Decoder


def save_checkpoint(
    cfg: BaseConfig,
    model: Decoder,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step_counter: int,
    loss_history: list,
    avg_loss: float,
):
    """
    Save a checkpoint.

    Args:
        cfg: The config object.
        model: The model object.
        optimizer: The optimizer object.
        scheduler: The scheduler object.
        step_counter: The step counter that was reached during training.
        loss_history: The loss history during training.
        avg_loss: The last loss during training.
    """
    chpt_filename = f"step_{step_counter:06d}.pt"
    save_path = cfg.CURR_CHPT_DIR / chpt_filename

    torch.save(
        {
            "step_counter": step_counter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_history": loss_history, # NOTE: Loss history can get very large, consider capping it or savings it in its own JSON fle, and keeping only the last N values.
            "avg_loss": avg_loss,
        },
        save_path,
    )

    # Update latest.txt
    with open(cfg.CURR_CHPT_DIR / "latest.txt", "w") as f:
        f.write(chpt_filename)

    print(f"Saved Checkpoint to -> {save_path}")


def load_checkpoint(
    cfg: BaseConfig,
    model: torch.nn.Module,
    chpt_path: Path,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
):
    """
    Load a checkpoint. If optimizer and scheduler are provided, their states are also loaded.

    Args:
        chpt_path: Direct path to the checkpoint file.
    """
    if not chpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at: {chpt_path}")

    print(f"Loading checkpoint at: {chpt_path}...")
    checkpoint = torch.load(chpt_path, map_location=cfg.device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])  # weights

    # Only load the optimizer and scheduler, when continuing training
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if cfg.device == "mps":
            # MPS device bug issue, fix by moving optimizer state tensors to mps device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.device)

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    cfg.step_counter = checkpoint.get("step_counter", 0)

    print(
        f"Successfully loaded checkpoint: {chpt_path.name} (Step: {cfg.step_counter})"
    )
