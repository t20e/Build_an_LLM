"""Loading, and saving model checkpoints."""

from typing import TYPE_CHECKING
from pathlib import Path
import torch

if TYPE_CHECKING:
    from configs import ConfigTemplate
    from model.model_text_only import TextOnlyModel


def save_checkpoint(
    cfg: ConfigTemplate,
    model: TextOnlyModel,
    optimizer: torch.optim.AdamW,
    loss_history: list,
    save_path: Path,  # ex: .../Scaled_down_Base/global_step_000250
):
    """
    Save a checkpoint.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,  # NOTE: Loss history can get very large, consider capping it or savings it in its own JSON fle, and keeping only the last N values.
        },
        save_path / "checkpoint.pt",
    )

    cfg.save(save_path)

    # # Update latest.txt
    # with open(cfg.chpts_dir / "latest.txt", "w") as f:
    #     f.write(save_path.name)

    print(f"Saved Checkpoint to -> {save_path}")


def load_checkpoint(
    cfg,
    model,
    chpt_dir: Path,
    optimizer=None,
):
    """
    Load a checkpoint folder snapshot containing the model states.

    Args:
        optimizer: If true, loads the model's optimizer as well
    """
    if not chpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory found at: {chpt_dir}")

    chpt_file_path = chpt_dir / "checkpoint.pt"
    if not chpt_file_path.exists():
        raise FileNotFoundError(f"Missing checkpoint.pt in: {chpt_dir}")

    print(f"Loading checkpoint states from: {chpt_file_path}...")
    checkpoint = torch.load(
        chpt_file_path,
        map_location=cfg.device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])  # Weights

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if cfg.device.type == "mps":
            # MPS device bug issue, fix by moving optimizer state tensors to mps device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.device)

    print(f"Successfully loaded checkpoint: {chpt_file_path.name}")
