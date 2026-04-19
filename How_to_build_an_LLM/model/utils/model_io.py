"""Loading, and saving model checkpoints."""

from typing import TYPE_CHECKING
import os  # move to Pathlib
import torch

if TYPE_CHECKING:
    from llama_config import BaseConfig
    from model.training import PreTrainModel
    from model.decoder import Decoder


# TODO review code to make it work with this project
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
    """

    checkpoint_name = f"scaled_down_llama_stepCount_{step_counter}.pt"
    checkpoint_path = os.path.join(cfg.MODEL_DIR, "checkpoints", checkpoint_name)

    torch.save(
        {
            "step_counter": step_counter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step_counter": step_counter,
            "loss_history": loss_history,
            "loss": avg_loss,
        },
        checkpoint_path,
    )
    print(f"Saved Checkpoint to -> {checkpoint_path}")


def load_checkpoint(trainer: PreTrainModel, cfg: BaseConfig) -> int:
    """
    Load a checkpoint.

    Returns:
        The last step the checkpoint was trained on.
    """
    device = cfg.device
    chpt_path = os.path.join(cfg.MODEL_DIR, "checkpoints", cfg.checkpoint_name)

    if os.path.exists(chpt_path):
        print(f"Loading checkpoint: {cfg.checkpoint_name}...")
        chpt = torch.load(chpt_path, map_location=device)

        # Load all states into the trainer
        trainer.model.load_state_dict(chpt["model_state_dict"])
        trainer.optimizer.load_state_dict(chpt["optimizer_state_dict"])

        # MPS device bug issue, fix by moving optimizer state tensors to mps device
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if "scheduler_state_dict" in chpt:
            trainer.scheduler.load_state_dict(chpt["scheduler_state_dict"])
        if "step_counter" in chpt:
            trainer.step_counter = chpt["step_counter"]

        last_step = chpt.get("step_counter", 0)
        print(f"\nResuming training from Step {last_step}...")
        return last_step

    else:
        print(f"\n\nCheckpoint not found at {chpt_path}! Check configurations!")
        import sys
        sys.exit(1)
