"""Loading, and saving model checkpoints."""

from typing import TYPE_CHECKING
from pathlib import Path
import torch

if TYPE_CHECKING:
    from llama_configs import BaseConfig
    from model.training import PreTrainModel
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
    # TODO how should I resolve when a checkpoint is already saved to the same name?

    chpt_dir_path = cfg.CHPT_DIR / cfg.model_name
    chpt_dir_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step_counter": step_counter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step_counter": step_counter,
            "loss_history": loss_history,
            "avg_loss": avg_loss,
        },
        chpt_dir_path / f"{cfg.model_name}.pt",
    )

    print(f"Saved Checkpoint to -> {chpt_dir_path}")


def load_checkpoint(trainer: PreTrainModel, cfg: BaseConfig) -> int:
    """
    Load a checkpoint.

    Args:
        load_path: The path that contains all the model files. e.g., ./model/checkpoints/<model_name>.
    Returns:
        The last step the checkpoint was trained on.
    """

    chpt_dir_path = cfg.CHPT_DIR / cfg.model_name
    chpt_dir_path.mkdir(parents=True, exist_ok=True)


    if chpt_dir_path.is_dir():
        print(f"Loading checkpoint from: {cfg.checkpoint_name}...")
        chpt_file = chpt_dir_path / f"{cfg.model_name}.pt"
        chpt = torch.load(chpt_dir_path, map_location=cfg.device)

        # Load all states into the trainer
        trainer.model.load_state_dict(chpt["model_state_dict"])
        trainer.optimizer.load_state_dict(chpt["optimizer_state_dict"])

        # MPS device bug issue, fix by moving optimizer state tensors to mps device
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)

        if "scheduler_state_dict" in chpt:
            trainer.scheduler.load_state_dict(chpt["scheduler_state_dict"])
        if "step_counter" in chpt:
            trainer.step_counter = chpt["step_counter"]

        last_step = chpt.get("step_counter", 0)
        print(f"\nResuming training from Step {last_step}...")
        return last_step

    else:
        print(f"\n\nCheckpoint not found at {chpt_dir_path}! Check configurations!")
        import sys
        sys.exit(1)

