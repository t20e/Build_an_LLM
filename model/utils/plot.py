import matplotlib.pyplot as plt
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_configs import BaseConfig


def plot_loss_history(loss_history, cfg: BaseConfig, step_counter: int):
    """
    Args:
        loss_history: The loss history during training.
        step_counter: The step counter that was reached during training.
    """

    plt.figure(figsize=(12, 8))
    plt.plot(loss_history, label="Raw Loss", alpha=0.3, color="blue")

    if len(loss_history) > 100:
        window = 100
        import numpy as np

        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(loss_history)),
            smoothed,
            label="Smoothed Loss",
            color="red",
        )

    plt.title(f"Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    config_info = (
        f"Model: d_model={cfg.d_model}, N={cfg.num_layers}, h ={cfg.attn_heads}, d_ff={cfg.d_ff}\n"
        f"Training: vocab_size={cfg.vocab_size}, total_training_steps={cfg.total_training_steps},"
        f"num_workers={cfg.num_workers}, warmup_steps={cfg.warmup_steps},\n"  # TODO any any more new hyperparameters
    )

    # Place config info at bottom of plot
    plt.figtext(
        0.5,
        0.01,
        config_info,
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        bbox={"facecolor": "orange", "alpha": 0.1, "pad": 10},
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    plot_img_path = (
        cfg.CURR_CHPT_DIR / f"step_{step_counter:06d}_training_loss_plot.png"
    )

    plt.savefig(plot_img_path)
    print(f"Loss plot saved to {plot_img_path}")
    # plt.close()
    plt.show()
