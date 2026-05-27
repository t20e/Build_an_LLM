import easyjupyter  
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import ConfigTemplate, Scaled_down_text


def plot_loss_history(
    cfg: ConfigTemplate, loss_history: list, lr_history: list, save_path: Path
):
    """Plot for the text-only pre-training run"""
    global_step = cfg.global_step
    pretrain = cfg.text_only.pretrain
    initial_cfg = pretrain.initial_stage

    fig, ax_loss = plt.subplots(figsize=(14, 6))
    ax_lr = ax_loss.twinx()  # Shares x-axis, independent y-axis

    steps = range(len(loss_history))

    # --- Loss
    ax_loss.plot(steps, loss_history, label="Raw Loss", alpha=0.3, color="blue")
    if len(loss_history) > 100:
        window = 100

        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax_loss.plot(
            range(window - 1, len(loss_history)),
            smoothed,
            label="Smoothed Loss",
            color="red",
        )

    # --- Learning Rate
    if lr_history:
        ax_lr.plot(
            range(len(lr_history)),
            lr_history,
            label="Learning Rate",
            color="green",
            alpha=0.7,
            linewidth=1,
            linestyle="--",
        )
        ax_lr.set_ylabel("Learning Rate", color="green")
        ax_lr.tick_params(axis="y", labelcolor="green")
        ax_lr.set_yscale("log")

    # --- Mark stage boundaries with vertical lines
    stage_boundaries = {
        "→ lc_stage": (
            initial_cfg.steps_completed
            if pretrain.curr_stage_name != "initial_stage"
            else None
        ),
        "→ annealing_stage": (
            initial_cfg.steps_completed
            + (pretrain.lc_stage.tokens // (cfg.curr_batch_size or 1))
            if pretrain.curr_stage_name == "annealing_stage"
            else None
        ),
    }

    for label, boundary_step in stage_boundaries.items():
        if boundary_step and 0 < boundary_step < len(loss_history):
            ax_loss.axvline(x=boundary_step, color="gray", linestyle="--", alpha=0.7)
            ax_loss.text(
                boundary_step,
                ax_loss.get_ylim()[1],
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=9,
                color="gray",
            )

    # --- Layout
    ax_loss.set_title(f"Pre-Training Loss - {cfg.model_name} (Global_step: {global_step:,})")
    ax_loss.set_xlabel("Global Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Combine legends from both axes into one
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_lr, labels_loss + labels_lr, loc="upper right")

    config_info = (
        f"Model: d_model={cfg.d_model}, N={cfg.num_layers}, h ={cfg.attn_heads}, d_ff={cfg.d_ff}, vocab_size={cfg.vocab_size:,}\n"
        f"See all hyperparameters in config.json"
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

    plot_img_path = save_path / "training_loss_plot.png"

    plt.savefig(plot_img_path)
    print(f"Loss plot saved to → {plot_img_path}")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    from configs import  Scaled_down_text
    
    loss_history = np.linspace(0.99, 0.01, 1000).tolist()
    loss_lr = np.linspace(0.1, 0.0001, 1000).tolist()
    Scaled_down_text.initialize()

    save_path = Scaled_down_text.chpts_dir / f"global_step_{10}"
    save_path.mkdir(parents=True, exist_ok=True)

    plot_loss_history(Scaled_down_text, loss_history, loss_lr, save_path)