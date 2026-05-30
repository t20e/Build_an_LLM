import easyjupyter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import ConfigTemplate, Scaled_down_text


def plot_loss_history(
    cfg: ConfigTemplate,
    loss_history: list,
    lr_history: list,
    save_path: Path,
    shape_history: list[dict] = None,
):
    """
    Plot for the text-only pre-training run

    Args:
        shape_history: A history of the warmup ending, seq_len, micro_batch, global_batch_tokens
    """
    global_step = cfg.global_step
    pretrain = cfg.text_only.pretrain
    steps = range(len(loss_history))
    ax_shapes = None

    if shape_history and len(shape_history) == len(loss_history):
        fig, (ax_loss, ax_shapes) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    else:
        fig, ax_loss = plt.subplots(figsize=(14, 6))

    ax_lr = ax_loss.twinx()  # Shares x-axis, independent y-axis

    # ==========
    # Loss and Learning rate trackers
    # ==========
    ax_loss.plot(steps, loss_history, label="Raw Loss", alpha=0.3, color="blue")
    if len(loss_history) > 1:
        window = min(100, max(2, len(loss_history) // 10))

        if len(loss_history) >= window:
            smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
            offset = (window - 1) // 2
            smoothed_steps = steps[offset : offset + len(smoothed)]
            ax_loss.plot(
                smoothed_steps,
                smoothed,
                label=f"Smoothed Loss (window={window})",
                color="red",
                linewidth=1.5,
            )

    if lr_history:
        ax_lr.plot(
            steps,
            lr_history,
            label="Learning Rate",
            color="green",
            alpha=0.7,
            linewidth=1,
            linestyle="--",
        )
        ax_lr.set_ylabel("Learning Rate", color="green")
        ax_lr.tick_params(axis="y", labelcolor="green")
        ax_lr.set_yscale("symlog", linthresh=1e-7)

    # ==========
    # Shape trackers (seq_len, micro_batch, global_batch_tokens)
    # ==========
    if ax_shapes and shape_history:
        seq_lens = [s["seq_len"] for s in shape_history]
        micro_batches = [s["micro_batch"] for s in shape_history]
        global_batch_tokens = [s["global_batch_tokens"] for s in shape_history]

        # Primary axis (seq_len)
        ax_shapes.plot(
            steps,
            seq_lens,
            color="purple",
            label="Sequence Length (seq_len)",
            linewidth=1.5,
        )
        ax_shapes.set_ylabel("Sequence Length", color="purple")
        ax_shapes.tick_params(axis="y", labelcolor="purple")

        # Twin Axis 1 (global_batch_tokens)
        ax_tokens = ax_shapes.twinx()
        ax_tokens.plot(
            steps,
            global_batch_tokens,
            color="orange",
            label="Global Batch Tokens",
            alpha=0.8,
            linestyle="-.",
        )
        ax_tokens.set_ylabel("Global Batch Tokens", color="orange")
        ax_tokens.tick_params(axis="y", labelcolor="orange")

        # Twin Axis 2 (micro_batch_size)
        ax_micro = ax_shapes.twinx()
        # Offset the right spine so it doesn't overlap with ax_tokens
        ax_micro.spines["right"].set_position(("axes", 1.08))
        ax_micro.plot(
            steps,
            micro_batches,
            color="cyan",
            label="Micro Batch Size",
            alpha=0.8,
            linestyle=":",
        )
        ax_micro.set_ylabel("Micro Batch Size", color="cyan")
        ax_micro.tick_params(axis="y", labelcolor="cyan")

        # Merge lower subplots
        lines_s, labels_s = ax_shapes.get_legend_handles_labels()
        lines_t, labels_t = ax_tokens.get_legend_handles_labels()
        ax_shapes.legend(lines_s + lines_t, labels_s + labels_t, loc="upper left")
        ax_shapes.grid(True, alpha=0.3)

    # ==========
    # Add vertical line markers (Stage boundaries and warmup stage)
    # ==========

    stage_boundaries = {
        "→ Warmup End": pretrain.initial_stage.warmup_steps,
        "→ lc_stage": pretrain.transition_steps.get("lc_stage"),
        "→ annealing_stage": pretrain.transition_steps.get("annealing_stage"),
    }

    for label, boundary_step in stage_boundaries.items():
        if boundary_step and 0 < boundary_step < len(loss_history):
            ax_loss.axvline(x=boundary_step, color="gray", linestyle="--", alpha=0.7)

            if ax_shapes:
                ax_shapes.axvline(
                    x=boundary_step, color="gray", linestyle="--", alpha=0.7
                )

            ax_loss.text(
                boundary_step,
                ax_loss.get_ylim()[1] * 0.95,
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=9,
                color="dimgray",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )

    # --- Layout & Metadata
    ax_loss.set_title(
        f"Pre-Training Diagnostics - {cfg.model_name} (Global_step: {global_step:,})",
        fontsize=12,
        fontweight="bold",
    )

    if not ax_shapes:
        ax_loss.set_xlabel("Global Step")
    else:
        ax_shapes.set_xlabel("Global Step")

    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    # Combine legends from both axes into one
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_lr, labels_loss + labels_lr, loc="upper right")

    config_info = (
        f"Model: d_model={cfg.d_model}, num_layers={cfg.num_layers}, attn_heads={cfg.attn_heads}, d_ff={cfg.d_ff}, vocab_size={cfg.vocab_size:,}\n"
        f"See all hyperparameters in config.json"
    )

    # Place config info at bottom of plot
    plt.figtext(
        0.5,
        0.01,
        config_info,
        wrap=True,
        horizontalalignment="center",
        fontsize=9,
        bbox={"facecolor": "orange", "alpha": 0.05, "pad": 8},
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    plot_img_path = save_path / "training_loss_plot.png"

    plt.savefig(plot_img_path, dpi=150)
    print(f"Loss plot saved to → {plot_img_path}")
    plt.close(fig)


if __name__ == "__main__":
    from configs import Scaled_down_text

    loss_history = np.linspace(0.99, 0.01, 1000).tolist()
    loss_lr = np.linspace(0.1, 0.0001, 1000).tolist()
    Scaled_down_text.initialize()

    save_path = (
        Scaled_down_text.chpts_dir
        / Scaled_down_text.text_only.pretrain.save_dir_name
        / f"global_step_{Scaled_down_text.global_step}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    plot_loss_history(Scaled_down_text, loss_history, loss_lr, save_path)
