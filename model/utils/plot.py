import easyjupyter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import FuncFormatter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import ConfigTemplate, Scaled_down_text


def plot_loss_history(
    cfg: ConfigTemplate,
    loss_history: list,
    lr_history: list,
    save_path: Path,
    val_loss_history: list[tuple[int, float]] = None,
    diagnostic_history: list[dict] = None,
):
    """
    Plot for the text-only pre-training run

    Args:
        diagnostic_history: A history of the warmup ending, seq_len, micro_batch, global_batch_tokens, total_tokens_processed
    """
    plt.style.use("default")

    num_grad_updates = cfg.num_grad_updates
    pretrain = cfg.text_only.pretrain

    start_step = num_grad_updates - len(loss_history) + 1
    steps = range(start_step, num_grad_updates + 1)
    ax_shapes = None

    if diagnostic_history and len(diagnostic_history) == len(loss_history):
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

    if val_loss_history:
        val_points = [
            v for v in val_loss_history if start_step <= v[0] <= num_grad_updates
        ]
        if val_points:
            val_steps_x = [v[0] for v in val_points]
            val_losses_y = [v[1] for v in val_points]
            ax_loss.plot(
                val_steps_x,
                val_losses_y,
                marker="o",
                linestyle="--",
                color="magenta",
                label="'Online' Validation Loss",
                markersize=6,
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
    if ax_shapes and diagnostic_history:
        seq_lens = [s["seq_len"] for s in diagnostic_history]
        micro_batches = [s["micro_batch"] for s in diagnostic_history]
        global_batch_tokens = [s["global_batch_tokens"] for s in diagnostic_history]

        # Primary axis (seq_len)
        ax_shapes.plot(
            steps,
            seq_lens,
            color="purple",
            label="Sequence Length (seq_len)",
            alpha=0.8,
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
        lines_m, labels_m = ax_micro.get_legend_handles_labels()
        ax_shapes.legend(
            lines_s + lines_t + lines_m,
            labels_s + labels_t + labels_m,
            loc="upper left",
        )
        ax_shapes.grid(True, alpha=0.3)

    # ==========
    # Add vertical line markers (Stage boundaries and warmup stage)
    # ==========

    stage_boundaries = {
        "→ Warmup End & Initial stage start": pretrain.initial_stage.warmup_steps,
        "→ lc_stage Start": pretrain.transition_steps.get("lc_stage"),
        "→ annealing_stage Start": pretrain.transition_steps.get("annealing_stage"),
        "→ annealing_stage End": pretrain.transition_steps.get("complete"),
    }
    print(stage_boundaries)

    for label, boundary_step in stage_boundaries.items():
        if boundary_step and start_step <= boundary_step <= num_grad_updates:
            ax_loss.axvline(x=boundary_step, color="black", linestyle="--", alpha=0.7)

            if ax_shapes:
                ax_shapes.axvline(
                    x=boundary_step, color="black", linestyle="--", alpha=0.7
                )

            ax_loss.text(
                boundary_step,
                ax_loss.get_ylim()[1] * 0.75,
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=7.5,
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

    # --- Layout & Metadata
    ax_loss.set_title(
        f"{"OVERFIT " if cfg.is_overfit else ''}Pre-Training Diagnostics - {cfg.model_name} (num_grad_updates: {num_grad_updates:,})",
        fontsize=12,
        fontweight="bold",
    )

    if not ax_shapes:
        ax_loss.set_xlabel("num_grad_updates\n(Total Tokens Processed)")
    else:
        ax_shapes.set_xlabel("num_grad_updates\n(Total Tokens Processed)")

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
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if diagnostic_history and len(diagnostic_history) == len(loss_history):
        token_snapshots = [s["total_tokens_processed"] for s in diagnostic_history]

        def step_and_token_formatter(x, pos):
            idx = int(round(x)) - start_step
            # Keep bounds safe during plot rendering lookups
            if 0 <= idx < len(diagnostic_history):
                tokens = token_snapshots[idx]

                # Format to thousands, Millions for scannable axis labels
                if tokens >= 1e9:
                    token_str = f"{tokens / 1e9:.3f}B"
                elif tokens >= 1e6:
                    token_str = f"{tokens / 1e6:.3f}M"
                elif tokens >= 1e3:
                    token_str = f"{tokens / 1e3:.3f}K"
                else:
                    token_str = f"{tokens:,}"
                return f"{int(round(x))}\n({token_str})"

            return f"{int(round(x))}"

        active_ax = ax_shapes if ax_shapes else ax_loss
        active_ax.xaxis.set_major_formatter(FuncFormatter(step_and_token_formatter))

    # Force specific ticks on x-axis so the num_grad_updates of the lc_stage ad annealing_stage is written on ploy
    curr_ticks = active_ax.get_xticks()
    explicit_ticks = [
        pretrain.transition_steps.get("lc_stage"),
        pretrain.transition_steps.get("annealing_stage"),
        num_grad_updates,  # Final step
    ]
    combined_ticks = set(curr_ticks) | {t for t in explicit_ticks if t is not None}
    valid_ticks = sorted(
        [t for t in combined_ticks if start_step <= t <= num_grad_updates]
    )
    active_ax.set_xticks(valid_ticks)

    plt.setp(active_ax.get_xticklabels(), rotation=45, ha="right")

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
        / f"num_grad_updates_{Scaled_down_text.num_grad_updates}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    plot_loss_history(Scaled_down_text, loss_history, loss_lr, save_path)
