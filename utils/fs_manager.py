from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_configs import BaseConfig


def resolve_checkpoint_path(cfg: BaseConfig, args, PHASES_CONFIG):
    """
    Returns
        chpt_path: Full path to the .pt file, or None for a fresh pretrain start.
        is_transition (bool): If the checkpoint is from a different phase.
        save_path (str): Directory to save checkpoints for this run.
    """

    pipeline = list(PHASES_CONFIG.keys())

    if args.phase not in pipeline:
        raise ValueError(
            f"Model {args.model} does not support phase {args.phase}.\nIts pipeline is: {pipeline}"
        )

    # Determine which checkpoint directory we will be working in
    save_path = cfg.CHPTS_DIR / PHASES_CONFIG[args.phase]["chpt_dir"]
    save_path.mkdir(parents=True, exist_ok=True)

    if args.transition:
        phase_idx = pipeline.index(args.phase)
        if phase_idx == 0:
            raise ValueError(
                f"Cannot transition into '{args.phase}', as its the first phase."
            )
        source_phase = pipeline[phase_idx - 1]
        is_transition = True
    else:
        source_phase = args.phase
        is_transition = False

    source_dir = cfg.CHPTS_DIR / PHASES_CONFIG[source_phase]["chpt_dir"]
    checkpoint_file = args.checkpoint

    if not checkpoint_file:
        # If the checkpoint was not provide, get from the latest.txt
        latest_file = source_dir / "latest.txt"

        if latest_file.exists():
            checkpoint_file = latest_file.read_text().strip()
        elif not is_transition and source_phase == pipeline[0]:
            # No checkpoint found in the first phase, begin a fresh start
            return None, False, save_path
        else:
            raise FileNotFoundError(
                f"No checkpoint or latest.txt found in {source_dir}."
            )

    return (source_dir / checkpoint_file), is_transition, save_path
