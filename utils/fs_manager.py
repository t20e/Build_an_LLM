from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_configs import BaseConfig


def resolve_checkpoint_path(cfg: BaseConfig, args, PHASE_MAP):
    """
    Returns
        full_checkpoint_path: based on the --from_phase flag,
        is_transition (bool): if the checkpoint is from a different phase
        new_save_path (str): The path of the new location to save models to, e.g., if we go from SFT to DPO, the new_save_path will be to the location that has DPO models
    """

    # Determine which checkpoint directory we will be working in
    new_save_path = cfg.CHPTS_DIR / PHASE_MAP[args.phase]["chpt_dir"]
    new_save_path.mkdir(parents=True, exist_ok=True)

    # Determine from which config directory the checkpoint from, either from another phase, or continue training from the same phase
    # if --from_phase is provided, we are transitioning, otherwise we are continuing training in the same phase
    source_phase = args.from_phase if args.from_phase else args.phase
    source_dir = cfg.CHPTS_DIR / PHASE_MAP[source_phase]["chpt_dir"]

    is_transition = (args.from_phase is not None) and (args.from_phase != args.phase)

    checkpoint_file = args.checkpoint
    if (
        not checkpoint_file
    ):  # If the checkpoint was not provide, get from the latest.txt
        latest_file = source_dir / "latest.txt"
        if latest_file.exists():
            checkpoint_file = latest_file.read_text().strip()
        elif args.phase == "pretrain" and not is_transition:
            # Start fresh in pretrain phase
            return None, False, new_save_path
        else:
            raise FileNotFoundError(
                f"No checkpoint or latest.txt found in {source_dir}."
            )
    return (source_dir / checkpoint_file), is_transition, new_save_path