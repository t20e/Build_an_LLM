import torch
from pathlib import Path
import gc


def average_checkpoints(checkpoints_dirs: list[Path], output_dir: Path):
    """
    Perform Polyak Checkpoint Averaging. 

    Should be called once at the end of pre-training to average checkpoints that were saved during the annealing stage.
    Accumulates weights in float32 for numerical stability, then casts each tensor back to its original dtype.
    NOTE: The resulting checkpoint has no optimizer state, it is only meant for post-training or inference/eval!
    """

    if not checkpoints_dirs:
        raise ValueError("No checkpoint directories provided.")

    num_chpts = len(checkpoints_dirs)
    print(f"\n\nAveraging {num_chpts} checkpoints...")
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    def load_model_state(chpt_dir: Path):
        chpt_file = chpt_dir / "checkpoint.pt"
        if not chpt_file.exists():
            raise FileNotFoundError(f"Missing checkpoint.pt in: {chpt_dir}")

        raw = torch.load(chpt_file, map_location="cpu", weights_only=True)
        return raw["model_state_dict"]

    # Load the first checkpoint as a base
    base_state_dict = load_model_state(checkpoints_dirs[0])

    avg_state_dict = {}
    orig_dtypes = {}

    # Up cast to float32 for numerical stability
    for key, tensor in base_state_dict.items():
        if tensor.dtype in float_dtypes:
            orig_dtypes[key] = tensor.dtype
            avg_state_dict[key] = tensor.float().clone()
        else:
            avg_state_dict[key] = tensor.clone()

    # Iteratively sum the weights
    for chpt_dir in checkpoints_dirs[1:]:
        print(f"Adding {chpt_dir.name}...")
        current_state_dict = load_model_state(chpt_dir)

        if current_state_dict.keys() != base_state_dict.keys():
            raise ValueError(f"{chpt_dir.name} has mismatched keys vs base checkpoint.")

        for key in orig_dtypes:
            if current_state_dict[key].shape != avg_state_dict[key].shape:
                raise ValueError(f"Shape mismatch for '{key}' in {chpt_dir.name}.")
            avg_state_dict[key] += current_state_dict[key].float()
    

    # Divide by N to get the true average
    print("computing final average...")
    for key in orig_dtypes:
        # Down cast back down to original dtype
        avg_state_dict[key] = (avg_state_dict[key] / num_chpts).to(orig_dtypes[key])


    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": avg_state_dict,
        },
        output_dir / "checkpoint.pt",
    )
    print(f"Averaged checkpoint saved to {output_dir / 'checkpoint.pt'}")