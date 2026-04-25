# TODO make it work for this project

# import argparse
# import torch
# from safetensors.torch import save_file
# from pathlib import Path
# from safetensors.torch import load_model
# import os
# import torch

# from configs.english_german_config import English_german_config
# from model.Transformer import Transformer


# def export_for_sharing(bulky_checkpoint_path: Path, output_filename: str):
#     """
#     Extract only the weights from a .pt or .pth file and save it as a .safetensors file. Which the modern way to safely share weights.

#     Args:
#         bulky_checkpoint_path: Full path to the bulky checkpoint file. e.g, ../../checkpoint_100.pt
#         save_path: The new filename to save the safetensors file to.
#     """

#     suffixes = (".pth", ".pt", "..")
#     if not output_filename.endswith(suffixes):
#         output_filename += ".safetensors"
#     else:
#         for s in suffixes: 
#             if output_filename.endswith(s):
#                 output_filename = output_filename.removesuffix(s)
#                 output_filename += ".safetensors"
#                 break

#     parent_dir = bulky_checkpoint_path.parent
#     save_path = parent_dir / output_filename

#     print(f"\n\nExporting bulky checkpoint -> {bulky_checkpoint_path}")

#     # load to CPU
#     checkpoint = torch.load(
#         bulky_checkpoint_path, map_location="cpu", weights_only=False
#     )

#     # Extract only the weights
#     weights = checkpoint["model_state_dict"]


#     # --------- FOR TRANSFORMER ONLY ---------
#     # Because the some of the weights are tied, to fix duplication error:
#     unique_weights = {}
#     shared_pointers = set()
#     for key, value in weights.items():
#         data_ptr = value.data_ptr()
#         if data_ptr not in shared_pointers:
#             unique_weights[key] = value
#             shared_pointers.add(data_ptr)
#         else:
#             print(f"Skipping shared weight key: {key} (shares memory with another layer)")


#     # ---- Save as a safetensors file ----
#     save_file(unique_weights, save_path)
#     print(f"\nSaved safetensors file to -> {save_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Llama 3 Training/Inference CLI.")
#     parser.add_argument(
#         "--chpt",
#         type=str,
#         help="Full path to the checkpoint .pt or .pth file to export.",
#         required=True,
#     )
#     parser.add_argument("--name", type=str, default="model.safetensors", help="Output filename, e.g., transformer.safetensors")

#     export_for_sharing(
#         Path(parser.parse_args().chpt),
#         output_filename=parser.parse_args().name,
#     )



# def load_trained_model(cfg: English_german_config, device) -> Transformer:
#     """Load a model's safetensors weights"""

#     model = Transformer(cfg=cfg)

#     weights_path = os.path.join(cfg.MODEL_DIR, "checkpoints", cfg.safetensors_filename)

#     if not os.path.exists(weights_path):
#         raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

#     print(f"\nLoading weights from ({weights_path})...")
#     load_model(model, weights_path)

#     # Re-establish the tied weights
#     model.tie_weights()
#     model.to(device)
#     model.eval()
#     return model