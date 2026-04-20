import EasyJupyter # type: ignore
import argparse
from llama_configs import Llama3_scaled_down
from model.model_text_only import TextOnlyModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for project.")

    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument(
        "--inference", action="store_true", help="Run inference on a model."
    )
    node_group.add_argument(
        "--pre_train", action="store_true", help="Pre-train a model."
    )
    node_group.add_argument(
        "--post_train", action="store_true", help="Post-train a model."
    )

    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training the model from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_dir_name",
        type=str,
        help="The name of the directory that contains the checkpoint. The directory must have the model's <config>.json and <tokenizer>.json",
    )

    parser.add_argument(
        "--model_summary_prompt",
        action="store_true",
        help="Prompt for a summary of the model.",
    )

    parser.add_argument(
        "--use_config",
        type=str,
        help="Which model configurations to use, defaults to Scaled_down_Llama_3.",
        default="Scaled_down_Llama_3",
    )
    parser.add_argument(
        "--override_default_model_name",
        action="store_true",
        help="Override the default model name that is in teh class.",
    )

    parser.add_argument(
        "--multi_modal_model",
        action="store_true",
        help="Do you want to use the multi-modal model. If not set, it will default to the text-only model.",
    )

    args = parser.parse_args()

    # Parse and validate the arguments
    if args.use_config == "Scaled_down_Llama_3":
        cfg = Llama3_scaled_down()
        print("Using the Scaled down Llama 3.0 model.")
    elif args.use_config == "Llama_3_8B":
        print("Using the Llama 3.8B model.")
    else:
        parser.error(f"Invalid configuration name: {args.use_config}")

    if args.override_default_model_name:
        cfg.model_name = input("Override the default name for this model: ")
        print("Overriding the model name to:", cfg.model_name)

    if args.model_summary_prompt:
        cfg.model_summary_input = input("Enter a summary for this model/run: ")
        print(f"Model summary: {cfg.model_summary_input}")

    if (args.continue_training or args.inference) and not args.checkpoint_dir_name:
        parser.error(
            "--checkpoint_dir_name is required when --continue_training or --inference is set."
        )

    cfg.continue_training = args.continue_training

    if args.continue_training or args.inference:
        print("Loading the model from the checkpoint directory.")
        # TODO use the config that is found in the config.json file from the checkpoint_dir_name directory, and load the model weights
        pass

    if args.multi_modal_model:
        print("Using the multi-modal model.")
        # TODO load the MultiModalModel
    else:
        print("Using the text-only model.")
        # model = TextOnlyModel(cfg)

    if args.post_train:
        print("Post-training the model.")
    elif args.inference:
        print("Running inference.")
    elif args.pre_train:
        print("Pre-training the model.")
