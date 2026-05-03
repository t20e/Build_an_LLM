import EasyJupyter  # type: ignore
import argparse
import torch
from llama_configs import Llama3_scaled_down, Llama3_8B
from model.model_text_only import TextOnlyModel
from model.utils.model_io import load_checkpoint
from model.tokenizer import BPETokenizer
from model.utils.data_loader import create_causal_dataloader, get_raw_dataset
from model.pre_training import PreTrainModel
from utils.fs_manager import resolve_checkpoint_path
from model.utils.optimizer_and_scheduler import get_optimizer, get_scheduler

PHASES_MAP = {
    # Used with --phase flag
    "pretrain": {
        "chpt_dir": "Scaled_down_Llama_3_1_Base",
        "trainer": PreTrainModel,
        "model": TextOnlyModel,
        "tokenizer": BPETokenizer,
    },
    "sft": {
        "suffix": "_SFT",
        "tokenizer": BPETokenizer,
        "trainer": None,
        "model": None,
        "chpt_dir": "Scaled_down_Llama_3_1_SFT",
    },
    "dpo": {
        "suffix": "_Instruct",
        "trainer": None,
        "chpt_dir": "Scaled_down_Llama_3_1_Instruct",
    },
}

CONFIG_MAP = {
    # Used with --model flag
    "scaled_down": Llama3_scaled_down,
    "3_8b": Llama3_8B,
    # Add more configs here
}


if __name__ == "__main__":

    # Scenario 1: Phase Transition (e.g., from pre-training to post-training SFT, or from SFT to DPO)
    #   - Only load the model's weights
    #   - Things like learning rate can be different.
    # Scenario 2: Continue Training in the same phase.
    #   - Load the model's weights and optimizer, and scheduler.

    parser = argparse.ArgumentParser(description="Llama 3 Training/Inference CLI.")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "sft", "dpo", "inference"],
        default="pretrain",
        help="The phase to train the model in.",
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=CONFIG_MAP.keys(),
        default="scaled_down",
        help="The model configuration to use.",
    )
    parser.add_argument(
        "--from_phase",
        choices=["pretrain", "sft", "dpo"],
        help="Source phase directory to load weights from. If not provided, will use the current phase.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific step file to load. e.g., step_001000.pt",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    # ---------- Parse and validate the arguments ----------
    args = parser.parse_args()
    cfg = CONFIG_MAP[args.model]()

    model = PHASES_MAP[args.phase]["model"](cfg).to(cfg.device)

    checkpoint_path, is_transition, new_save_path = resolve_checkpoint_path(
        cfg, args, PHASES_MAP
    )
    cfg.CURR_CHPT_DIR = new_save_path

    if args.dry_run:
        print("\n------------ Dry Run ------------")
        cfg.gradient_accumulation_steps = 1
        cfg.token_budget = 1 * cfg.global_batch_size_tokens
        cfg.CURR_CHPT_DIR = cfg.CHPTS_DIR / "Scaled_down_Llama_3_1_DRY_RUN"
        cfg.CURR_CHPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "inference":
        print("\n-------- Running inference... --------")
        load_checkpoint(cfg, model, checkpoint_path)
        print("-" * 64)
        print(cfg)
        print("-" * 64)
    else:
        TrainerClass = PHASES_MAP[args.phase]["trainer"]
        if not TrainerClass:
            print(f"Trainer for {args.phase} not implemented yet.")
            exit(1)

        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer=optimizer)
        # Instantiate the trainer with its default optimizer and scheduler. If we are continuing from a checkpoint, we will update them to that of the saved checkpoint's.
        trainer = TrainerClass(cfg, model, optimizer=optimizer, scheduler=scheduler)

        if checkpoint_path:
            cfg.load_config_from_json(checkpoint_path.parent / "config.json")
            cfg.CURR_CHPT_DIR = new_save_path  # Paths are not saved to config.json, to allow moving from one system to another, so set the save path here.
            print("\nWill save the new model to", new_save_path)

            # Check if we have to load the optimizer and scheduler from the checkpoint, if we are transitioning we will only load the model's weights, and not its optimizer and scheduler.
            load_opt = None if is_transition else trainer.optimizer
            load_sch = None if is_transition else trainer.scheduler

            load_checkpoint(
                cfg=cfg,
                model=model,
                chpt_path=checkpoint_path,
                optimizer=load_opt,
                scheduler=load_sch,
            )

        print("-" * 64)
        print(cfg)
        print("-" * 64)
        print(f"\n-------- Starting {args.phase.upper()} Phase... --------")
        tokenizer = PHASES_MAP[args.phase]["tokenizer"](cfg)
        success, loaded_tokenizer = tokenizer.load_tokenizer()
        if not success:
            tokenizer = loaded_tokenizer.train_tokenizer(get_raw_dataset())

        trainer.train(tokenizer)
