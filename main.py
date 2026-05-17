import easyjupyter  # type: ignore
import argparse
from llama_configs import Llama3_scaled_down, Llama3_8B, BaseConfig
from model.model_text_only import TextOnlyModel
from model.pre_train_text_only import PreTrainTextOnly, setupPreTraining
from utils.fs_manager import resolve_checkpoint_path
from utils.misc import print_args

MODELS_SUITE = {
    "scaled_down_text": {
        # 3.1 8b text-only model
        "config_cls": Llama3_scaled_down,
        "model_cls": TextOnlyModel,
        "is_multimodal": False,
        "train_pipeline": {
            "pretrain": {
                "chpt_dir": "Scaled_down_Llama_3_1_Base",
                "trainer": PreTrainTextOnly,
                "suffix": "_Base",
            },
            "sft": {
                "chpt_dir": "Scaled_down_Llama_3_1_SFT",
                "suffix": "_SFT",
                "trainer": None,
            },
            "dpo": {
                "chpt_dir": "Scaled_down_Llama_3_1_Instruct",
                "suffix": "_Instruct",
                "trainer": None,
            },
        },
    },
    "scaled_down_multi": {
        # Multi-model (Vision + text) 3.2 11b model
        "config_cls": None,
        "model_cls": None,
        "is_multimodal": True,
        "train_pipeline": {
            "sft": {
                "chpt_dir": "Scaled_down_Llama_3_11_multi_SFT",
                "suffix": "_Multi_SFT",
                "trainer": None,
            },
            "dpo": {
                "chpt_dir": "Scaled_down_Llama_3_11_multi_Instruct",
                "suffix": "_Multi_Instruct",
                "trainer": None,
            },
        },
    },
    # =======================================================================
    # External Meta Trained models from HuggingFace (Inference only)
    # =======================================================================
    "3.1_8b": {
        "config_cls": Llama3_8B,
        "model_cls": None,
        "is_multimodal": False,
    },
    "3.2_11b_multi": {
        "config_cls": None,
        "model_cls": None,
        "is_multimodal": True,
    },
}


if __name__ == "__main__":
    """
    Scenario 1: Phase Transition (i.e., from pre-training to post-training SFT, or from SFT to DPO)
      - Only load the model's weights
      - Things like learning rate can be different.
    Scenario 2: Continue Training in the same phase.
      - Load the model's weights and optimizer, and scheduler.
    Scenario 3: Inference on either my trained model, or one from HuggingFace.
    """

    parser = argparse.ArgumentParser(description="Llama 3 Training/Inference CLI.")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "sft", "dpo", "inference"],
        default="pretrain",
        help="The phase to train the model in.",
        required=True,
    )

    parser.add_argument("--model", type=str, default="scaled_down_text")
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Transition to the next phase, only load the checkpoint's weights (no optimizer/scheduler) from the previous phase.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific step file to load. e.g., step_001000.pt. Defaults to latest.txt",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    args = parser.parse_args()
    print_args(args)
    if args.model not in MODELS_SUITE:
        raise ValueError(f"Model {args.model} not recognized!")

    if (
        args.model == "3.2_11b_multi"
        or args.model == "3.1_8b"
        and args.phase != "inference"
    ):
        raise ValueError(
            f"Models 3.2_11b_multi and 3.1_8b are only supported for inference!"
        )

    if args.model == "scaled_down_text":
        checkpoint_path, is_transition, new_chpt_save_path = resolve_checkpoint_path(
            BaseConfig, args, MODELS_SUITE["scaled_down_text"]["train_pipeline"]
        )
    elif args.model == "scaled_down_multi":
        checkpoint_path, is_transition, new_chpt_save_path = resolve_checkpoint_path(
            BaseConfig, args, MODELS_SUITE["scaled_down_multi"]["train_pipeline"]
        )

    if is_transition or checkpoint_path is not None:
        # Continue training, load from the config.json
        cfg = MODELS_SUITE[args.model]["config_cls"].load_from_json(
            checkpoint_path.parent / "config.json"
        )
        print(cfg)
    else:
        # Start fresh in pretrain phase
        cfg = MODELS_SUITE[args.model]["config_cls"]()
        print(cfg)

    cfg.CURR_CHPT_DIR = new_chpt_save_path
    model = MODELS_SUITE[args.model]["model_cls"](cfg).to(cfg.device)

    if args.dry_run:
        print("\n------------ Dry Run ------------")
        cfg.CURR_CHPT_DIR = cfg.CHPTS_DIR / "Scaled_down_Llama_3_1_DRY_RUN"
        cfg.CURR_CHPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "inference":
        print("\n-------- Running inference... --------")
        # TODO: For inference, we need to use save tensors
        print("-" * 64)
        print(cfg)
        print("-" * 64)

    else:  # === Train ===
        if args.phase == "pretrain":
            if args.model == "scaled_down_text":
                trainer = setupPreTraining(
                    cfg=cfg,
                    model=model,
                    chpt_path=checkpoint_path,
                    is_transition=is_transition,
                    is_dry_run=args.dry_run,
                )
                trainer.train(is_dry_run=args.dry_run)
            elif args.model == "scaled_down_multi":
                raise NotImplementedError(
                    "Multi-modal pre-training not yet implemented!"
                )
        elif args.phase == "sft":
            if args.model == "scaled_down_text":
                raise NotImplementedError("Text-only SFT not yet implemented!")
            elif args.model == "scaled_down_multi":
                raise NotImplementedError("Multi-modal SFT not yet implemented!")

        elif args.phase == "dpo":
            if args.model == "scaled_down_text":
                raise NotImplementedError("Text-only DPO not yet implemented!")
            elif args.model == "scaled_down_multi":
                raise NotImplementedError("Multi-modal DPO not yet implemented!")
        else:
            raise ValueError(f"Phase {args.phase} not recognized!")
