import easyjupyter  # type: ignore

import argparse
from configs import ConfigTemplate, Llama3_1_405B, Scaled_down_text
from model.model_text_only import TextOnlyModel
from model.text_only_training.pre_training import PreTrainTextOnly
from model.utils.model_io import load_checkpoint
from model.optimizer_and_lr_schedule import get_optimizer

from utils.misc import print_args


MODELS_SUITE = {
    "scaled_down_text": {
        "config_cls": Scaled_down_text,
        "model_cls": TextOnlyModel,
        "is_multimodal": False,
        "train_phase": {
            "pretrain": {
                "trainer": PreTrainTextOnly,
            },
            "sft": {
                "trainer": None,
            },
            "dpo": {
                "trainer": None,
            },
        },
    },
    "scaled_down_multi": {
        # Multi-model (Vision + text) 3.2 11b model
        "config_cls": None,
        "model_cls": None,
        "is_multimodal": True,
        "train_phase": {
            "sft": {
                "trainer": None,
            },
            "dpo": {
                "trainer": None,
            },
        },
    },
    # =======================================================================
    # External Meta Trained models from HuggingFace (Inference only)
    # =======================================================================
    # "3.1_8b": {
    #     "config_cls": Llama3_8B,
    #     "model_cls": None,
    #     "is_multimodal": False,
    # },
    # "3.2_11b_multi": {
    #     "config_cls": None,
    #     "model_cls": None,
    #     "is_multimodal": True,
    # },
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
    parser.add_argument("--model", type=str, default="scaled_down_text")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "sft", "dpo", "inference"],
        default="pretrain",
        help="The phase to train the model in.",
        required=True,
    )
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Transition to the next phase, only load the checkpoint's weights (no optimizer/scheduler) from the previous phase.",
    )
    parser.add_argument(
        "--chpt_dir",
        type=str,
        help="Specific num_grad_updates checkpoint directory to load. e.g., num_grad_updates_001000",
    )

    parser.add_argument(
        "--overfit",
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

    if args.chpt_dir:
        cfg = ConfigTemplate.load(args.chpt_dir)
    else:
        cfg = MODELS_SUITE[args.model]["config_cls"].initialize(args.overfit)

    print(cfg)

    model = MODELS_SUITE[args.model]["model_cls"](cfg).to(cfg.device)

    if args.transition and not args.chpt_dir:
        raise ValueError("Transition requires checkpoint directory!")

    optimizer = get_optimizer(cfg, model)

    if args.chpt_dir:
        if args.transition:
            # Only load the checkpoint's weights
            load_checkpoint(cfg, model, args.chpt_dir)
            optimizer = None
        else:
            # Load the model's weights and optimizer.
            load_checkpoint(cfg, model, args.chpt_dir, optimizer)

    if args.overfit:
        cfg.prefix = "OVERFIT_"

    if args.phase == "inference":
        print("\n-------- Running inference... --------")
        print("-" * 64)
        print(cfg)
        print("-" * 64)


    else:  # === Train ===
        if args.phase == "pretrain":
            if args.model == "scaled_down_text":
                trainer = PreTrainTextOnly(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    is_overfit=args.overfit,
                )
                trainer.train()
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
