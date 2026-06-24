import easyjupyter

from pathlib import Path
from typing import List, Tuple
import argparse
import torch
from configs import ConfigTemplate
from model.bpe_tokenizer import BPETokenizer
from model.model_text_only import TextOnlyModel
from model.utils.masking import create_causal_doc_mask
from tqdm import tqdm
from utils.misc import print_args

from model.utils.model_io import load_checkpoint
import json

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


class Eval_wrapper(LightevalModel):
    """
    Wrapper for `Lighteval`, to make it work with my custom model.
    """

    def __init__(self, cfg, custom_model, eval_config, custom_tokenizer):
        super().__init__()

        # Expose the config public attribute so Lighteval's logger can read it
        self._config = eval_config
        self.config = eval_config

        self._cache = SampleCache(eval_config)

        self.cfg = cfg
        self.model = custom_model
        self._tokenizer = custom_tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return self.cfg.curr_seq_len

    @property
    def add_special_tokens(self) -> bool:
        return False

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: List[Doc]) -> List[ModelResponse]:
        """
        Compute log probabilities of continuations.

        Args:
            docs: list of documents containing context and continuation pairs

        Returns:
            list of model responses with log probabilities
        """
        results = []
        with torch.inference_mode():
            for doc in tqdm(docs, desc="Calculating Loglikelihood"):
                context, continuation = doc.query, doc.choices

                # Tokenize
                context_ids = self.tokenizer.encode(context).ids
                continuation_ids = self.tokenizer.encode(continuation).ids
                con_len = len(continuation_ids)

                # Truncate to fit model's precomputed RoPE context window
                max_seq_len = self.model.freqs_cis.shape[0]
                allowed_context_len = max_seq_len - con_len

                if allowed_context_len < 1:
                    # If continuation alone is larger than context window, truncate it too
                    continuation_ids = continuation_ids[: max_seq_len - 1]
                    context_ids = [context_ids[-1]]
                    con_len = len(continuation_ids)
                elif len(context_ids) > allowed_context_len:
                    # Truncate context from the left
                    context_ids = context_ids[-allowed_context_len:]

                # Combine them to feed the model
                input_ids = torch.tensor(
                    [context_ids + continuation_ids], device=self.cfg.device
                )
                batch_mask = create_causal_doc_mask(self.cfg, token_ids=input_ids)

                logits, _ = self.model(input_ids, batch_mask)

                # logit[:, i, :] predicts token i+i, so the continuation's predicting
                # logits sit one position before each continuation token
                cont_logits = logits[0, -con_len - 1 : -1, :]
                cont_ids_tensor = torch.tensor(continuation_ids, device=self.cfg.device)

                log_probs = torch.log_softmax(cont_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    1, cont_ids_tensor.unsqueeze(-1)
                ).squeeze(-1)
                total_logprob = token_log_probs.sum().item()

                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = bool((greedy_tokens == cont_ids_tensor).all().item())

                results.append(
                    ModelResponse(
                        logprobs=[total_logprob], argmax_logits_eq_gold=[is_greedy]
                    )
                )

        return results

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: List[Doc]) -> List[ModelResponse]:
        """
        Compute rolling log probabilities of sequences.

        Args:
            docs: list of documents containing text sequences

        Returns:
            list of model responses with rolling log probabilities
        """
        raise NotImplementedError(
            "`Eval_wrapper.loglikelihood_rolling()` is not implemented yet!"
        )

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        """
        Generate text until stop sequence or max tokens.

        Args:
            docs: list of documents containing prompts and generation parameters

        Returns:
            list of model responses with generated text
        """

        results = []
        with torch.inference_mode():
            for doc in tqdm(docs, desc="Generating"):
                # Extract the prompt and generation arguments
                prompt, max_tokens, stop_sequences = (
                    doc.query,
                    doc.generation_size,
                    doc.stop_sequences,
                )

                if isinstance(stop_sequences, str):
                    stop_sequences = [stop_sequences]

                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt).ids
                prompt_len = len(input_ids)
                max_seq_len = self.model.freqs_cis.shape[0]

                # Truncate the prompt to fit model's precomputed RoPE context window minus 1 token to make space for an answer/response
                if prompt_len >= max_seq_len:
                    input_ids = input_ids[-(max_seq_len - 1) :]
                    prompt_len = len(input_ids)

                # Shrink max_tokens so prompt + generated tokens do no exceed max_seq_len
                if prompt_len + max_tokens > max_seq_len:
                    max_tokens = max_seq_len - prompt_len

                input_tensor = torch.tensor([input_ids], device=self.cfg.device)

                # Process prompt entirely and initial KV cache
                batch_mask = create_causal_doc_mask(
                    cfg=self.cfg, token_ids=input_tensor
                )
                logits, kv_cache = self.model(
                    input_tensor, batch_mask, start_pos=0, kv_cache=None
                )
                next_token_id = logits[0, -1, :].argmax(dim=-1).item()

                generated_tokens = []

                start_pos = input_tensor.shape[1]

                # Increment token by token processing via KV Cache
                for _ in range(max_tokens - 1):
                    if next_token_id == self.cfg.special_tokens["doc_end_token"]["ID"]:
                        # If we hit the `doc_end_token` we can break
                        break

                    generated_tokens.append(next_token_id)

                    # Feed only the single newest token ID
                    next_tensor = torch.tensor(
                        [[next_token_id]], device=self.cfg.device
                    )

                    # Single token attention tracking requires a simple matching 1x1 mask matrix
                    single_mask = torch.ones(
                        (1, 1), device=self.cfg.device, dtype=torch.bool
                    )

                    logits, kv_cache = self.model(
                        next_tensor, single_mask, start_pos=start_pos, kv_cache=kv_cache
                    )
                    start_pos += 1
                    next_token_id = logits[0, -1, :].argmax(dim=-1).item()

                generated_text = self.tokenizer.decode(generated_tokens)

                # Stop at the earliest occurrence of a string from stop_sequences in the generated_text
                stop_indices = [
                    generated_text.index(s)
                    for s in stop_sequences
                    if s in generated_text
                ]
                if stop_indices:
                    generated_text = generated_text[: min(stop_indices)]

                results.append(ModelResponse(text=[generated_text]))

            return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3 Pre-Training Evaluation CLI.")
    parser.add_argument(
        "--chpt",
        type=str,
        help="Path to the checkpoint directory that contains the checkpoint.pt file.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
    )

    args = parser.parse_args()
    print_args(args)

    chpt_step_dir = Path(args.chpt)

    cfg = ConfigTemplate.load(chpt_step_dir)
    cfg.initialize(is_overfit=True if args.overfit else False)

    t = BPETokenizer(cfg)
    success, tokenizer = t.load_tokenizer()
    if not success:
        raise ValueError("Tokenizer file not found!\n")

    model = TextOnlyModel(cfg).to(cfg.device)
    chpt = load_checkpoint(cfg, model=model, chpt_dir=chpt_step_dir)

    model.eval()

    eval_config = CustomModelConfig(
        model_name=cfg.model_name,
        model_definition_file_path=str(chpt_step_dir / "checkpoint.pt"),
    )

    eval_model = Eval_wrapper(cfg, model, eval_config, tokenizer)

    curr_stage = cfg.text_only.pretrain.curr_stage_name

    if curr_stage == "lc_stage":
        """Benchmarks: Benchmarks: Needle-in-a-Haystack, ZeroSCROLLS, InfiniteBench.

        # NOTE: Lighteval doesn't not support Needle-in-a-Haystack, ZeroSCROLLS, or InfiniteBench yet, because those benchmarks need to be ran dynamically, and Lighteval mandates that benchmarks must be pre-processed for security reasons. I could code those benchmarks, but it would take a couple of days, so I instead decided to skip it. Also, I tried lm-evaluation-harness, but that was it's own headache!
        """
        print("\n\nEvaluating checkpoint saved during the Long-Context Stage.\n\n")
        # tasks_to_run = ""
        raise NotImplementedError("Long-Context benchmarks: Needle-in-a-Haystack, ZeroSCROLLS, and InfiniteB not yet implemented!")

    elif curr_stage == "annealing_stage" or curr_stage == "complete":
        print(
            "\n\nEvaluating checkpoint saved during the Annealing Stage or the final averaged checkpoint.\n\n"
        )
        """Benchmarks: GSM8k and MATH."""
        tasks_to_run = "gsm8k|5,math|4"

    evaluationTracker = EvaluationTracker(
        output_dir=str(chpt_step_dir),
        save_details=True,  # Saves exact predictions in a parquet file
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=cfg.text_only.pretrain.max_eval_ds_samples,  # If max_eval_ds_samples != None -> Retrieve a limited number of samples from the dataset, else evaluate against the entire dataset.
    )

    pipeline = Pipeline(
        tasks=tasks_to_run,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluationTracker,
        model=eval_model,
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    print(f"\n\nEvaluation results saved to -> {chpt_step_dir}")
