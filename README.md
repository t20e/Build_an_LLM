# How To Build An LLM

-# TODO The Llama 1 and 2 are from different papers, anywhere in code where I have a link to the Llama 1 & 2 papers, fix it!

#TODO make sure this renders in the github repo

✨ This project is a guide to building a large language model like ChatGPT, Gemini, or Llama from scratch. I will build and train a scale down version of the Llama 3 architecture (Note: training a full model is very expensive!). Also, I will add options to import pre-trained Llama 3.1 models. I choose Llama over the Gemini and ChatGPT models because it is the most open-sourced and well-documented. Almost all LLMs are built on the Transformer-decoder architecture, with some minor tweaks. To build a full Llama 3 model, you would take my scaled-down config, increase some hyperparameters, adjust the tokenizer so that it is not a universal one, and change the datasets to larger ones.

- There are two phases when training an LLM are: # TODO maybe add this info in a train notebook
  - Phase 1: You train a base model on a massive corpus of raw text using self-supervised learning, where its only objective is to predict the next token (e.g., the next word in a sentence). Here the model learns grammar, facts, and reasoning.
  - Phase 2: You take the base model and train it to become a chat/assistant model.This is done by applying fine tuning using structured conversational data (Prompt/Response pairs), which is often followed by Reinforcement Learning from Human Feedback or direct preference optimization to force the model to behave an assistant.

**Useful Links:**

- [Brendan Bycroft LLMs Visualization](https://bbycroft.net/llm)
- [Andrej Karpathy's Deep Dive into LLMs video](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- [My Transformer Project](https://github.com/t20e/AI_projects_and_res/tree/main/Transformer)

**Goals:**

- [x] Add and pre-process datasets:
  - [x]  [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) subset of the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset. I will only use a small portion of the FineWeb-edu, which is ~5.84 TB, while the FineWeb is ~54.8 TB. Used for the initial and long-context stages of the pre-training.
  - [x]  [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) used for the annealing stage of the pre-training.
- [x] Implement Llama 3 architecture components.
  - [x] Build the [tokenizer](./model/tokenizer.ipynb).
  - [x] [RoPE](./model/RoPE.ipynb)
  - [x] [GQA Attention](./model/GQA.ipynb)
  - [x] [SwiGLU Feed Forward](./model/SwiGLU_FFN.ipynb)
  - [x] [RMSNorm](./model/RMSNorm.ipynb)
  - [x] AdamW Optimizer
  - [x] The transformer [decoder](./model/decoder.ipynb)
- [ ] Build the training pipeline.
  - [ ] Pre-training
  - [ ] Post-training
    - [ ] Supervised Fine-tuning (SFT)
    - [ ] Direct Preference Optimization (DPO)
- [ ] Train a scaled down model along with its tokenizer, that is feasible on my Mac.
- [ ] Import a Pre-trained Llama model (e.g., Llama 3.1 8B) from HuggingFace to showcase a SOTA model working with my built-out architecture.
- [ ] Implement Multi-modal so that the model works with:
  - [ ] Code
  - [ ] Speech
  - [ ] Vision

## Llama 3 Architecture

![Llama 3 Architecture Text-Only](./showcase_images/llama_and_transformer_diagram.png)

- ✨ All the model's layers are implemented in their own notebooks in [./model](./model/).

The fundamental block of an LLM is the **Transformer Decoder**. Most modern frontier LLMs modify the decoder by adding a **RMSNorm**, **RoPE**, and **GQA** sub-layer. There are other variations, for example the [Google Gemma model](https://developers.googleblog.com/gemma-explained-new-in-gemma-2/#:~:text=the%20new%20models%3A-,Key%20Differences,-Gemma%202%20shares) has **GeGLU** non-linearity.

The Llama architecture was first described in [LlaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971), which is Llama 1. The Llama 3 which is described in this paper: [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783), made a few modifications such as:

1. Adding **GQA Attention** with $\mathbf{8}$ key-value heads.
2. Used an attention mask that prevents self-attention between different documents withing the same sequence.
3. Used a vocabulary with $128\text{K}$ total tokens.
   1. Of which $100\text{K}$ is from the **tiktoken** library, and the other $28\text{K}$ is additional tokens to better support non-English languages.
4. Increased the **RoPE** base frequency hyperparameter to $500{,}000$


**#TODOs:**

- [ ] Add info of the Transformers decoder
- [ ] Explain the differences between how chatgpt and gemini are implemented.
- [ ] Add papers from GoodNotes into ./papers
