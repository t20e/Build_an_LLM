# TODO ad this to the notebooks!

Why **RoPE** is only on Q and K: The attention mechanism calculates the relevance between tokens using the dot product of Queries and Keys ($QK^T$). By applying rotary transformations only to Q and K, the resulting attention scores mathematically encode the relative distance between tokens. The Values ($V$) represent the actual content payload of the token; rotating the payload does nothing for spatial awareness.

**Why KV is Cached:** This is not explicitly detailed in the Llama papers because it is not unique to Llama. KV Caching is the universal industry standard for autoregressive generation. During inference, you generate one token at a time. Caching the Keys and Values of previously processed tokens prevents the model from redundantly recalculating the attention matrices for the entire context window on every single forward pass.



**Pre-normalization (RMSNorm):** "To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output... We use the RMSNorm normalizing function." (This confirms steps 6 and 10).

**SwiGLU:** "We replace the ReLU non-linearity by the SwiGLU activation function..." (This confirms step 11).

