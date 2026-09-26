"""GW1B — the shared JAX toolchain for the GW "$10,000 Language Model Challenge".

Everything students need to design, train, evaluate and release a ~1B-parameter
decoder-only language model on the GWU Pegasus cluster:

    gw1b.config      – YAML / dataclass configuration (model, data, optimizer, run)
    gw1b.model       – Llama-style decoder in Flax NNX (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings)
    gw1b.tokenizer   – train / load SentencePiece tokenizers (BPE or unigram)
    gw1b.data        – token-shard writer + memory-mapped, deterministic, sharded batch loader
    gw1b.train       – sharded (FSDP) training loop with Orbax checkpoints and auto-resume
    gw1b.evaluate    – validation loss / perplexity
    gw1b.generate    – greedy / temperature sampling with a KV cache
    gw1b.lm_eval_adapter – plug the JAX model into EleutherAI's lm-evaluation-harness
    gw1b.export_hf   – convert a checkpoint to Hugging Face Llama format for release
    gw1b.budget      – FLOPs / GPU-hour / wall-clock calculator ("compute as an experimental variable")
"""

__version__ = "2026.09.0"
