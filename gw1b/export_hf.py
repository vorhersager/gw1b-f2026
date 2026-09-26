"""Export a GW1B checkpoint to Hugging Face `LlamaForCausalLM` format (safetensors) for release.

    python -m gw1b.export_hf --run $GW1B_SCRATCH/checkpoints/gw1b-base --out $GW1B_GROUP/release/GW-1B-Base --verify

The exported directory loads with `transformers.AutoModelForCausalLM.from_pretrained(out_dir)` and can
be uploaded with `huggingface-cli upload`. Only Llama-compatible configurations can be exported:
rmsnorm + rope + swiglu (tied or untied embeddings, MHA/GQA/MQA). Other ablation variants stay in
Orbax format.
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import numpy as np
from flax import nnx

from .config import ModelConfig, TrainConfig
from .model import GW1BModel


def hf_config(m: ModelConfig, tokenizer_ids: dict | None = None) -> dict:
    assert m.norm == "rmsnorm" and m.pos == "rope" and m.activation == "swiglu", \
        "only rmsnorm+rope+swiglu models map onto the HF Llama architecture"
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": m.vocab_size,
        "hidden_size": m.d_model,
        "intermediate_size": m.d_ff,
        "num_hidden_layers": m.n_layers,
        "num_attention_heads": m.n_heads,
        "num_key_value_heads": m.n_kv_heads,
        "head_dim": m.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": m.max_seq_len,
        "rms_norm_eps": m.norm_eps,
        "rope_theta": m.rope_theta,
        "tie_word_embeddings": m.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "initializer_range": m.init_std,
        "torch_dtype": "bfloat16" if m.dtype == "bfloat16" else "float32",
        "use_cache": True,
    }
    ids = tokenizer_ids or {}
    cfg.update({"bos_token_id": ids.get("bos", 1), "eos_token_id": ids.get("eos", 2), "pad_token_id": ids.get("pad", 3)})
    return cfg


def to_hf_state_dict(model: GW1BModel) -> dict[str, np.ndarray]:
    """Map NNX parameters to HF Llama names. NNX Linear kernels are [in, out]; HF wants [out, in]."""
    def A(v):
        return np.asarray(jax.device_get(v[...]))

    sd = {"model.embed_tokens.weight": A(model.embed.embedding)}
    for i, blk in enumerate(model.blocks):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = A(blk.input_norm.scale)
        sd[p + "post_attention_layernorm.weight"] = A(blk.post_attn_norm.scale)
        sd[p + "self_attn.q_proj.weight"] = A(blk.attn.q_proj.kernel).T
        sd[p + "self_attn.k_proj.weight"] = A(blk.attn.k_proj.kernel).T
        sd[p + "self_attn.v_proj.weight"] = A(blk.attn.v_proj.kernel).T
        sd[p + "self_attn.o_proj.weight"] = A(blk.attn.o_proj.kernel).T
        sd[p + "mlp.gate_proj.weight"] = A(blk.mlp.gate_proj.kernel).T
        sd[p + "mlp.up_proj.weight"] = A(blk.mlp.up_proj.kernel).T
        sd[p + "mlp.down_proj.weight"] = A(blk.mlp.down_proj.kernel).T
    sd["model.norm.weight"] = A(model.final_norm.scale)
    if not model.cfg.tie_embeddings:
        sd["lm_head.weight"] = A(model.lm_head.kernel).T
    return {k: np.ascontiguousarray(v) for k, v in sd.items()}


def export(model: GW1BModel, out_dir: str, tokenizer_path: str | None = None, dtype: str = "bfloat16",
           model_card: dict | None = None) -> str:
    from safetensors.numpy import save_file
    import ml_dtypes

    os.makedirs(out_dir, exist_ok=True)
    sd = to_hf_state_dict(model)
    if dtype == "bfloat16":
        sd = {k: v.astype(ml_dtypes.bfloat16) for k, v in sd.items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    ids = None
    if tokenizer_path:
        from .tokenizer import Tokenizer
        tok = Tokenizer(tokenizer_path)
        ids = {"bos": tok.bos_id, "eos": tok.eos_id, "pad": tok.pad_id}
        try:
            tok.export_hf(out_dir)
        except Exception as e:  # transformers/protobuf missing -> still ship the raw .model file
            print(f"[export] HF tokenizer export skipped ({e}); copying the SentencePiece model instead")
            import shutil
            shutil.copy(tokenizer_path, os.path.join(out_dir, "tokenizer.model"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config(model.cfg, ids), f, indent=2)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": (ids or {}).get("bos", 1), "eos_token_id": (ids or {}).get("eos", 2)}, f, indent=2)
    if model_card:
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write("---\n" + "\n".join(f"{k}: {v}" for k, v in model_card.items()) + "\n---\n")
    n_bytes = sum(v.nbytes for v in sd.values())
    print(f"[export] wrote {len(sd)} tensors ({n_bytes/1e9:.2f} GB, {dtype}) to {out_dir}")
    return out_dir


def verify(model: GW1BModel, out_dir: str, n_tokens: int = 32, seed: int = 0, atol: float = 2e-2) -> float:
    """Load the exported model with transformers (CPU) and compare logits with the JAX model."""
    import jax.numpy as jnp
    import torch
    from transformers import AutoModelForCausalLM

    try:
        hf = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32)
    except TypeError:  # transformers < 4.56
        hf = AutoModelForCausalLM.from_pretrained(out_dir, torch_dtype=torch.float32)
    hf.eval()
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, model.cfg.vocab_size, size=(2, n_tokens))
    with torch.no_grad():
        hf_logits = hf(torch.tensor(toks)).logits.float().numpy()
    jax_logits = np.asarray(model(jnp.asarray(toks)), dtype=np.float32)
    # compare log-probabilities (invariant to the per-position logit offset)
    def logp(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    diff = float(np.abs(logp(hf_logits) - logp(jax_logits)).max())
    argmax_agree = float((hf_logits.argmax(-1) == jax_logits.argmax(-1)).mean())
    print(f"[export] verify: max |Δ log-prob| = {diff:.4f}, argmax agreement = {argmax_agree:.1%}")
    if diff > atol:
        print("[export] WARNING: mismatch larger than tolerance (bf16 export rounds weights; use --dtype float32 to check)")
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser(description="export a GW1B checkpoint to Hugging Face Llama format")
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--tokenizer", default=None, help="SentencePiece .model (default: the run's tokenizer)")
    ap.add_argument("--verify", action="store_true", help="reload with transformers on CPU and compare logits")
    a = ap.parse_args(argv)
    from .evaluate import load_run
    model, cfg, step = load_run(a.run, a.step, shard_params=False)
    tokenizer = a.tokenizer or cfg.data.tokenizer
    card = {"license": "apache-2.0", "language": "en", "tags": "[gw1b, jax, flax, pretrained-from-scratch]",
            "base_model_step": step, "n_params": cfg.model.n_params}
    export(model, a.out, tokenizer if os.path.exists(tokenizer) else None, a.dtype, card)
    if a.verify:
        verify(model, a.out)


if __name__ == "__main__":
    main()
