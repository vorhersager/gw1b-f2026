"""GW1B model: a Llama-style decoder-only Transformer written in Flax NNX.

Design (from the GW1B design document):
    * pre-norm blocks (RMSNorm or LayerNorm)
    * rotary position embeddings (RoPE) or learned positions
    * grouped-query attention (n_kv_heads < n_heads), multi-head when n_kv_heads == n_heads
    * SwiGLU (or GELU / ReLU) feed-forward
    * tied input/output embeddings (optional)
    * bf16 compute with f32 master weights

Conventions follow Hugging Face's `LlamaForCausalLM` (RoPE "rotate_half", separate gate/up/down
projections) so a trained checkpoint exports to HF format without any weight permutation
(see gw1b/export_hf.py).

Every Linear/Embed parameter is annotated with a sharding spec so the model can be created
directly sharded across GPUs (FSDP) — see gw1b/sharding.py.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jax.sharding import PartitionSpec as P

from .config import ModelConfig

DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}
MESH_AXIS = "data"  # the single mesh axis used for data parallelism + parameter sharding (FSDP)


# ---------------------------------------------------------------------------
# sharding helpers
# ---------------------------------------------------------------------------
def shard_spec(shape: tuple[int, ...], n_shards: int) -> tuple[str | None, ...]:
    """FSDP rule: shard the largest dimension that divides evenly by the number of shards.

    Dimensions that do not divide evenly are replicated. With n_shards == 1 everything is replicated.
    """
    spec: list[str | None] = [None] * len(shape)
    if n_shards > 1:
        for i in sorted(range(len(shape)), key=lambda i: -shape[i]):
            if shape[i] % n_shards == 0:
                spec[i] = MESH_AXIS
                break
    return tuple(spec)


def shard_activations(x: jax.Array) -> jax.Array:
    """Keep activations [B, T, ...] sharded along the batch axis when a mesh is active (no-op otherwise)."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or mesh.empty or mesh.size == 1:
        return x
    return jax.lax.with_sharding_constraint(x, P(MESH_AXIS, *([None] * (x.ndim - 1))))


def _normal(std: float, shape: tuple[int, ...], n_shards: int):
    """A normal(0, std) initializer; carries FSDP sharding metadata when the model is sharded.

    With n_shards <= 1 (single GPU / CPU notebooks) no metadata is attached, so the model can be
    created without a mesh context. With n_shards > 1 the model must be created inside
    `with jax.set_mesh(mesh):` (Flax >= 0.12 shards variables eagerly at creation).
    """
    init = jax.nn.initializers.normal(std)
    if n_shards <= 1:
        return init
    return nnx.with_partitioning(init, shard_spec(shape, n_shards))


def _ones(shape: tuple[int, ...], n_shards: int = 1):
    init = jax.nn.initializers.ones
    return init if n_shards <= 1 else nnx.with_partitioning(init, (None,) * len(shape))


def _zeros(shape: tuple[int, ...], n_shards: int = 1):
    init = jax.nn.initializers.zeros
    return init if n_shards <= 1 else nnx.with_partitioning(init, (None,) * len(shape))


# ---------------------------------------------------------------------------
# rotary position embeddings (HF Llama convention)
# ---------------------------------------------------------------------------
def rope_tables(positions: jax.Array, head_dim: int, theta: float, dtype: Any) -> tuple[jax.Array, jax.Array]:
    """cos/sin tables of shape [T, head_dim] for the given absolute positions [T]."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # [T, head_dim/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)                        # [T, head_dim]
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """x: [B, T, N, H]; cos/sin: [T, H]."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (x * cos + _rotate_half(x) * sin).astype(x.dtype)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------
def make_norm(cfg: ModelConfig, rngs: nnx.Rngs, n_shards: int = 1) -> nnx.Module:
    dtype, pdtype = DTYPES[cfg.dtype], DTYPES[cfg.param_dtype]
    if cfg.norm == "rmsnorm":
        return nnx.RMSNorm(cfg.d_model, epsilon=cfg.norm_eps, dtype=dtype, param_dtype=pdtype,
                           scale_init=_ones((cfg.d_model,), n_shards), rngs=rngs)
    return nnx.LayerNorm(cfg.d_model, epsilon=cfg.norm_eps, dtype=dtype, param_dtype=pdtype,
                         scale_init=_ones((cfg.d_model,), n_shards), bias_init=_zeros((cfg.d_model,), n_shards),
                         rngs=rngs)


class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, n_shards: int = 1):
        d, H, K, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        dtype, pdtype = DTYPES[cfg.dtype], DTYPES[cfg.param_dtype]
        out_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)  # GPT-2 style scaled init for residual projections
        common = dict(use_bias=False, dtype=dtype, param_dtype=pdtype, rngs=rngs)
        self.n_heads, self.n_kv_heads, self.head_dim = H, K, hd
        self.q_proj = nnx.Linear(d, H * hd, kernel_init=_normal(cfg.init_std, (d, H * hd), n_shards), **common)
        self.k_proj = nnx.Linear(d, K * hd, kernel_init=_normal(cfg.init_std, (d, K * hd), n_shards), **common)
        self.v_proj = nnx.Linear(d, K * hd, kernel_init=_normal(cfg.init_std, (d, K * hd), n_shards), **common)
        self.o_proj = nnx.Linear(H * hd, d, kernel_init=_normal(out_std, (H * hd, d), n_shards), **common)

    def __call__(self, x, cos, sin, *, cache=None, pos=None, valid=None, impl="xla"):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=impl)
            new_cache = None
        else:
            # incremental decoding: write the new K/V at [pos, pos+T) and attend to everything <= own position
            k_all = jax.lax.dynamic_update_slice_in_dim(cache["k"], k.astype(cache["k"].dtype), pos, axis=1)
            v_all = jax.lax.dynamic_update_slice_in_dim(cache["v"], v.astype(cache["v"].dtype), pos, axis=1)
            S = k_all.shape[1]
            q_pos = pos + jnp.arange(T)[:, None]
            k_pos = jnp.arange(S)[None, :]
            mask = (k_pos <= q_pos)[None, None, :, :]  # [1, 1, T, S]
            if valid is not None:  # [B, S] False for (left-)padding positions
                mask = mask & valid[:, None, None, :]
            out = jax.nn.dot_product_attention(q, k_all, v_all, mask=mask, implementation="xla")
            new_cache = {"k": k_all, "v": v_all}
        return self.o_proj(out.reshape(B, T, self.n_heads * self.head_dim)), new_cache


class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, n_shards: int = 1):
        d, ff = cfg.d_model, cfg.d_ff
        dtype, pdtype = DTYPES[cfg.dtype], DTYPES[cfg.param_dtype]
        out_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        common = dict(use_bias=False, dtype=dtype, param_dtype=pdtype, rngs=rngs)
        self.activation = cfg.activation
        if cfg.activation == "swiglu":
            self.gate_proj = nnx.Linear(d, ff, kernel_init=_normal(cfg.init_std, (d, ff), n_shards), **common)
        self.up_proj = nnx.Linear(d, ff, kernel_init=_normal(cfg.init_std, (d, ff), n_shards), **common)
        self.down_proj = nnx.Linear(ff, d, kernel_init=_normal(out_std, (ff, d), n_shards), **common)

    def __call__(self, x):
        if self.activation == "swiglu":
            return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
        h = self.up_proj(x)
        h = jax.nn.gelu(h, approximate=True) if self.activation == "gelu" else jax.nn.relu(h)
        return self.down_proj(h)


class Block(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, n_shards: int = 1):
        self.input_norm = make_norm(cfg, rngs, n_shards)
        self.attn = Attention(cfg, rngs=rngs, n_shards=n_shards)
        self.post_attn_norm = make_norm(cfg, rngs, n_shards)
        self.mlp = MLP(cfg, rngs=rngs, n_shards=n_shards)

    def __call__(self, x, cos, sin, *, cache=None, pos=None, valid=None, impl="xla"):
        h, new_cache = self.attn(self.input_norm(x), cos, sin, cache=cache, pos=pos, valid=valid, impl=impl)
        x = shard_activations(x + h)
        x = shard_activations(x + self.mlp(self.post_attn_norm(x)))
        return x, new_cache


class GW1BModel(nnx.Module):
    """Decoder-only LM. `model(tokens)` -> logits [B, T, vocab] in float32."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, n_shards: int = 1, attn_impl: str = "xla"):
        self.cfg = cfg
        self.attn_impl = attn_impl
        dtype, pdtype = DTYPES[cfg.dtype], DTYPES[cfg.param_dtype]
        self.embed = nnx.Embed(cfg.vocab_size, cfg.d_model, dtype=dtype, param_dtype=pdtype,
                               embedding_init=_normal(cfg.init_std, (cfg.vocab_size, cfg.d_model), n_shards),
                               rngs=rngs)
        if cfg.pos == "learned":
            self.wpe = nnx.Embed(cfg.max_seq_len, cfg.d_model, dtype=dtype, param_dtype=pdtype,
                                 embedding_init=_normal(cfg.init_std, (cfg.max_seq_len, cfg.d_model), n_shards),
                                 rngs=rngs)
        self.blocks = nnx.List([Block(cfg, rngs=rngs, n_shards=n_shards) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg, rngs, n_shards)
        if not cfg.tie_embeddings:
            self.lm_head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, dtype=dtype, param_dtype=pdtype,
                                      kernel_init=_normal(cfg.init_std, (cfg.d_model, cfg.vocab_size), n_shards),
                                      rngs=rngs)

    # -- forward --------------------------------------------------------------
    def __call__(self, tokens: jax.Array, *, cache: list | None = None, pos: int | jax.Array | None = None,
                 valid: jax.Array | None = None):
        """Training/eval: model(tokens) -> logits.  Decoding: model(tokens, cache=, pos=, valid=) -> (logits, cache).

        valid: optional [B, cache_len] bool mask marking real (non-padding) cache positions (left padding).
        """
        cfg = self.cfg
        B, T = tokens.shape
        compute_dtype = DTYPES[cfg.dtype]
        positions = jnp.arange(T) if pos is None else pos + jnp.arange(T)
        x = shard_activations(self.embed(tokens))
        if cfg.pos == "learned":
            x = x + self.wpe(positions)
        if cfg.pos == "rope":
            cos, sin = rope_tables(positions, cfg.head_dim, cfg.rope_theta, compute_dtype)
        else:  # no rotation: cos=1, sin=0
            cos = jnp.ones((T, cfg.head_dim), compute_dtype)
            sin = jnp.zeros((T, cfg.head_dim), compute_dtype)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, cos, sin, cache=None if cache is None else cache[i], pos=pos, valid=valid,
                         impl=self.attn_impl)
            new_caches.append(c)
        x = self.final_norm(x)
        logits = shard_activations(self.logits(x))
        return logits if cache is None else (logits, new_caches)

    def logits(self, x: jax.Array) -> jax.Array:
        """Project hidden states to vocabulary logits in float32 (numerically important for the loss)."""
        compute_dtype = DTYPES[self.cfg.dtype]
        if self.cfg.tie_embeddings:
            w = self.embed.embedding[...].astype(compute_dtype)  # [V, d]
            return jnp.einsum("btd,vd->btv", x, w, preferred_element_type=jnp.float32)
        w = self.lm_head.kernel[...].astype(compute_dtype)  # [d, V]
        return jnp.einsum("btd,dv->btv", x, w, preferred_element_type=jnp.float32)

    def init_cache(self, batch_size: int, max_len: int, dtype: Any = None) -> list[dict[str, jax.Array]]:
        cfg = self.cfg
        dtype = dtype or DTYPES[cfg.dtype]
        shape = (batch_size, max_len, cfg.n_kv_heads, cfg.head_dim)
        return [{"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype)} for _ in range(cfg.n_layers)]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def resolve_attn_impl(cfg: ModelConfig) -> str:
    """Pick cuDNN flash attention on Ampere+ GPUs with bf16/fp16, XLA otherwise (V100, CPU, f32)."""
    if cfg.attn_implementation != "auto":
        return cfg.attn_implementation
    dev = jax.devices()[0]
    if dev.platform != "gpu" or cfg.dtype == "float32" or cfg.head_dim % 8 or cfg.head_dim > 128:
        return "xla"
    kind = dev.device_kind.lower()
    ampere_or_newer = any(k in kind for k in ("a100", "a10", "a30", "a40", "l40", "l4", "h100", "h200", "gh200",
                                               "b200", "rtx 30", "rtx 40", "rtx 50", "rtx a", "rtx 6000"))
    return "cudnn" if ampere_or_newer else "xla"


def cross_entropy(logits: jax.Array, targets: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """Mean next-token cross-entropy. logits [B,T,V] f32, targets [B,T] int, mask [B,T] (1=count)."""
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    if mask is None:
        return nll.mean()
    mask = mask.astype(nll.dtype)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def count_params(model: nnx.Module) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
