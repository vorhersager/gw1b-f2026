"""A GW1B checkpoint as plain numpy — the forward pass with every intermediate exposed, for the visualizer.

`NumpyModel` re-implements gw1b/model.py (RMSNorm/LayerNorm, RoPE, grouped-query attention with a KV cache,
SwiGLU/GELU/ReLU MLP, tied or separate output head) on the raw parameter arrays that `export.load_params` reads from
a checkpoint. It is deliberately simple (~120 lines) so students can read "the whole model" in one place; the test
suite checks its logits against the JAX model to 1e-3. CPU only, one token at a time after the prompt.

`trace_generate` runs a prompt and an autoregressive completion and records, for every generated token, what the
page animates: the residual stream after the embedding and after every block (as 64 cells + its norm), the attention
of the new token over all previous positions per layer (mean over heads), and the next-token distribution.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..config import ModelConfig


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _cells(v: np.ndarray, n: int = 64) -> list[float]:
    """A d-vector as n block means (signed), for the residual-stream strips."""
    d = v.shape[0]
    if d <= n:
        return np.round(v.astype(np.float64), 6).tolist()
    b = math.ceil(d / n)
    pad = np.pad(v.astype(np.float64), (0, b * math.ceil(d / b) - d))
    return np.round(pad.reshape(-1, b).mean(1), 6).tolist()


class NumpyModel:
    def __init__(self, params: dict[str, np.ndarray], cfg: ModelConfig):
        self.p, self.cfg = params, cfg
        self.E = params["embed.embedding"]                       # [V, d]
        self.head = params.get("lm_head.kernel")                 # [d, V] or None (tied embeddings)
        self.wpe = params.get("wpe.embedding")                   # [max_seq_len, d] for pos == "learned"
        self.n_layers = cfg.n_layers
        self.reset()

    # -- cache ------------------------------------------------------------------
    def reset(self) -> None:
        self.k_cache: list[np.ndarray | None] = [None] * self.n_layers   # per layer [S, K, hd]
        self.v_cache: list[np.ndarray | None] = [None] * self.n_layers
        self.n_cached = 0

    # -- pieces -----------------------------------------------------------------
    def norm(self, x: np.ndarray, prefix: str) -> np.ndarray:
        scale = self.p[prefix + ".scale"]
        eps = self.cfg.norm_eps
        if self.cfg.norm == "rmsnorm":
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * scale
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * scale + self.p.get(prefix + ".bias", 0.0)

    def rope_tables(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hd = self.cfg.head_dim
        if self.cfg.pos != "rope":
            return np.ones((len(positions), hd), np.float32), np.zeros((len(positions), hd), np.float32)
        inv = 1.0 / (self.cfg.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        f = positions.astype(np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([f, f], -1)
        return np.cos(emb), np.sin(emb)

    @staticmethod
    def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:  # x [T, N, hd]
        x1, x2 = np.split(x, 2, axis=-1)
        rot = np.concatenate([-x2, x1], -1)
        return x * cos[:, None, :] + rot * sin[:, None, :]

    def attention(self, h: np.ndarray, layer: int, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """h [T, d] (new positions) -> (attention output [T, d], weights of the LAST new position [N, S])."""
        c, pre = self.cfg, f"blocks.{layer}.attn."
        T = h.shape[0]
        N, K, hd = c.n_heads, c.n_kv_heads, c.head_dim
        q = (h @ self.p[pre + "q_proj.kernel"]).reshape(T, N, hd)
        k = (h @ self.p[pre + "k_proj.kernel"]).reshape(T, K, hd)
        v = (h @ self.p[pre + "v_proj.kernel"]).reshape(T, K, hd)
        q, k = self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)
        kc, vc = self.k_cache[layer], self.v_cache[layer]
        k_all = k if kc is None else np.concatenate([kc, k], 0)        # [S, K, hd]
        v_all = v if vc is None else np.concatenate([vc, v], 0)
        self.k_cache[layer], self.v_cache[layer] = k_all, v_all
        S, G = k_all.shape[0], N // K
        qg = q.reshape(T, K, G, hd)
        scores = np.einsum("tkgh,skh->tkgs", qg, k_all) / math.sqrt(hd)  # [T, K, G, S]
        start = S - T                                                    # absolute position of the first new token
        causal = np.arange(S)[None, :] <= (start + np.arange(T))[:, None]  # [T, S]
        scores = np.where(causal[:, None, None, :], scores, -1e30)
        probs = _softmax(scores, -1)
        out = np.einsum("tkgs,skh->tkgh", probs, v_all).reshape(T, N * hd)
        return out @ self.p[pre + "o_proj.kernel"], probs[-1].reshape(N, S)

    def mlp(self, h: np.ndarray, layer: int) -> np.ndarray:
        pre = f"blocks.{layer}.mlp."
        if self.cfg.activation == "swiglu":
            g = h @ self.p[pre + "gate_proj.kernel"]
            return (g / (1.0 + np.exp(-g)) * (h @ self.p[pre + "up_proj.kernel"])) @ self.p[pre + "down_proj.kernel"]
        u = h @ self.p[pre + "up_proj.kernel"]
        if self.cfg.activation == "gelu":
            u = 0.5 * u * (1.0 + np.tanh(0.7978845608 * (u + 0.044715 * u ** 3)))
        else:
            u = np.maximum(u, 0.0)
        return u @ self.p[pre + "down_proj.kernel"]

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ (self.E.T if self.head is None else self.head)

    # -- forward ----------------------------------------------------------------
    def forward(self, tokens: list[int] | np.ndarray, trace: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Process new tokens after what is cached. Returns (logits of the last position [V], trace of it)."""
        tokens = np.asarray(tokens, dtype=np.int64)
        T = len(tokens)
        positions = self.n_cached + np.arange(T)
        x = self.E[tokens].astype(np.float32)
        if self.cfg.pos == "learned" and self.wpe is not None:
            x = x + self.wpe[positions]
        cos, sin = self.rope_tables(positions)
        resid, attn = [x[-1].copy()], []
        for layer in range(self.n_layers):
            a, w = self.attention(self.norm(x, f"blocks.{layer}.input_norm"), layer, cos, sin)
            x = x + a
            x = x + self.mlp(self.norm(x, f"blocks.{layer}.post_attn_norm"), layer)
            if trace:
                resid.append(x[-1].copy())
                attn.append(w)
        self.n_cached += T
        x = self.norm(x, "final_norm")
        logits = self.logits(x[-1:])[0].astype(np.float64)
        info: dict[str, Any] = {}
        if trace:
            info = {"resid": resid, "attn": attn, "final": x[-1].copy()}
        return logits, info


# ---------------------------------------------------------------------------
# traced generation
# ---------------------------------------------------------------------------
def piece(tokenizer, i: int) -> str:
    """The displayable piece of one token id ('▁the' -> ' the'; decode() would drop the leading space)."""
    try:
        s = tokenizer.sp.id_to_piece(int(i))
    except Exception:  # noqa: BLE001
        return tokenizer.decode([int(i)])
    if s in ("<s>", "</s>", "<unk>", "<pad>"):
        return s
    return s.replace("\u2581", " ")


def trace_generate(model: NumpyModel, tokenizer, prompt: str, max_new: int = 32, temperature: float = 0.0,
                   top_k: int = 8, seed: int = 0, cells: int = 64) -> dict[str, Any]:
    """Prompt -> completion, with one trace per generated token (what the page animates)."""
    t0 = time.time()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.bos_id if tokenizer.bos_id >= 0 else 0]
    max_ctx = model.cfg.max_seq_len
    ids = ids[-(max_ctx - 1):]
    rng = np.random.default_rng(seed)
    model.reset()
    prompt_tokens = [{"id": int(i), "text": piece(tokenizer, i)} for i in ids]
    steps = []
    logits, info = model.forward(ids, trace=True)
    all_ids = list(ids)
    eos = tokenizer.eos_id
    reached_eos = False
    for n in range(max_new):
        probs = _softmax(logits)
        if temperature <= 0:
            choice = int(np.argmax(probs))
        else:
            p = _softmax(logits / temperature)
            choice = int(rng.choice(len(p), p=p / p.sum()))
        top = np.argsort(-probs)[:top_k]
        resid = info["resid"]
        norms = [float(np.linalg.norm(r)) for r in resid]
        steps.append({
            "position": len(all_ids),                               # the position this token is predicted from
            "token": {"id": choice, "text": piece(tokenizer, choice), "p": float(probs[choice])},
            "top": [{"id": int(i), "text": piece(tokenizer, i), "p": float(probs[i])} for i in top],
            "resid": [_cells(r, cells) for r in resid],            # L+1 strips: after embedding, after each block
            "resid_norm": norms,
            "attn": [np.round(w.mean(0), 5).tolist() for w in info["attn"]],   # per layer: last token over S positions
            "attn_max_head": [np.round(w.max(0), 5).tolist() for w in info["attn"]],
            "final": _cells(info["final"], cells),
        })
        all_ids.append(choice)
        if choice == eos:
            reached_eos = True
            break
        if len(all_ids) >= max_ctx:
            break
        logits, info = model.forward([choice], trace=True)
    text = tokenizer.decode(all_ids[len(ids):])
    return {"prompt": prompt, "prompt_tokens": prompt_tokens, "steps": steps, "completion": text,
            "reached_eos": reached_eos, "temperature": temperature, "seed": seed, "cells": cells,
            "seconds": round(time.time() - t0, 2)}
