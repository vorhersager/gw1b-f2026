"""Text generation with a KV cache (greedy, temperature, top-k), batched with left padding.

    python -m gw1b.generate --run $GW1B_SCRATCH/users/$USER/runs/50m --prompt "The capital of France is" --max-new-tokens 40
"""
from __future__ import annotations

import argparse
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .model import GW1BModel
from .tokenizer import Tokenizer


def _pad_left(seqs: list[list[int]], pad_id: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    B = len(seqs)
    toks = np.full((B, length), pad_id, dtype=np.int32)
    valid = np.zeros((B, length), dtype=bool)
    for i, s in enumerate(seqs):
        s = s[-length:]
        toks[i, length - len(s):] = s
        valid[i, length - len(s):] = True
    return toks, valid


@nnx.jit
def _prefill(model, tokens, cache, valid):
    logits, cache = model(tokens, cache=cache, pos=0, valid=valid)
    return logits[:, -1], cache


@functools.partial(nnx.jit, static_argnames=("top_k",))
def _decode_step(model, tokens, cache, pos, valid, key, temperature, top_k=None):
    logits, cache = model(tokens, cache=cache, pos=pos, valid=valid)
    next_tok = _sample(logits[:, -1], key, temperature, top_k)
    return next_tok, cache


def _sample(logits, key, temperature, top_k):
    if top_k:
        vals, idx = jax.lax.top_k(logits, top_k)
        masked = jnp.full_like(logits, -jnp.inf).at[jnp.arange(logits.shape[0])[:, None], idx].set(vals)
        logits = masked
    greedy = jnp.argmax(logits, axis=-1)
    sampled = jax.random.categorical(key, logits / jnp.maximum(temperature, 1e-6), axis=-1)
    return jnp.where(temperature <= 0.0, greedy, sampled).astype(jnp.int32)


def generate(model: GW1BModel, tokenizer: Tokenizer, prompts: str | list[str], max_new_tokens: int = 64,
             temperature: float = 0.0, top_k: int | None = None, seed: int = 0, stop: list[str] | None = None,
             add_bos: bool = False, echo: bool = False) -> list[str]:
    """Generate continuations for one or more prompts. temperature=0 -> greedy."""
    single = isinstance(prompts, str)
    prompts = [prompts] if single else list(prompts)
    enc = [tokenizer.encode(p, add_bos=add_bos) for p in prompts]
    max_ctx = model.cfg.max_seq_len
    Lp = min(max(len(e) for e in enc), max_ctx - 1)
    Lp = int(math.ceil(Lp / 64) * 64)  # pad prompts to a multiple of 64 -> few recompiles
    Lp = min(Lp, max_ctx - 1)
    total = min(Lp + max_new_tokens, max_ctx)
    pad_id = tokenizer.pad_id if tokenizer.pad_id >= 0 else 0
    toks, valid_prompt = _pad_left(enc, pad_id, Lp)
    B = len(prompts)
    valid = np.zeros((B, total), dtype=bool)
    valid[:, :Lp] = valid_prompt
    valid[:, Lp:] = True
    valid = jnp.asarray(valid)
    cache = model.init_cache(B, total)
    key = jax.random.key(seed)

    logits, cache = _prefill(model, jnp.asarray(toks), cache, valid)
    key, sub = jax.random.split(key)
    next_tok = _sample(logits, sub, jnp.float32(temperature), top_k)
    out = [[int(next_tok[i])] for i in range(B)]
    done = [False] * B
    eos = tokenizer.eos_id
    stop = stop or []
    for t in range(1, total - Lp):
        for i in range(B):
            if not done[i]:
                if eos >= 0 and out[i][-1] == eos:
                    done[i] = True
                elif stop and any(s in tokenizer.decode(out[i]) for s in stop):
                    done[i] = True
        if all(done):
            break
        key, sub = jax.random.split(key)
        next_tok, cache = _decode_step(model, next_tok[:, None], cache, jnp.int32(Lp + t - 1), valid, sub,
                                       jnp.float32(temperature), top_k)
        for i in range(B):
            if not done[i]:
                out[i].append(int(next_tok[i]))

    texts = []
    for i in range(B):
        ids = out[i]
        if eos >= 0 and ids and ids[-1] == eos:
            ids = ids[:-1]
        text = tokenizer.decode(ids)
        for s in stop:  # truncate at the first stop sequence
            if s in text:
                text = text[: text.index(s)]
        texts.append((prompts[i] + text) if echo else text)
    return texts


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate text from a GW1B checkpoint")
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--prompt", action="append", required=True, help="prompt (repeatable)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from .evaluate import load_run
    model, cfg, step = load_run(a.run, a.step)
    tok = Tokenizer(cfg.data.tokenizer)
    for p, t in zip(a.prompt, generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k, a.seed)):
        print(f"--- {p!r}\n{t}\n")


if __name__ == "__main__":
    main()
