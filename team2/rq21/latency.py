"""The latency stage: time KV-cached decoding on one A100, as shipped and compiled-only."""
from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gw1b.budget import gpu_key
from gw1b.evaluate import load_run
from gw1b.generate import _decode_step, _prefill, _sample
from gw1b.model import GW1BModel
from rq21.runs import save_json, select_runs

CACHE_LENGTHS = (512, 2048)
N_REPEATS = 5


def make_compiled_step(model: GW1BModel) -> Callable[..., Any]:
    """Return the decode computation of gw1b.generate._decode_step through a plain jax.jit over nnx.split state.

    Call the result as step(tokens, cache, pos, valid, key, temperature). The gap between this and the as-shipped
    step is software overhead that grows with depth, not model cost.
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def step(state, tokens, cache, pos, valid, key, temperature):
        logits, cache = nnx.merge(graphdef, state)(tokens, cache=cache, pos=pos, valid=valid)
        return _sample(logits[:, -1], key, temperature, None), cache

    # state is an argument, not captured inside step, so the weights are not compiled in as constants
    return functools.partial(step, state)


def time_decode(step_fn: Callable[..., Any], model: GW1BModel, cache_len: int, prompt_len: int = 64,
                n_warmup: int = 32, n_timed: int = 256) -> np.ndarray:
    """Return per-step decode times in ms at batch size 1, blocking on each step.

    step_fn is called as step_fn(tokens, cache, pos, valid, key, temperature). Prefills a seeded prompt without
    timing it, then discards compile and warm-up steps. Decoding is greedy and EOS is ignored, so the token count is
    fixed. Raises if prompt_len + n_warmup + n_timed exceeds cache_len.
    """
    n_steps = n_warmup + n_timed
    if prompt_len + n_steps > cache_len:
        raise ValueError(f"{prompt_len} prompt + {n_steps} decode positions do not fit in cache length {cache_len}")

    prompt = np.random.default_rng(0).integers(model.cfg.vocab_size, size=(1, prompt_len), dtype=np.int32)
    valid = jnp.ones((1, cache_len), bool)
    key = jax.random.key(0)  # greedy decoding never uses it, but the step takes one
    temperature = jnp.float32(0.0)
    logits, cache = _prefill(model, jnp.asarray(prompt), model.init_cache(1, cache_len), valid)
    next_token = _sample(logits, key, temperature, None)

    times = []
    for i in range(n_steps):
        tokens, pos = next_token[:, None], jnp.int32(prompt_len + i)  # prepared here so only the step is timed
        start = time.perf_counter()
        next_token, cache = step_fn(tokens, cache, pos, valid, key, temperature)
        jax.block_until_ready((next_token, cache))  # JAX returns before the GPU finishes, so wait for the work
        times.append(time.perf_counter() - start)
    return np.array(times[n_warmup:]) * 1e3


def cmd_latency(args: argparse.Namespace) -> int:
    """Time every selected run as shipped and compiled-only, at each cache length, N_REPEATS times.

    Measures gw1b.generate._decode_step (as shipped) and make_compiled_step (compiled-only). Each repeat visits all
    measurements in a fresh random order seeded by the Slurm job id, so a slow period on the GPU cannot line up with
    one shape. Writes <out>/<run name>/latency_<job id>.json, so repeat Slurm jobs add results instead of overwriting
    them. Raises on anything but exactly one A100, or outside a Slurm job.
    """
    kind = jax.devices()[0].device_kind
    if jax.device_count() != 1 or gpu_key(kind) != "a100":
        raise RuntimeError(f"latency needs exactly one A100, found {jax.device_count()} x {kind}")
    job_id = os.environ["SLURM_JOB_ID"]

    runs = select_runs(args)
    if not runs:
        return 1

    paths, checkpoint_steps, measurements, timings = {}, {}, [], {}
    for name, run_dir in runs.items():
        path = Path(args.out) / name / f"latency_{job_id}.json"
        if path.exists() and not args.force:
            print(f"{name}: skipped, {path} exists")
            continue
        model, _, step = load_run(run_dir, shard_params=False)
        paths[name], checkpoint_steps[name] = path, step
        versions = {"as_shipped": functools.partial(_decode_step, model), "compiled_only": make_compiled_step(model)}
        timings[name] = {version: {s: [] for s in CACHE_LENGTHS} for version in versions}
        measurements += [(name, version, step_fn, model, s) for version, step_fn in versions.items() for s in CACHE_LENGTHS]

    rng = np.random.default_rng(int(job_id))
    for _ in range(N_REPEATS):
        for i in rng.permutation(len(measurements)):
            name, version, step_fn, model, cache_len = measurements[i]
            timings[name][version][cache_len].append(time_decode(step_fn, model, cache_len).tolist())

    for name, path in paths.items():
        save_json(str(path), {"run": name, "job_id": job_id, "timings_ms": timings[name]}, step=checkpoint_steps[name])
        medians = {v: {s: round(float(np.median(t)), 3) for s, t in by_len.items()} for v, by_len in timings[name].items()}
        print(f"{name}: median ms/step {medians}")
    return 0
