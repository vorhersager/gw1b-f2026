"""Validation loss / perplexity of a checkpoint, and a helper to load a trained model for notebooks.

    python -m gw1b.evaluate --run $GW1B_SCRATCH/users/$USER/runs/50m            # latest checkpoint
    python -m gw1b.evaluate --run .../runs/50m --step 4000 --batches 50 --data $GW1B_SCRATCH/data/heldout
"""
from __future__ import annotations

import argparse
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import checkpoint as ckpt_lib
from .config import TrainConfig, load_config
from .data import BatchLoader, TokenDataset
from .model import GW1BModel, cross_entropy
from .sharding import create_model, make_mesh, put_batch


def load_run(run_dir: str, step: int | None = None, overrides: list[str] | None = None,
             shard_params: bool | None = None) -> tuple[GW1BModel, TrainConfig, int]:
    """Rebuild the model from <run_dir>/config.yaml and load parameters from its checkpoints."""
    cfg = load_config(os.path.join(run_dir, "config.yaml"), overrides)
    if shard_params is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, shard_params=shard_params))
    mesh = make_mesh()
    model = create_model(cfg, mesh)
    loaded = ckpt_lib.load_model_only(os.path.join(run_dir, "checkpoints"), model, step)
    return model, cfg, loaded


@nnx.jit
def _loss_step(model, batch):
    logits = model(batch["inputs"])
    return cross_entropy(logits, batch["targets"])


def eval_loss(model: GW1BModel, data_dir: str, seq_len: int, batch_size: int = 16, n_batches: int = 50) -> dict:
    ds = TokenDataset(data_dir, seq_len)
    mesh = make_mesh()
    loader = BatchLoader(ds, batch_size, shuffle=False, rank=jax.process_index(), world=jax.process_count(),
                         prefetch=2, num_workers=1)
    n = min(n_batches, max(1, ds.n_windows // batch_size))
    losses = []
    with jax.set_mesh(mesh):
        for _ in range(n):
            losses.append(_loss_step(model, put_batch(next(loader), mesh)))
    loader.close()
    loss = float(jnp.mean(jnp.stack(losses)))
    return {"loss": loss, "ppl": math.exp(loss), "tokens_evaluated": n * batch_size * seq_len, "batches": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description="evaluate a GW1B checkpoint")
    ap.add_argument("--run", required=True, help="run directory (contains config.yaml and checkpoints/)")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--data", default=None, help="token-shard directory (default: the run's val_dir)")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    model, cfg, step = load_run(a.run, a.step)
    data = a.data or cfg.data.val_dir
    res = eval_loss(model, data, cfg.data.seq_len, a.batch_size, a.batches)
    res.update({"run": a.run, "step": step, "data": data, "n_params": cfg.model.n_params})
    print(json.dumps(res, indent=2))
    with open(os.path.join(a.run, f"eval_step{step}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
