"""The loss stage: score each final checkpoint on one fixed set of validation windows."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import numpy as np

from gw1b.data import TokenDataset
from gw1b.evaluate import _loss_step, load_run
from gw1b.model import GW1BModel
from gw1b.sharding import make_mesh, put_batch
from rq21.runs import save_json, select_runs


def select_windows(n_windows: int, max_windows: int = 8192, batch_size: int = 8) -> np.ndarray:
    """Return the validation window indices every run is scored on.

    All windows if there are at most max_windows, otherwise max_windows evenly spaced ones, trimmed to a whole
    number of batches. The result depends only on the arguments, so it is identical across runs. Raises ValueError
    if there are too few windows for one batch.
    """
    n = min(n_windows, max_windows)
    n -= n % batch_size  # every batch must be full, so all batch scores cover the same amount of text
    if n == 0:
        raise ValueError(f"{n_windows} validation windows is too few for one batch of {batch_size}")
    return np.arange(n) * n_windows // n


def score_windows(model: GW1BModel, dataset: TokenDataset, indices: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """Return the mean next-token loss of each fixed-size batch of windows, in nats/token.

    Per-batch values are kept because the analysis bootstraps over batches. Raises on a non-finite loss.
    """
    mesh = make_mesh()
    losses = []
    with jax.set_mesh(mesh):
        for start in range(0, len(indices), batch_size):
            batch = dataset.batch(indices[start:start + batch_size])
            losses.append(float(_loss_step(model, put_batch(batch, mesh))))
    scores = np.array(losses)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite loss in {np.count_nonzero(~np.isfinite(scores))} of {len(scores)} batches")
    return scores


def cmd_loss(args: argparse.Namespace) -> int:
    """Score each selected run's final checkpoint and write <out>/<run name>/loss.json.

    Saves batch losses, mean and perplexity. Runs with an existing result are skipped unless --force is given.
    Returns 1 if no runs match, or if a name given with --run does not exist.
    """
    runs = select_runs(args)
    if not runs:
        return 1

    for name, run_dir in runs.items():
        path = Path(args.out) / name / "loss.json"
        if path.exists() and not args.force:
            print(f"{name}: skipped, {path} exists")
            continue
        model, cfg, step = load_run(run_dir, shard_params=False)
        dataset = TokenDataset(cfg.data.val_dir, cfg.data.seq_len)
        indices = select_windows(dataset.n_windows)
        scores = score_windows(model, dataset, indices)
        loss = float(scores.mean())
        save_json(str(path), {"run": name, "val_dir": cfg.data.val_dir, "val_windows": dataset.n_windows,
                              "windows_scored": len(indices), "loss": loss, "perplexity": math.exp(loss),
                              "batch_losses": scores.tolist()}, step=step)
        print(f"{name}: loss {loss:.4f} at step {step}, on {len(indices)} windows")
    return 0
