"""Orbax checkpointing of the NNX model + optimizer + training metadata, with auto-resume.

Layout:  <run_dir>/checkpoints/<step>/{model,opt,meta}
"""
from __future__ import annotations

import json
import os
from typing import Any

import jax
import orbax.checkpoint as ocp
from flax import nnx


def make_manager(ckpt_dir: str, save_every: int, keep: int = 3, keep_every: int | None = None,
                 async_save: bool = True) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=max(1, save_every),
        max_to_keep=keep,
        keep_period=keep_every,          # every keep_every-th step is never deleted (Team 5 checkpoint ladder)
        create=True,
        enable_async_checkpointing=async_save,
    )
    return ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=options)


def save(mgr: ocp.CheckpointManager, step: int, model: nnx.Module, optimizer: nnx.Optimizer | None,
         meta: dict[str, Any], force: bool = False) -> bool:
    items = {"model": ocp.args.StandardSave(nnx.state(model, nnx.Param)), "meta": ocp.args.JsonSave(meta)}
    if optimizer is not None:
        items["opt"] = ocp.args.StandardSave(nnx.state(optimizer))
    return mgr.save(step, args=ocp.args.Composite(**items), force=force)


def latest_step(ckpt_dir: str) -> int | None:
    if not os.path.isdir(ckpt_dir):
        return None
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        return mgr.latest_step()
    finally:
        mgr.close()


def restore(mgr: ocp.CheckpointManager, model: nnx.Module, optimizer: nnx.Optimizer | None = None,
            step: int | None = None) -> tuple[int, dict[str, Any]] | None:
    """Restore into existing (sharded) model/optimizer objects. Returns (step, meta) or None."""
    step = mgr.latest_step() if step is None else step
    if step is None:
        return None
    model_target = jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model, nnx.Param))
    items = {"model": ocp.args.StandardRestore(model_target), "meta": ocp.args.JsonRestore()}
    if optimizer is not None:
        opt_target = jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer))
        items["opt"] = ocp.args.StandardRestore(opt_target)
    restored = mgr.restore(step, args=ocp.args.Composite(**items))
    nnx.update(model, restored["model"])
    if optimizer is not None:
        nnx.update(optimizer, restored["opt"])
    return step, dict(restored["meta"])


def load_model_only(ckpt_dir: str, model: nnx.Module, step: int | None = None) -> int:
    """Load just the parameters (evaluation, generation, export). Returns the step loaded."""
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        step = mgr.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        target = jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model, nnx.Param))
        restored = mgr.restore(step, args=ocp.args.Composite(model=ocp.args.StandardRestore(target)))
        nnx.update(model, restored["model"])
        return step
    finally:
        mgr.close()


def read_meta(ckpt_dir: str, step: int | None = None) -> dict[str, Any]:
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        step = mgr.latest_step() if step is None else step
        restored = mgr.restore(step, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
        return dict(restored["meta"])
    finally:
        mgr.close()


def write_run_summary(run_dir: str, summary: dict[str, Any]) -> str:
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path
