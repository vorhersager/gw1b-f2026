"""The GW1B trainer.

    python -m gw1b.train --config configs/50m.yaml --set run.name=team2-gqa-ablation
    python -m gw1b.train --config configs/gw1b_1p15b.yaml            # resumes automatically if checkpoints exist

What it does
    * builds the model sharded across all visible GPUs (FSDP) — multi-node when launched with srun
    * AdamW (or Lion/SGD) with warmup + cosine/WSD/linear/constant schedule, gradient clipping,
      gradient accumulation, weight decay on matrices only
    * logs loss / lr / grad-norm / tokens-per-second / MFU / GPU-hours / $-equivalent to
      metrics.jsonl + TensorBoard (+ Weights & Biases if enabled)
    * evaluates on the validation shards, checkpoints with Orbax and resumes where it stopped
    * writes summary.json at the end: performance, tokens, FLOPs, GPU-hours, USD — the numbers every
      GW1B experiment has to report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from . import checkpoint as ckpt_lib
from .budget import flops_per_token, peak_flops
from .config import OptimConfig, TrainConfig, config_summary, load_config, save_config
from .data import BatchLoader, TokenDataset
from .model import count_params, cross_entropy
from .sharding import create_model, describe_devices, init_distributed, make_mesh, put_batch


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------
def lr_schedule(o: OptimConfig, total_steps: int) -> optax.Schedule:
    peak, end = o.lr, o.lr * o.min_lr_ratio
    warmup = max(0, min(o.warmup_steps, total_steps - 1))
    if o.schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(0.0, peak, warmup, total_steps, end_value=end)
    if o.schedule == "linear":
        return optax.join_schedules([optax.linear_schedule(0.0, peak, warmup),
                                     optax.linear_schedule(peak, end, max(1, total_steps - warmup))], [warmup])
    if o.schedule == "constant":
        return optax.join_schedules([optax.linear_schedule(0.0, peak, warmup), optax.constant_schedule(peak)], [warmup])
    if o.schedule == "wsd":  # warmup - stable - decay
        decay = max(1, int(total_steps * o.decay_fraction))
        stable = max(0, total_steps - warmup - decay)
        return optax.join_schedules([optax.linear_schedule(0.0, peak, warmup), optax.constant_schedule(peak),
                                     optax.linear_schedule(peak, end, decay)], [warmup, warmup + stable])
    raise ValueError(o.schedule)


def build_optimizer(o: OptimConfig, total_steps: int) -> tuple[optax.GradientTransformation, optax.Schedule]:
    sched = lr_schedule(o, total_steps)

    def decay_mask(params):  # weight decay on matrices/embeddings only, not on norm scales / biases
        return jax.tree.map(lambda p: p.ndim >= 2, params)

    mask = None if o.decay_norm_and_bias else decay_mask
    if o.name == "adamw":
        core = optax.adamw(sched, b1=o.beta1, b2=o.beta2, eps=o.eps, weight_decay=o.weight_decay, mask=mask)
    elif o.name == "lion":
        core = optax.lion(sched, b1=o.beta1, b2=o.beta2, weight_decay=o.weight_decay, mask=mask)
    elif o.name == "sgd":
        core = optax.sgd(sched, momentum=o.beta1, nesterov=True)
    else:
        raise ValueError(f"unknown optimizer {o.name!r} (add it in gw1b/train.py:build_optimizer)")
    parts = [optax.clip_by_global_norm(o.grad_clip)] if o.grad_clip and o.grad_clip > 0 else []
    tx = optax.chain(*parts, core)
    if o.grad_accum > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=o.grad_accum)
    return tx, sched


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["inputs"])
        return cross_entropy(logits, batch["targets"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return loss, grad_norm


@nnx.jit
def eval_step(model, batch):
    return cross_entropy(model(batch["inputs"]), batch["targets"])


def evaluate(model, dataset: TokenDataset, cfg: TrainConfig, mesh, n_batches: int) -> float:
    micro = cfg.run.batch_size // cfg.optim.grad_accum
    loader = BatchLoader(dataset, micro, seed=0, shuffle=False, rank=jax.process_index(), world=jax.process_count(),
                         prefetch=2, num_workers=1)
    n = min(n_batches, max(1, dataset.n_windows // micro))
    losses = [eval_step(model, put_batch(next(loader), mesh)) for _ in range(n)]
    loader.close()
    return float(jnp.mean(jnp.stack(losses)))


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class MetricsLogger:
    def __init__(self, run_dir: str, cfg: TrainConfig, enabled: bool):
        self.enabled = enabled
        self.tb = self.wandb = None
        if not enabled:
            return
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        if cfg.run.tensorboard:
            try:
                from tensorboardX import SummaryWriter
                self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
            except ImportError:
                print("[train] tensorboardX not installed; skipping TensorBoard logs")
        if cfg.run.wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=os.environ.get("WANDB_PROJECT", "gw1b"), name=cfg.run.name, dir=run_dir,
                           config=cfg.to_dict(), resume="allow", id=cfg.run.name.replace("/", "-"))
            except Exception as e:  # never let logging kill a training run
                print(f"[train] wandb disabled: {e}")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.f.write(json.dumps({"step": step, **metrics}) + "\n")
        self.f.flush()
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self.f.close()
        if self.tb:
            self.tb.close()
        if self.wandb:
            self.wandb.finish()


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig, resume: bool = True) -> dict[str, Any]:
    distributed = init_distributed()
    is_main = jax.process_index() == 0
    mesh = make_mesh()
    n_devices = jax.device_count()
    total_steps = cfg.resolved_total_steps()
    run_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        save_config(cfg, os.path.join(run_dir, "config.yaml"))
        print(f"[train] run dir: {run_dir}")
        print(f"[train] {describe_devices()}" + (" (multi-process)" if distributed else ""))
        print("[train] " + config_summary(cfg).replace("\n", "\n[train] "))

    # ---- model / optimizer -------------------------------------------------
    t0 = time.time()
    model = create_model(cfg, mesh)
    tx, sched = build_optimizer(cfg.optim, total_steps)
    with jax.set_mesh(mesh):
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    n_params = count_params(model)
    if is_main:
        print(f"[train] model created in {time.time() - t0:.1f}s: {n_params/1e6:,.1f}M params, "
              f"attention={model.attn_impl}, params sharded={cfg.run.shard_params}")

    # ---- checkpoints / resume ----------------------------------------------
    mgr = ckpt_lib.make_manager(ckpt_dir, cfg.run.ckpt_every, cfg.run.ckpt_keep, cfg.run.ckpt_keep_every)
    step, tokens_seen, prev_seconds = 0, 0, 0.0
    if resume:
        restored = ckpt_lib.restore(mgr, model, optimizer)
        if restored:
            step, meta = restored
            tokens_seen = int(meta.get("tokens_seen", step * cfg.tokens_per_step))
            prev_seconds = float(meta.get("train_seconds", 0.0))
            if is_main:
                print(f"[train] resumed from step {step} ({tokens_seen/1e9:.3f}B tokens, "
                      f"{_fmt_time(prev_seconds)} of training so far)")
    if step >= total_steps:
        if is_main:
            print(f"[train] run already finished ({step}/{total_steps} steps)")
        return {"step": step, "finished": True}

    # ---- data --------------------------------------------------------------
    train_ds = TokenDataset(cfg.data.train_dir, cfg.data.seq_len)
    val_ds = None
    if cfg.data.val_dir and os.path.isdir(cfg.data.val_dir):
        try:
            val_ds = TokenDataset(cfg.data.val_dir, cfg.data.seq_len)
        except (FileNotFoundError, ValueError) as e:
            if is_main:
                print(f"[train] no validation data ({e}); skipping eval")
    micro = cfg.run.batch_size // cfg.optim.grad_accum
    loader = BatchLoader(train_ds, micro, seed=cfg.data.seed, shuffle=cfg.data.shuffle, rank=jax.process_index(),
                         world=jax.process_count(), start_step=step * cfg.optim.grad_accum,
                         prefetch=cfg.data.prefetch, num_workers=cfg.data.num_workers)
    if is_main:
        print(f"[train] train data: {train_ds.total_tokens/1e9:.3f}B tokens in {len(train_ds.files)} shard(s), "
              f"{train_ds.n_windows:,} windows of {cfg.data.seq_len}; "
              f"{total_steps * cfg.tokens_per_step / train_ds.total_tokens:.2f} epoch(s) planned")

    # ---- throughput accounting --------------------------------------------
    fpt = flops_per_token(n_params, cfg.model.n_layers, cfg.data.seq_len, cfg.model.d_model)
    peak = peak_flops(jax.devices()[0].device_kind, cfg.model.dtype)
    logger = MetricsLogger(run_dir, cfg, enabled=is_main)

    # ---- loop --------------------------------------------------------------
    t_start = time.time()
    t_log, tokens_log = t_start, tokens_seen
    last_loss = float("nan")
    val_loss = float("nan")
    first_step = step
    try:
        while step < total_steps:
            if cfg.run.profile and step == first_step + 5 and is_main:
                jax.profiler.start_trace(os.path.join(run_dir, "profile"))
            losses, gnorms = [], []
            for _ in range(cfg.optim.grad_accum):
                batch = put_batch(next(loader), mesh)
                loss, gnorm = train_step(model, optimizer, batch)
                losses.append(loss)
                gnorms.append(gnorm)
            step += 1
            tokens_seen += cfg.tokens_per_step
            if cfg.run.profile and step == first_step + 10 and is_main:
                jax.profiler.stop_trace()

            if step % cfg.run.log_every == 0 or step == total_steps or step == first_step + 1:
                last_loss = float(jnp.mean(jnp.stack(losses)))
                gnorm_v = float(jnp.mean(jnp.stack(gnorms)))
                now = time.time()
                dt = max(1e-6, now - t_log)
                tps = (tokens_seen - tokens_log) / dt
                train_seconds = prev_seconds + (now - t_start)
                gpu_hours = train_seconds * n_devices / 3600
                mfu = (fpt * tps) / (peak * n_devices) if peak else float("nan")
                lr = float(sched(step))
                metrics = {"train/loss": last_loss, "train/ppl": math.exp(min(last_loss, 20)), "train/lr": lr,
                           "train/grad_norm": gnorm_v, "tokens_seen": tokens_seen,
                           "perf/tokens_per_s": tps, "perf/mfu": mfu, "perf/step_time_s": dt / cfg.run.log_every,
                           "cost/gpu_hours": gpu_hours, "cost/usd": gpu_hours * cfg.run.gpu_hour_price_usd,
                           "cost/flops": fpt * tokens_seen, "epoch": loader.epoch_of_step(loader.step)}
                logger.log(step, metrics)
                if is_main:
                    eta = (total_steps - step) * (dt / cfg.run.log_every)
                    print(f"[train] step {step}/{total_steps} loss {last_loss:.4f} lr {lr:.2e} gnorm {gnorm_v:.2f} | "
                          f"{tps/1e3:,.1f}k tok/s mfu {mfu:.1%} | {tokens_seen/1e9:.3f}B tok | "
                          f"{gpu_hours:.2f} GPU-h (${gpu_hours*cfg.run.gpu_hour_price_usd:,.0f}) | eta {_fmt_time(eta)}",
                          flush=True)
                t_log, tokens_log = now, tokens_seen

            if val_ds is not None and cfg.run.eval_every and (step % cfg.run.eval_every == 0 or step == total_steps):
                val_loss = evaluate(model, val_ds, cfg, mesh, cfg.run.eval_batches)
                logger.log(step, {"val/loss": val_loss, "val/ppl": math.exp(min(val_loss, 20)), "tokens_seen": tokens_seen})
                if is_main:
                    print(f"[train] step {step} val loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.1f})", flush=True)

            if step % cfg.run.ckpt_every == 0 or step == total_steps:
                meta = {"step": step, "tokens_seen": tokens_seen, "train_seconds": prev_seconds + time.time() - t_start,
                        "train_loss": last_loss, "val_loss": val_loss, "config": cfg.to_dict()}
                ckpt_lib.save(mgr, step, model, optimizer, meta, force=(step == total_steps))
                if is_main:
                    print(f"[train] checkpoint saved at step {step}", flush=True)
    except KeyboardInterrupt:
        if is_main:
            print("[train] interrupted — saving checkpoint")
        meta = {"step": step, "tokens_seen": tokens_seen, "train_seconds": prev_seconds + time.time() - t_start,
                "train_loss": last_loss, "val_loss": val_loss, "config": cfg.to_dict()}
        ckpt_lib.save(mgr, step, model, optimizer, meta, force=True)
    finally:
        loader.close()
        mgr.wait_until_finished()
        logger.close()

    train_seconds = prev_seconds + time.time() - t_start
    gpu_hours = train_seconds * n_devices / 3600
    summary = {
        "run": cfg.run.name, "finished": step >= total_steps, "step": step, "tokens": tokens_seen,
        "n_params": n_params, "flops": fpt * tokens_seen, "gpu_hours": gpu_hours, "gpu_kind": jax.devices()[0].device_kind,
        "n_devices": n_devices, "usd_equivalent": gpu_hours * cfg.run.gpu_hour_price_usd,
        "train_loss": last_loss, "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else None,
        "train_seconds": train_seconds, "checkpoint_dir": ckpt_dir,
    }
    if is_main:
        path = ckpt_lib.write_run_summary(run_dir, summary)
        print(f"[train] done. summary -> {path}")
        print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_dir"}, indent=2, default=str))
    mgr.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GW1B trainer")
    ap.add_argument("--config", required=True, help="YAML config (configs/*.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="section.field=value", help="override a config value")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints in the run directory")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and compute budget, then exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.dry_run:
        print(cfg.to_yaml())
        print(config_summary(cfg))
        from .budget import estimate, format_estimate
        for gpu in ("v100", "a100"):
            print(format_estimate(estimate(cfg.model.n_params, cfg.resolved_total_steps() * cfg.tokens_per_step, gpu, 8,
                                           dtype=cfg.model.dtype, n_layers=cfg.model.n_layers,
                                           seq_len=cfg.data.seq_len, d_model=cfg.model.d_model)))
        return
    train(cfg, resume=not a.no_resume)


if __name__ == "__main__":
    main()
