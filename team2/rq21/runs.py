"""Reading and validating run directories, and saving results."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import jax

from gw1b.config import TrainConfig, load_config

N_SHAPES = 5
BUDGETS = (20, 60)  # tokens/param
SHAPE_FIELDS = ("n_layers", "d_model", "d_ff", "n_heads", "n_kv_heads", "head_dim")
# May differ between runs: what the study varies, plus settings that cannot change the trained model.
IGNORED_SETTINGS = {
    *(f"model.{f}" for f in SHAPE_FIELDS),
    "run.name", "run.out_dir", "run.total_tokens", "run.total_steps",
    "run.log_every", "run.eval_every", "run.eval_batches", "run.eval_seqs", "run.ckpt_every", "run.ckpt_keep",
    "run.ckpt_keep_every", "run.shard_params", "run.remat", "run.loss_chunk", "run.profile", "run.wandb",
    "run.tensorboard", "run.gpu_hour_price_usd", "data.prefetch", "data.num_workers",
}


def find_runs(runs_root: str, prefix: str = "rq_2_1_100m_") -> list[str]:
    """Return the sorted run directories under runs_root whose names start with prefix and contain config.yaml."""
    return sorted(str(d) for d in Path(runs_root).iterdir()
                  if d.name.startswith(prefix) and (d / "config.yaml").is_file())


def select_runs(args: argparse.Namespace) -> dict[str, str]:
    """Return {run name: run directory} for a stage's --runs-root, --prefix and --run options.

    Prints why and returns an empty dict if no runs match, or if a name given with --run does not exist.
    """
    run_dirs = {Path(d).name: d for d in find_runs(args.runs_root, args.prefix)}
    names = args.run or sorted(run_dirs)
    unknown = [n for n in names if n not in run_dirs]
    if unknown or not names:
        print(f"No matching runs in {args.runs_root}" + (f": {', '.join(unknown)}" if unknown else ""))
        return {}
    return {n: run_dirs[n] for n in names}


def read_run(run_dir: str) -> dict[str, Any]:
    """Return one run's name, config, summary (None if missing) and the values calculated from them.

    Calculated: n_params, embedding share of parameters, log2(n_layers / d_model) and tokens/param. Tokens are the
    actual count from the summary, or the planned count from the config if the summary is missing. Other shape
    fields are read from run["config"].model.
    """
    path = Path(run_dir)
    cfg = load_config(str(path / "config.yaml"))
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    m = cfg.model
    tokens = summary["tokens"] if summary else cfg.resolved_total_steps() * cfg.tokens_per_step
    return {
        "name": path.name,
        "dir": str(path),
        "config": cfg,
        "summary": summary,
        "n_params": m.n_params,
        "embedding_share": m.vocab_size * m.d_model / m.n_params,
        "aspect": math.log2(m.n_layers / m.d_model),
        "tokens": tokens,
        "tokens_per_param": tokens / m.n_params,
    }


def validate(args: argparse.Namespace) -> int:
    """Print the shape table and every problem found, and return 1 if any run fails a check.

    Fails on: no runs, a missing or repeated (shape, budget) cell, a budget more than 5% from 20 or 60 tokens/param,
    an unfinished run, a non-finite validation loss, n_params more than 2% from args.nominal_params, or any
    setting other than shape and token budget differing between runs. Reports without failing: training FLOPs,
    GPU, grad_accum, and whether run.eval_seqs was set.
    """
    runs = [read_run(d) for d in find_runs(args.runs_root, args.prefix)]
    if not runs:
        print(f"No runs starting with {args.prefix!r} in {args.runs_root}")
        return 1

    problems = []
    cells: Counter = Counter()
    shapes = set()
    rows = []
    for r in runs:
        m, s = r["config"].model, r["summary"]
        budget = min(BUDGETS, key=lambda b: abs(r["tokens_per_param"] - b))
        if abs(r["tokens_per_param"] - budget) / budget > 0.05:
            problems.append(f"{r['name']}: {r['tokens_per_param']:.2f} tokens/param is not near any of {BUDGETS}")
        shape = tuple(getattr(m, f) for f in SHAPE_FIELDS)
        shapes.add(shape)
        cells[shape, budget] += 1
        if abs(r["n_params"] - args.nominal_params) / args.nominal_params > 0.02:
            problems.append(f"{r['name']}: {r['n_params']:,} parameters is over 2% from {args.nominal_params:,.0f}")
        if s is None:
            problems.append(f"{r['name']}: no summary.json")
        else:
            if not s.get("finished"):
                problems.append(f"{r['name']}: not finished (step {s.get('step')})")
            if not math.isfinite(s.get("val_loss") or math.nan):
                problems.append(f"{r['name']}: validation loss is {s.get('val_loss')}")
        s = s or {}
        rows.append((r["aspect"], budget, f"{r['name']:<24} {m.n_layers:>3} {m.d_model:>5} {m.d_ff:>5} {m.n_heads:>3} "
                     f"{m.n_kv_heads:>3} {m.head_dim:>3} {r['n_params'] / 1e6:>7.1f} {r['embedding_share']:>5.1%} "
                     f"{r['aspect']:>6.2f} {r['tokens_per_param']:>6.2f} {s.get('flops', math.nan):>9.2e} "
                     f"{str(s.get('grad_accum', '-')):>5} {str(r['config'].run.eval_seqs):>9} "
                     f"{s.get('gpu_kind', '-')}"))

    if len(shapes) != N_SHAPES:
        problems.append(f"found {len(shapes)} shapes, expected {N_SHAPES}")
    for shape in sorted(shapes):
        for budget in BUDGETS:
            if cells[shape, budget] != 1:
                problems.append(f"{dict(zip(SHAPE_FIELDS, shape))} at t{budget}: {cells[shape, budget]} runs, expected 1")

    settings = [_comparable_settings(r["config"]) for r in runs]
    for key in settings[0]:
        values = {repr(s[key]) for s in settings}
        if len(values) > 1:
            problems.append(f"{key} differs between runs: {', '.join(sorted(values))}")

    print(f"{'run':<24} {'L':>3} {'d':>5} {'ff':>5} {'H':>3} {'K':>3} {'hd':>3} {'params_M':>7} {'emb':>5} "
          f"{'aspect':>6} {'tok/p':>6} {'flops':>9} {'accum':>5} {'eval_seqs':>9} gpu")
    for *_, row in sorted(rows, key=lambda x: x[:2]):
        print(row)
    print(f"\n{len(runs)} runs, {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


def _comparable_settings(cfg: TrainConfig) -> dict[str, Any]:
    """Flatten a config to section.field values, leaving out the fields that may differ between runs."""
    flat = {f"{section}.{k}": v for section, values in cfg.to_dict().items() for k, v in values.items()}
    return {k: v for k, v in flat.items() if k not in IGNORED_SETTINGS}


def save_json(path: str, payload: dict[str, Any], step: int | None = None) -> None:
    """Write payload to path with the device kind and checkpoint step added, creating the folder if needed.

    Raises on non-finite numbers instead of writing NaN, and leaves no file behind when it does.
    """
    text = json.dumps({**payload, "device_kind": jax.devices()[0].device_kind, "step": step}, indent=2, allow_nan=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
