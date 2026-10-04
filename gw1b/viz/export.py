"""Reduce a checkpoint to what the 3D visualizer needs: one small tile per weight matrix + statistics.

    python -m gw1b.viz.export --run <run dir> [--step N] [--compare M] [--tile 48] [--out file.json] [--html page.html]

A 1.15B-parameter checkpoint is 4.6 GB; the browser gets ~5 MB: every 2-D parameter becomes an at most tile x tile
grid of block RMS values (magnitude), block means (sign), and block RMS of the change since the previous checkpoint;
vectors (norm scales) become one row. Runs on CPU (set JAX_PLATFORMS=cpu when a GPU is busy next to it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import numpy as np

from ..config import load_config

ROLES = {  # parameter name -> (group, short label) for the layout
    "embed.embedding": ("embed", "token embedding"),
    "wpe.embedding": ("embed", "position embedding"),
    "input_norm.scale": ("norm", "norm (attn)"),
    "post_attn_norm.scale": ("norm", "norm (mlp)"),
    "attn.q_proj.kernel": ("attn", "Q"),
    "attn.k_proj.kernel": ("attn", "K"),
    "attn.v_proj.kernel": ("attn", "V"),
    "attn.o_proj.kernel": ("attn", "O"),
    "mlp.gate_proj.kernel": ("mlp", "gate"),
    "mlp.up_proj.kernel": ("mlp", "up"),
    "mlp.down_proj.kernel": ("mlp", "down"),
    "final_norm.scale": ("norm", "final norm"),
    "lm_head.kernel": ("head", "LM head"),
}


# ---------------------------------------------------------------------------
# checkpoint access (no model object needed: the saved tree is read as plain arrays)
# ---------------------------------------------------------------------------
def checkpoint_steps(run_dir: str) -> list[int]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    import orbax.checkpoint as ocp
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        return sorted(int(s) for s in mgr.all_steps())
    finally:
        mgr.close()


def load_params(run_dir: str, step: int) -> dict[str, np.ndarray]:
    """{'blocks.0.attn.q_proj.kernel': ndarray, ...} for one checkpoint step."""
    import jax
    import orbax.checkpoint as ocp
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        tree = mgr.restore(step, args=ocp.args.Composite(model=ocp.args.StandardRestore()))["model"]
    finally:
        mgr.close()
    out: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = [getattr(k, "key", getattr(k, "name", getattr(k, "idx", k))) for k in path]
        keys = [str(k) for k in keys]
        if keys and keys[-1] == "value":
            keys = keys[:-1]
        out[".".join(keys)] = np.asarray(leaf, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------
def _blocks(x: np.ndarray, n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """View a 2-D array as (rows, bh, cols, bw) blocks with rows <= n_rows, cols <= n_cols (zero-padded),
    plus the matching block view of a ones-mask so padding does not count."""
    h, w = x.shape
    bh, bw = max(1, math.ceil(h / n_rows)), max(1, math.ceil(w / n_cols))
    rows, cols = math.ceil(h / bh), math.ceil(w / bw)
    ones = np.ones_like(x, dtype=np.float32)
    if rows * bh != h or cols * bw != w:
        pad = ((0, rows * bh - h), (0, cols * bw - w))
        x, ones = np.pad(x, pad), np.pad(ones, pad)
    return x.reshape(rows, bh, cols, bw), ones.reshape(rows, bh, cols, bw)


def tile_stats(x: np.ndarray, n: int) -> dict[str, Any]:
    """Block RMS and block mean as flat row-major lists: at most n x n cells for a matrix, 1 x n*n for a vector."""
    if x.ndim == 1:
        b, m = _blocks(x.reshape(1, -1), 1, n * n)
    else:
        b, m = _blocks(x.reshape(x.shape[0], -1), n, n)
    cnt = m.sum(axis=(1, 3)).clip(1)
    rms = np.sqrt((b.astype(np.float64) ** 2).sum(axis=(1, 3)) / cnt)
    mean = b.astype(np.float64).sum(axis=(1, 3)) / cnt
    return {"rows": int(b.shape[0]), "cols": int(b.shape[2]),
            "rms": np.round(rms, 7).astype(float).ravel().tolist(),
            "mean": np.round(mean, 7).astype(float).ravel().tolist()}


def array_stats(x: np.ndarray) -> dict[str, float]:
    f = x.astype(np.float64).ravel()
    return {"mean": float(f.mean()), "std": float(f.std()), "rms": float(np.sqrt((f ** 2).mean())),
            "absmax": float(np.abs(f).max()), "min": float(f.min()), "max": float(f.max())}


def histogram(x: np.ndarray, bins: int = 24) -> dict[str, list[float]]:
    f = x.ravel()
    if f.size > 2_000_000:   # sample for the big embeddings
        f = f[np.random.default_rng(0).choice(f.size, 2_000_000, replace=False)]
    lim = float(np.percentile(np.abs(f), 99.9)) or 1e-6
    counts, edges = np.histogram(f, bins=bins, range=(-lim, lim))
    return {"edges": np.round(edges, 7).astype(float).tolist(), "counts": counts.astype(int).tolist()}


def classify(name: str) -> tuple[str, str, int | None]:
    """'blocks.3.attn.q_proj.kernel' -> ('attn', 'Q', 3); 'embed.embedding' -> ('embed', 'token embedding', None)."""
    parts = name.split(".")
    layer = None
    if parts[0] == "blocks" and len(parts) > 2 and parts[1].isdigit():
        layer = int(parts[1])
        parts = parts[2:]
    key = ".".join(parts)
    group, label = ROLES.get(key, ("other", key))
    return group, label, layer


def metrics_at(run_dir: str, step: int) -> dict[str, Any]:
    """The last metrics.jsonl record at or before `step` (train loss, val loss, tokens, lr, throughput)."""
    path = os.path.join(run_dir, "metrics.jsonl")
    best: dict[str, Any] = {}
    if not os.path.exists(path):
        return best
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("step", 0) <= step:
                best.update({k: v for k, v in rec.items() if v is not None})
            else:
                break
    return best


def export_run(run_dir: str, step: int | None = None, compare: int | None | str = "previous", tile: int = 48,
               hist_bins: int = 24) -> dict[str, Any]:
    """The visualizer's data for one checkpoint of a run (JSON-serialisable)."""
    run_dir = os.path.abspath(run_dir)
    steps = checkpoint_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {run_dir}/checkpoints")
    step = steps[-1] if step is None else int(step)
    if step not in steps:
        raise FileNotFoundError(f"step {step} not in {steps}")
    if compare == "previous":
        earlier = [s for s in steps if s < step]
        compare = earlier[-1] if earlier else None
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    t0 = time.time()
    params = load_params(run_dir, step)
    prev = load_params(run_dir, int(compare)) if compare is not None else {}
    tensors = []
    n_params = 0
    for name in sorted(params, key=_sort_key):
        w = params[name]
        n_params += int(w.size)
        group, label, layer = classify(name)
        entry: dict[str, Any] = {"id": name, "label": label, "group": group, "layer": layer,
                                 "shape": [int(s) for s in w.shape], "n": int(w.size),
                                 "stats": array_stats(w), "hist": histogram(w, hist_bins)}
        entry.update(tile_stats(w, tile))
        if name in prev and prev[name].shape == w.shape:
            d = w - prev[name]
            entry["delta"] = tile_stats(d, tile)["rms"]
            entry["delta_stats"] = array_stats(d)
            entry["delta_stats"]["rel"] = float(entry["delta_stats"]["rms"] / (entry["stats"]["rms"] or 1e-12))
        tensors.append(entry)
    m = cfg.model
    return {
        "run": os.path.basename(run_dir.rstrip("/")), "run_dir": run_dir, "step": step, "steps": steps,
        "compare_step": compare, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "export_seconds": round(time.time() - t0, 1), "tile": tile, "n_params": n_params,
        "model": {"vocab_size": m.vocab_size, "d_model": m.d_model, "n_layers": m.n_layers, "n_heads": m.n_heads,
                  "n_kv_heads": m.n_kv_heads, "head_dim": m.head_dim, "d_ff": m.d_ff, "max_seq_len": m.max_seq_len,
                  "activation": m.activation, "norm": m.norm, "pos": m.pos, "tie_embeddings": m.tie_embeddings,
                  "dtype": m.dtype},
        "metrics": metrics_at(run_dir, step),
        "tensors": tensors,
    }


def _sort_key(name: str) -> tuple:
    parts = name.split(".")
    if parts[0] == "blocks" and parts[1].isdigit():
        return (1, int(parts[1]), name)
    return (0 if parts[0] in ("embed", "wpe") else 2, 0, name)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def clean(obj: Any) -> Any:
    """NaN/inf -> null (browsers reject them in JSON); numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(clean(data), separators=(",", ":"), allow_nan=False)


def write_json(data: dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(dumps(data))
    os.replace(tmp, path)
    return path


def write_html(data: dict[str, Any], path: str) -> str:
    """A single self-contained page: index.html + viz.js + three.js + the data (open it anywhere, no server)."""
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        html = f.read()
    with open(os.path.join(STATIC_DIR, "viz.js")) as f:
        js = f.read()
    with open(os.path.join(STATIC_DIR, "vendor", "three.module.min.js")) as f:
        three = f.read()
    with open(os.path.join(STATIC_DIR, "vendor", "OrbitControls.js")) as f:
        orbit = f.read()
    payload = dumps(data).replace("</", "<\\/")
    assert js.startswith("// GW1B"), "viz.js header changed"
    js_inline = js.replace("import * as THREE from 'three';", "const THREE = await import(window.__gw1b_three);", 1)
    js_inline = js_inline.replace("import { OrbitControls } from './vendor/OrbitControls.js';",
                                  "const { OrbitControls } = await import(window.__gw1b_orbit);", 1)
    assert js_inline != js, "viz.js imports changed; update write_html"
    js_safe = js_inline.replace("</script", "<\\/script")
    # one file, no server, no internet: three.js and OrbitControls become blob-URL modules
    lit = lambda src: json.dumps(src).replace("</", "<\\/")   # a JS string literal that cannot close the <script>
    inline = f"""<script>
window.__gw1b_three = URL.createObjectURL(new Blob([{lit(three)}], {{type: 'text/javascript'}}));
window.__gw1b_orbit = URL.createObjectURL(new Blob([{lit(orbit)}.replace("from 'three'", "from '" + window.__gw1b_three + "'")], {{type: 'text/javascript'}}));
</script>
<script id="gw1b-data" type="application/json">{payload}</script>
<script type="module">{js_safe}</script>"""
    html = html.replace('<script type="importmap">', '<script type="text/plain" data-replaced="importmap">', 1)
    html = html.replace('<script type="module" src="viz.js"></script>', inline, 1)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="export a checkpoint for the 3D visualizer")
    ap.add_argument("--run", required=True, help="run directory (with checkpoints/ and config.yaml)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--compare", default="previous", help="step to diff against: a number, 'previous' (default) or 'none'")
    ap.add_argument("--tile", type=int, default=48, help="max cells per side of a weight tile (default 48)")
    ap.add_argument("--out", default=None, help="JSON output (default: <run>/viz/step-<N>.json)")
    ap.add_argument("--html", default=None, help="also write a self-contained HTML page")
    a = ap.parse_args(argv)
    compare: int | None | str = a.compare
    if isinstance(compare, str) and compare.lower() in ("none", "no", ""):
        compare = None
    elif isinstance(compare, str) and compare.isdigit():
        compare = int(compare)
    data = export_run(a.run, a.step, compare, a.tile)
    out = a.out or os.path.join(a.run, "viz", f"step-{data['step']}.json")
    write_json(data, out)
    print(f"[viz] {data['run']} step {data['step']} ({data['n_params']/1e6:.1f}M params, {len(data['tensors'])} tensors, "
          f"{os.path.getsize(out)/1e6:.1f} MB) -> {out}  [{data['export_seconds']}s]")
    if a.html:
        print(f"[viz] self-contained page -> {write_html(data, a.html)}")


if __name__ == "__main__":
    main()
