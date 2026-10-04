"""Serve the 3D visualizer: the static page plus a small JSON API over the run directories.

    python -m gw1b.viz.server --runs $GW1B_SCRATCH/users/$USER/runs [more dirs...] --port 6007

    GET /                                   the page (gw1b/viz/static/index.html)
    GET /api/runs                           runs under the roots: name, dir, checkpoint steps, last metrics
    GET /api/export?run=<name>&step=<N>     the visualizer data for one checkpoint (exported on first request,
                                            cached in <run>/viz/step-<N>[-vs-<M>].json); &compare=previous|none|<M>
    GET /api/generate?run=&step=&prompt=&max_new=32&temperature=0&seed=0
                                            a traced autoregressive completion (gw1b/viz/trace.py): per generated
                                            token the residual stream per layer, attention over the context and
                                            the next-token distribution; the run's tokenizer (config data.tokenizer)

`gw1b viz` runs this inside your Jupyter job on port 6007 so the `gw1b jupyter` tunnel reaches it. Runs on CPU
(JAX_PLATFORMS=cpu) so it never competes with training for the GPU; a 1.15B checkpoint takes ~1 minute to reduce.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import export as ex
from . import trace as tr

STATIC = ex.STATIC_DIR


def scan_runs(roots: list[str]) -> list[dict]:
    """Every directory with checkpoints/ directly under a root (or the root itself), newest step first."""
    runs = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        candidates = [root] + sorted(os.path.join(root, d) for d in os.listdir(root)) if os.path.isdir(root) else []
        for d in candidates:
            if d in seen or not os.path.isdir(os.path.join(d, "checkpoints")):
                continue
            seen.add(d)
            try:
                steps = ex.checkpoint_steps(d)
            except Exception:  # noqa: BLE001 - a run being written right now
                steps = []
            if not steps:
                continue
            info = {"name": os.path.relpath(d, root) if d != root else os.path.basename(d), "dir": d,
                    "steps": steps, "latest": steps[-1], "metrics": ex.metrics_at(d, steps[-1])}
            cfg_path = os.path.join(d, "config.yaml")
            if os.path.exists(cfg_path):
                try:
                    m = ex.load_config(cfg_path).model
                    info["model"] = {"n_layers": m.n_layers, "d_model": m.d_model, "n_params": m.n_params}
                except Exception:  # noqa: BLE001
                    pass
            runs.append(info)
    runs.sort(key=lambda r: os.path.getmtime(os.path.join(r["dir"], "checkpoints")), reverse=True)
    return runs


class Exports:
    """Export cache: memory + <run>/viz/*.json on disk; one export at a time (they are CPU/RAM heavy)."""

    def __init__(self, tile: int):
        self.tile = tile
        self.mem: dict[tuple, dict] = {}
        self.lock = threading.Lock()

    def get(self, run_dir: str, step: int | None, compare) -> dict:
        steps = ex.checkpoint_steps(run_dir)
        if not steps:
            raise FileNotFoundError(f"no checkpoints in {run_dir}")
        step = steps[-1] if step is None else step
        if compare == "previous":
            earlier = [s for s in steps if s < step]
            compare = earlier[-1] if earlier else None
        key = (run_dir, step, compare, self.tile)
        if key in self.mem:
            return self.mem[key]
        with self.lock:
            if key in self.mem:
                return self.mem[key]
            path = os.path.join(run_dir, "viz", f"step-{step}" + (f"-vs-{compare}" if compare is not None else "") + ".json")
            data = None
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("tile") != self.tile:
                        data = None
                except ValueError:
                    data = None
            if data is None:
                data = ex.export_run(run_dir, step, compare, self.tile)
                try:
                    ex.write_json(data, path)
                except OSError:
                    pass   # read-only run directory (someone else's): memory cache only
            self.mem[key] = data
            if len(self.mem) > 12:   # bound the memory: drop the oldest
                self.mem.pop(next(iter(self.mem)))
            return data


class Models:
    """Numpy models (+ tokenizer) per (run, step) for /api/generate; a few are kept, one generation at a time."""

    def __init__(self):
        self.mem: dict[tuple, tuple] = {}
        self.lock = threading.Lock()

    def get(self, run_dir: str, step: int | None):
        from ..tokenizer import Tokenizer
        steps = ex.checkpoint_steps(run_dir)
        if not steps:
            raise FileNotFoundError(f"no checkpoints in {run_dir}")
        step = steps[-1] if step is None else step
        key = (run_dir, step)
        if key not in self.mem:
            cfg = ex.load_config(os.path.join(run_dir, "config.yaml"))
            tok_path = cfg.data.tokenizer
            if not os.path.exists(tok_path):
                raise FileNotFoundError(f"this run has no tokenizer to encode a prompt with (config data.tokenizer = {tok_path})")
            model = tr.NumpyModel(ex.load_params(run_dir, step), cfg.model)
            self.mem[key] = (model, Tokenizer(tok_path))
            while len(self.mem) > 2:
                self.mem.pop(next(iter(self.mem)))
        return self.mem[key]

    def generate(self, run_dir: str, step: int | None, prompt: str, max_new: int, temperature: float, seed: int) -> dict:
        with self.lock:
            model, tok = self.get(run_dir, step)
            out = tr.trace_generate(model, tok, prompt, max_new=max_new, temperature=temperature, seed=seed)
            out["run"] = os.path.basename(run_dir.rstrip("/"))
            out["step"] = ex.checkpoint_steps(run_dir)[-1] if step is None else step
            return out


class Handler(SimpleHTTPRequestHandler):
    roots: list[str] = []
    exports: Exports
    models: Models

    def log_message(self, fmt, *args):  # quieter log
        if "/api/" in fmt % args or " 200 " not in fmt % args:
            super().log_message(fmt, *args)

    def _json(self, obj, status=HTTPStatus.OK):
        body = ex.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, rel: str):
        path = os.path.normpath(os.path.join(STATIC, rel.lstrip("/")))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript"
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _run_dir(self, name: str) -> str | None:
        runs = scan_runs(self.roots)
        for r in runs:
            if r["name"] == name or r["dir"] == os.path.abspath(name):
                return r["dir"]
        return None

    def do_GET(self):  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._file("index.html")
            if url.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT); self.end_headers(); return None
            if url.path == "/api/runs":
                return self._json({"roots": self.roots, "runs": scan_runs(self.roots)})
            if url.path == "/api/export":
                run_dir = self._run_dir(q.get("run", [""])[0])
                if run_dir is None:
                    return self._json({"error": f"unknown run {q.get('run')}; see /api/runs"}, HTTPStatus.NOT_FOUND)
                step = int(q["step"][0]) if q.get("step") and q["step"][0].isdigit() else None
                cmp_ = q.get("compare", ["previous"])[0]
                compare = None if cmp_ in ("none", "") else int(cmp_) if cmp_.isdigit() else "previous"
                return self._json(self.exports.get(run_dir, step, compare))
            if url.path == "/api/generate":
                run_dir = self._run_dir(q.get("run", [""])[0])
                if run_dir is None:
                    return self._json({"error": f"unknown run {q.get('run')}; see /api/runs"}, HTTPStatus.NOT_FOUND)
                step = int(q["step"][0]) if q.get("step") and q["step"][0].isdigit() else None
                prompt = q.get("prompt", [""])[0]
                max_new = max(1, min(256, int(q.get("max_new", ["32"])[0] or 32)))
                temperature = max(0.0, float(q.get("temperature", ["0"])[0] or 0))
                seed = int(q.get("seed", ["0"])[0] or 0)
                return self._json(self.models.generate(run_dir, step, prompt, max_new, temperature, seed))
            return self._file(url.path)
        except FileNotFoundError as e:
            return self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(roots: list[str], host: str = "0.0.0.0", port: int = 6007, tile: int = 48) -> ThreadingHTTPServer:
    Handler.roots = [os.path.abspath(r) for r in roots]
    Handler.exports = Exports(tile)
    Handler.models = Models()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main(argv=None) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")   # never take GPU memory from the training/notebook next to us
    ap = argparse.ArgumentParser(description="GW1B 3D model visualizer server")
    ap.add_argument("--runs", nargs="+", default=[os.path.join(os.environ.get("GW1B_SCRATCH", "."), "users",
                                                                os.environ.get("USER", "student"), "runs")],
                    help="directories whose sub-directories are runs (default: your runs folder)")
    ap.add_argument("--port", type=int, default=6007)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tile", type=int, default=48)
    a = ap.parse_args(argv)
    httpd = serve(a.runs, a.host, a.port, a.tile)
    n = len(scan_runs(Handler.roots))
    print(f"[viz] {n} run(s) under {', '.join(Handler.roots)}", flush=True)
    print(f"[viz] serving on http://{a.host}:{a.port}  (through the gw1b jupyter tunnel: http://localhost:{a.port})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
