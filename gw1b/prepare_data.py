"""Build the pretraining corpus: download FineWeb-Edu -> sample text for the tokenizer -> token shards.

Step 1 (login node — it has internet; compute nodes may not):
    python -m gw1b.prepare_data download --sample 10BT            # ~28 GB of parquet into $HF_HOME
Step 2 (anywhere):
    python -m gw1b.prepare_data sample-text --sample 10BT --out $GW1B_SCRATCH/data/tokenizer_corpus.txt --mb 2000
    python -m gw1b.tokenizer train --input $GW1B_SCRATCH/data/tokenizer_corpus.txt --out $GW1B_GROUP/tokenizer/gw1b-32k
Step 3 (CPU batch job, 40 cores: `gw1b run --cpu slurm/tokenize.sbatch` or the command below):
    python -m gw1b.prepare_data tokenize --sample 10BT --tokenizer $GW1B_GROUP/tokenizer/gw1b-32k.model \
        --out $GW1B_SCRATCH/data/fineweb-edu-10B --workers 40

Team 1 plugs its own filtering / mixing in by passing `--filter my_module:my_filter_fn` (a function
text -> bool) or by writing its own document iterator and calling `tokenize_documents()`.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import time
from typing import Callable, Iterable, Iterator

import numpy as np

from .data import ShardWriter

REPO = "HuggingFaceFW/fineweb-edu"
SAMPLES = {"10BT": "sample/10BT", "100BT": "sample/100BT", "350BT": "sample/350BT"}


def hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def local_dir(sample: str) -> str:
    return os.path.join(hf_home(), "fineweb-edu", sample)


def download(sample: str = "10BT", max_workers: int = 8) -> str:
    from huggingface_hub import snapshot_download
    out = local_dir(sample)
    os.makedirs(out, exist_ok=True)
    print(f"[data] downloading {REPO}/{SAMPLES[sample]} -> {out}")
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{SAMPLES[sample]}/*"], local_dir=out,
                      max_workers=max_workers)
    files = parquet_files(sample)
    print(f"[data] {len(files)} parquet files, {sum(os.path.getsize(f) for f in files)/1e9:.1f} GB")
    return out


def parquet_files(sample: str = "10BT") -> list[str]:
    files = sorted(glob.glob(os.path.join(local_dir(sample), SAMPLES[sample], "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {local_dir(sample)} — run `prepare_data download` first")
    return files


def iter_documents(files: list[str], text_column: str = "text", batch_rows: int = 2048) -> Iterator[str]:
    """Stream document texts out of parquet files without loading them fully into memory."""
    import pyarrow.parquet as pq
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[text_column]):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text


def sample_text(sample: str, out_path: str, mb: int = 2000, files_limit: int | None = None) -> str:
    """Write ~mb megabytes of raw text (one document per line) for tokenizer training."""
    files = parquet_files(sample)[:files_limit]
    limit = mb * 1_000_000
    n = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in iter_documents(files):
            line = doc.replace("\n", " ").strip()
            f.write(line + "\n")
            n += len(line) + 1
            if n >= limit:
                break
    print(f"[data] wrote {n/1e6:.0f} MB of text to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
_TOK = None
_FILTER: Callable[[str], bool] | None = None


def _init_worker(tokenizer_path: str, filter_spec: str | None) -> None:
    global _TOK, _FILTER
    from .tokenizer import Tokenizer
    _TOK = Tokenizer(tokenizer_path)
    _FILTER = load_filter(filter_spec) if filter_spec else None


def _encode_chunk(docs: list[str]) -> tuple[np.ndarray, int, int]:
    kept = [d for d in docs if _FILTER is None or _FILTER(d)]
    ids = _TOK.encode_batch(kept, add_eos=True) if kept else []
    flat = np.concatenate([np.asarray(x, dtype=np.uint16) for x in ids]) if ids else np.zeros(0, np.uint16)
    return flat, len(docs), len(kept)


def load_filter(spec: str) -> Callable[[str], bool]:
    """'my_module:my_function' -> callable(text) -> bool."""
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _chunks(it: Iterable[str], size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def tokenize_documents(docs: Iterable[str], tokenizer_path: str, out_dir: str, val_docs: int = 20_000,
                       workers: int = 8, shard_size: int = 100_000_000, filter_spec: str | None = None,
                       chunk_docs: int = 512, meta: dict | None = None) -> dict:
    """Tokenize an iterator of documents into train/ and val/ shards (the first `val_docs` go to val/)."""
    from .tokenizer import Tokenizer
    tok = Tokenizer(tokenizer_path)
    assert tok.vocab_size <= 65535, "uint16 shards need vocab <= 65535"
    train_w = ShardWriter(os.path.join(out_dir, "train"), "train", shard_size)
    val_w = ShardWriter(os.path.join(out_dir, "val"), "val", shard_size)
    t0 = time.time()
    n_docs = n_kept = 0
    docs_iter = iter(docs)
    with mp.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path, filter_spec)) as pool:
        for flat, seen, kept in pool.imap(_encode_chunk, _chunks(docs_iter, chunk_docs), chunksize=1):
            writer = val_w if n_docs < val_docs else train_w
            writer.add(flat)
            n_docs += seen
            n_kept += kept
            if n_docs % (chunk_docs * 200) == 0:
                total = train_w.total_tokens + train_w.n + val_w.total_tokens + val_w.n
                rate = total / max(1e-6, time.time() - t0)
                print(f"[data] {n_docs:,} docs ({n_kept:,} kept) -> {total/1e9:.3f}B tokens, {rate/1e6:.2f}M tok/s",
                      flush=True)
    train_m, val_m = train_w.close(), val_w.close()
    manifest = {
        "tokenizer": tokenizer_path, "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size,
        "documents_seen": n_docs, "documents_kept": n_kept, "filter": filter_spec,
        "train_tokens": train_m["total_tokens"], "val_tokens": val_m["total_tokens"],
        "train_shards": train_m["n_shards"], "val_shards": val_m["n_shards"], "seconds": time.time() - t0,
        **(meta or {}),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[data] done: {manifest['train_tokens']/1e9:.3f}B train + {manifest['val_tokens']/1e6:.1f}M val tokens "
          f"in {manifest['seconds']/60:.1f} min -> {out_dir}")
    return manifest


def tokenize_sample(sample: str, tokenizer_path: str, out_dir: str, workers: int, files_limit: int | None = None,
                    filter_spec: str | None = None, max_docs: int | None = None, val_docs: int = 20_000) -> dict:
    files = parquet_files(sample)[:files_limit]
    docs: Iterable[str] = iter_documents(files)
    if max_docs:
        import itertools
        docs = itertools.islice(docs, max_docs)
    meta = {"source": f"{REPO}/{SAMPLES[sample]}", "files": [os.path.basename(f) for f in files],
            "files_sha1": hashlib.sha1("".join(os.path.basename(f) for f in files).encode()).hexdigest()[:12]}
    return tokenize_documents(docs, tokenizer_path, out_dir, val_docs=val_docs, workers=workers,
                              filter_spec=filter_spec, meta=meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GW1B data preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download");        d.add_argument("--sample", default="10BT", choices=SAMPLES)
    d.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("sample-text");     s.add_argument("--sample", default="10BT", choices=SAMPLES)
    s.add_argument("--out", required=True); s.add_argument("--mb", type=int, default=2000)
    s.add_argument("--files-limit", type=int, default=None)
    t = sub.add_parser("tokenize");        t.add_argument("--sample", default="10BT", choices=SAMPLES)
    t.add_argument("--tokenizer", required=True); t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t.add_argument("--files-limit", type=int, default=None, help="only the first N parquet files (quick tests)")
    t.add_argument("--max-docs", type=int, default=None)
    t.add_argument("--val-docs", type=int, default=20_000)
    t.add_argument("--filter", default=None, help="module:function returning True to keep a document")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.sample, a.workers)
    elif a.cmd == "sample-text":
        sample_text(a.sample, a.out, a.mb, a.files_limit)
    elif a.cmd == "tokenize":
        tokenize_sample(a.sample, a.tokenizer, a.out, a.workers, a.files_limit, a.filter, a.max_docs, a.val_docs)


if __name__ == "__main__":
    main()
