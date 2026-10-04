"""Token shards on disk + a deterministic, resumable, data-parallel batch loader.

Shard format (`*.bin`, same idea as llm.c / nanoGPT):
    1024-byte header = 256 x int32 : [MAGIC, VERSION, n_tokens, 0, 0, ...]
    followed by n_tokens x uint16 (vocab <= 65535) or uint32.

Documents are concatenated with the tokenizer's EOS id between them at tokenization time, so a
training window of `seq_len + 1` tokens is simply a contiguous slice ("packed" pretraining).

The loader is deterministic given (seed, step): every process in a multi-GPU/multi-node job computes
the same global permutation and takes its own slice, and a restarted job continues exactly where it
stopped. That is what makes checkpoint/resume reproducible.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import threading
from dataclasses import dataclass

import numpy as np

MAGIC = 20260901
VERSION = 1
HEADER_BYTES = 1024


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
class ShardWriter:
    """Accumulates token ids and writes fixed-size shards: <out_dir>/<prefix>_000000.bin ..."""

    def __init__(self, out_dir: str, prefix: str = "train", shard_size: int = 100_000_000, dtype=np.uint16):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.shard_size, self.dtype = out_dir, prefix, shard_size, dtype
        self.buf = np.empty(shard_size, dtype=dtype)
        self.n = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.files: list[str] = []

    def add(self, tokens) -> None:
        tokens = np.asarray(tokens, dtype=self.dtype)
        while tokens.size:
            room = self.shard_size - self.n
            take = tokens[:room]
            self.buf[self.n:self.n + take.size] = take
            self.n += take.size
            tokens = tokens[room:]
            if self.n == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        if self.n == 0:
            return
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:06d}.bin")
        write_shard(path, self.buf[:self.n])
        self.files.append(path)
        self.total_tokens += self.n
        self.shard_idx += 1
        self.n = 0

    def close(self) -> dict:
        self._flush()
        manifest = {"prefix": self.prefix, "n_shards": self.shard_idx, "total_tokens": int(self.total_tokens),
                    "dtype": np.dtype(self.dtype).name, "files": [os.path.basename(f) for f in self.files]}
        with open(os.path.join(self.out_dir, f"{self.prefix}_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def write_shard(path: str, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, tokens.size
    header[3] = 2 if tokens.dtype == np.uint16 else 4
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(tokens).tobytes())


def read_shard(path: str) -> np.ndarray:
    """Memory-map a shard (no copy)."""
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"{path}: not a GW1B token shard"
    n, itemsize = int(header[2]), int(header[3]) or 2
    dtype = np.uint16 if itemsize == 2 else np.uint32
    return np.memmap(path, dtype=dtype, mode="r", offset=HEADER_BYTES, shape=(n,))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
class TokenDataset:
    """All shards in a directory, viewed as non-overlapping windows of `seq_len + 1` tokens."""

    def __init__(self, path: str | list[str], seq_len: int):
        paths = [path] if isinstance(path, str) else list(path)
        files: list[str] = []
        for p in paths:
            files += sorted(glob.glob(os.path.join(p, "*.bin"))) if os.path.isdir(p) else sorted(glob.glob(p))
        if not files:
            raise FileNotFoundError(
                f"No token shards (*.bin) found in {path}. The class dataset is built once by the instructor with "
                f"`gw1b-admin data` (FineWeb-Edu -> $GW1B_SCRATCH/data/fineweb-edu-10B); for your own data point "
                f"data.train_dir / data.val_dir at a folder of shards written by gw1b.prepare_data (notebook 02).")
        self.files = files
        self.seq_len = seq_len
        self.shards = [read_shard(f) for f in files]
        self.windows_per_shard = np.array([(len(s) - 1) // seq_len for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.windows_per_shard)])
        self.n_windows = int(self.offsets[-1])
        self.total_tokens = int(sum(len(s) for s in self.shards))
        if self.n_windows == 0:
            raise ValueError(f"Shards in {path} are shorter than seq_len+1={seq_len + 1}")

    def __len__(self) -> int:
        return self.n_windows

    def window(self, idx: int) -> np.ndarray:
        shard = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        j = int(idx - self.offsets[shard])
        start = j * self.seq_len
        return np.asarray(self.shards[shard][start:start + self.seq_len + 1], dtype=np.int32)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        w = np.stack([self.window(int(i)) for i in indices])
        return {"inputs": w[:, :-1], "targets": w[:, 1:]}


@dataclass
class LoaderState:
    step: int


class BatchLoader:
    """Deterministic, resumable loader.

    global batch `step` (0-based) = rows [step*B, (step+1)*B) of the epoch permutation, with
    epoch = (step*B) // n_windows and permutation = rng(seed + epoch).permutation(n_windows).
    Process `rank` of `world` receives rows [rank*B/world, (rank+1)*B/world) of that global batch.
    """

    def __init__(self, dataset: TokenDataset, batch_size: int, seed: int = 0, shuffle: bool = True,
                 rank: int = 0, world: int = 1, start_step: int = 0, prefetch: int = 4, num_workers: int = 2):
        assert batch_size % world == 0, "global batch size must be divisible by the number of processes"
        self.ds, self.B, self.seed, self.shuffle = dataset, batch_size, seed, shuffle
        self.rank, self.world = rank, world
        self.local_B = batch_size // world
        self.step = start_step
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None
        self.prefetch = prefetch
        self._q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._stop = threading.Event()
        self._next_to_produce = start_step
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(max(1, num_workers))]
        self._pending: dict[int, dict] = {}
        for t in self._threads:
            t.start()

    # -- indexing ------------------------------------------------------------
    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            if self.shuffle:
                self._perm = np.random.default_rng(self.seed + epoch).permutation(self.ds.n_windows)
            else:
                self._perm = np.arange(self.ds.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def indices_for_step(self, step: int) -> np.ndarray:
        n = self.ds.n_windows
        first = step * self.B
        epoch, pos = divmod(first, n)
        perm = self._permutation(epoch)
        rows = perm[pos:pos + self.B]
        if rows.size < self.B:  # batch straddles an epoch boundary
            rows = np.concatenate([rows, self._permutation(epoch + 1)[:self.B - rows.size]])
        lo, hi = self.rank * self.local_B, (self.rank + 1) * self.local_B
        return rows[lo:hi]

    def epoch_of_step(self, step: int) -> float:
        return step * self.B / self.ds.n_windows

    # -- prefetching ---------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                step = self._next_to_produce
                self._next_to_produce += 1
                idx = self.indices_for_step(step)
            batch = self.ds.batch(idx)
            while not self._stop.is_set():
                try:
                    self._q.put((step, batch), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        # batches may arrive slightly out of order from multiple workers; hand them out in order
        while self.step not in self._pending:
            step, batch = self._q.get()
            self._pending[step] = batch
        batch = self._pending.pop(self.step)
        self.step += 1
        return batch

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# helpers for notebooks / tests
# ---------------------------------------------------------------------------
def write_synthetic_dataset(out_dir: str, vocab_size: int = 512, n_tokens: int = 200_000, seed: int = 0,
                            shard_size: int = 50_000, prefix: str = "train") -> dict:
    """A tiny random-token dataset with some learnable structure (useful for CPU smoke tests)."""
    rng = np.random.default_rng(seed)
    # a random bigram process: each token has 4 equally likely successors -> optimal loss = ln(4) ~ 1.39
    # (the successor table is fixed, so train/val sets made with different seeds share the same "language")
    successors = np.random.default_rng(1234).integers(0, vocab_size, size=(vocab_size, 4))
    choice = rng.integers(0, 4, size=n_tokens)
    toks = np.empty(n_tokens, dtype=np.int64)
    toks[0] = rng.integers(0, vocab_size)
    for i in range(1, n_tokens):
        toks[i] = successors[toks[i - 1], choice[i]]
    w = ShardWriter(out_dir, prefix=prefix, shard_size=shard_size)
    w.add(toks)
    return w.close()
