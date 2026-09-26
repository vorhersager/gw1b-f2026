"""Run EleutherAI's lm-evaluation-harness directly on a JAX GW1B checkpoint (no PyTorch conversion).

    python -m gw1b.lm_eval_adapter --run $GW1B_SCRATCH/checkpoints/gw1b-base \
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande --limit 500 --batch-size 16

Loglikelihood tasks (HellaSwag, ARC, PIQA, Winogrande, MMLU, ...) use batched teacher-forced scoring;
generation tasks (GSM8K, ...) use gw1b.generate with a KV cache. Results land in <run>/lm_eval/.
Benchmark data is downloaded from Hugging Face on first use — run once on a login node (or set
HF_DATASETS_OFFLINE=1 after warming the cache in $HF_HOME).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from .generate import generate
from .model import GW1BModel
from .tokenizer import Tokenizer


@nnx.jit
def _logprobs(model, tokens):
    """tokens [B, T] -> log-probs of the *next* token at each position, [B, T, V]."""
    return jax.nn.log_softmax(model(tokens), axis=-1)


def _bucket(n: int, step: int = 128, max_len: int = 2048) -> int:
    return min(max_len, int(math.ceil(n / step) * step))


class GW1BLM:
    """lm-eval `TemplateLM` implementation around a GW1B model. Created lazily so importing this module
    does not require lm-eval to be installed."""

    def __new__(cls, *args, **kwargs):
        from lm_eval.api.model import TemplateLM

        class _GW1BLM(TemplateLM):
            def __init__(self, model: GW1BModel, tokenizer: Tokenizer, batch_size: int = 16,
                         max_length: int | None = None):
                super().__init__()
                self.model, self.tokenizer = model, tokenizer
                self.batch_size = int(batch_size)
                self.max_length = int(max_length or model.cfg.max_seq_len)

            # --- required API ------------------------------------------------
            @property
            def eot_token_id(self) -> int:
                return self.tokenizer.eos_id

            @property
            def max_gen_toks(self) -> int:
                return 256

            def tok_encode(self, string: str, add_special_tokens=None, **kwargs) -> list[int]:
                return self.tokenizer.encode(string)

            def tok_decode(self, tokens) -> str:
                return self.tokenizer.decode(list(tokens))

            def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
                # requests: [((context, continuation), context_enc, continuation_enc), ...]
                order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
                results: list[tuple[float, bool] | None] = [None] * len(requests)
                for start in tqdm(range(0, len(order), self.batch_size), disable=disable_tqdm, desc="loglikelihood"):
                    idx = order[start:start + self.batch_size]
                    seqs, cont_lens = [], []
                    for i in idx:
                        _, ctx, cont = requests[i]
                        full = (list(ctx) + list(cont))[-(self.max_length + 1):]
                        seqs.append(full)
                        cont_lens.append(min(len(cont), len(full) - 1))
                    T = _bucket(max(len(s) for s in seqs) - 1, max_len=self.max_length)
                    pad = self.tokenizer.pad_id if self.tokenizer.pad_id >= 0 else 0
                    inputs = np.full((len(seqs), T), pad, dtype=np.int32)
                    for r, s in enumerate(seqs):
                        s = s[-(T + 1):]
                        inputs[r, :len(s) - 1] = s[:-1]
                    lp = np.asarray(_logprobs(self.model, jnp.asarray(inputs)))
                    for r, (i, s, n_cont) in enumerate(zip(idx, seqs, cont_lens)):
                        s = s[-(T + 1):]
                        n_in = len(s) - 1
                        # continuation tokens are the last n_cont targets: predicted at positions n_in-n_cont .. n_in-1
                        positions = np.arange(n_in - n_cont, n_in)
                        targets = np.asarray(s[1:], dtype=np.int64)[positions]
                        token_lp = lp[r, positions, targets]
                        greedy = lp[r, positions].argmax(-1)
                        results[i] = (float(token_lp.sum()), bool((greedy == targets).all()))
                return results  # type: ignore[return-value]

            def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
                from lm_eval import utils
                out = []
                for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm, desc="rolling"):
                    windows = list(map(utils.make_disjoint_window,
                                       utils.get_rolling_token_windows(token_list=self.tok_encode(string),
                                                                       prefix_token=self.prefix_token_id,
                                                                       max_seq_len=self.max_length, context_len=1)))
                    reqs = [(None, ctx, cont) for ctx, cont in windows]
                    out.append(sum(x[0] for x in self._loglikelihood_tokens(reqs, disable_tqdm=True)))
                return out

            def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
                results = [None] * len(requests)
                groups: dict[str, list[int]] = {}
                for i, req in enumerate(requests):
                    groups.setdefault(json.dumps(req.args[1], sort_keys=True), []).append(i)
                for key, idxs in groups.items():
                    gen_kwargs = dict(json.loads(key))
                    until = gen_kwargs.get("until", None) or []
                    if isinstance(until, str):
                        until = [until]
                    max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                    temperature = float(gen_kwargs.get("temperature", 0.0)) if gen_kwargs.get("do_sample", False) else 0.0
                    for start in tqdm(range(0, len(idxs), self.batch_size), disable=disable_tqdm, desc="generate"):
                        batch = idxs[start:start + self.batch_size]
                        prompts = []
                        for i in batch:  # keep the last (max_length - max_new) tokens of each context
                            ids = self.tok_encode(requests[i].args[0])[-(self.max_length - max_new):]
                            prompts.append(self.tok_decode(ids))
                        texts = generate(self.model, self.tokenizer, prompts, max_new_tokens=max_new,
                                         temperature=temperature, stop=until)
                        for i, t in zip(batch, texts):
                            results[i] = t
                return results  # type: ignore[return-value]

        return _GW1BLM(*args, **kwargs)


def run(run_dir: str, tasks: list[str], step: int | None = None, limit: int | None = None, batch_size: int = 16,
        num_fewshot: int | None = None, out_dir: str | None = None) -> dict:
    import lm_eval
    from .evaluate import load_run

    model, cfg, loaded_step = load_run(run_dir, step, shard_params=False)
    tok = Tokenizer(cfg.data.tokenizer)
    lm = GW1BLM(model, tok, batch_size=batch_size)
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit, batch_size=batch_size,
                                  log_samples=False)
    results = res["results"] if isinstance(res, dict) else res.results  # EvalResults in newer versions
    table = {task: {k: v for k, v in metrics.items() if isinstance(v, (int, float))} for task, metrics in results.items()}
    summary = {"run": run_dir, "step": loaded_step, "tasks": tasks, "limit": limit, "num_fewshot": num_fewshot,
               "n_params": cfg.model.n_params, "seconds": time.time() - t0, "results": table}
    out_dir = out_dir or os.path.join(run_dir, "lm_eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step{loaded_step}_{'-'.join(tasks)[:60]}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(table, indent=2))
    print(f"[lm_eval] wrote {path}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="lm-evaluation-harness on a GW1B checkpoint")
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tasks", default="hellaswag,arc_easy,piqa", help="comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (for quick checks)")
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.run, a.tasks.split(","), a.step, a.limit, a.batch_size, a.num_fewshot, a.out)


if __name__ == "__main__":
    main()
