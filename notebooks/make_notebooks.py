"""Generates the starter notebooks (kept as a script so they stay valid JSON and easy to diff).
Run:  python notebooks/make_notebooks.py
"""
import nbformat as nbf
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''# --- GW1B setup cell (same in every notebook) -------------------------------------------------
import os, sys, json, time, pathlib
GW1B_HOME    = os.environ.get("GW1B_HOME", "/SEAS/groups/gw1b/gw1b-f2026")
GW1B_SCRATCH = os.environ.get("GW1B_SCRATCH", "/scratch/gw1b-class")
GW1B_GROUP   = os.environ.get("GW1B_GROUP", "/SEAS/groups/gw1b")
USER_DIR     = os.path.join(GW1B_SCRATCH, "users", os.environ.get("USER", "student"))
os.makedirs(USER_DIR, exist_ok=True)
if GW1B_HOME not in sys.path:
    sys.path.insert(0, GW1B_HOME)        # the shared gw1b package
import jax, jax.numpy as jnp, numpy as np
print("JAX", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())
print("your scratch dir:", USER_DIR)
'''


def nb(cells, title):
    n = nbf.v4.new_notebook()
    n["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}
    n["cells"] = [nbf.v4.new_markdown_cell(f"# {title}")] + [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


# ---------------------------------------------------------------------------------------------
nb00 = nb([
    ("md", """Welcome to the GW1B environment. This notebook checks that everything works and shows the
few things you need to know about where files live and how much compute you have.

**How you got here:** on Pegasus you ran `gw1b jupyter`, opened the ssh tunnel on your laptop, and
opened `http://localhost:8888/lab?token=...` — or connected Google Colab to it via *Connect → Connect to a local runtime*.
Either way, **the code runs on a Pegasus GPU node**, not on your laptop or on Google's servers."""),
    ("code", SETUP),
    ("md", "## 1. Do we have a GPU, and how fast is it?"),
    ("code", '''dev = jax.devices()[0]
print("device kind:", dev.device_kind, "| platform:", dev.platform)
x = jnp.ones((4096, 4096), jnp.bfloat16 if dev.platform == "gpu" else jnp.float32)
f = jax.jit(lambda a: (a @ a).sum())
f(x).block_until_ready()                      # first call compiles
t = time.time(); [f(x).block_until_ready() for _ in range(5)]; dt = (time.time() - t) / 5
print(f"matmul throughput: {2 * 4096**3 / dt / 1e12:.1f} TFLOP/s  (A100 bf16 peak ~312, V100 ~15 in f32)")'''),
    ("md", """## 2. Where things live

| what | where | notes |
|---|---|---|
| shared environment, code, tokenizer | `$GW1B_GROUP` (`/SEAS/groups/gw1b`) | persistent, read-only for students |
| datasets, HF cache, checkpoints | `$GW1B_SCRATCH` (`/scratch/gw1b-class`) | fast GPFS, **not backed up** — copy results out |
| your experiments | `$GW1B_SCRATCH/users/<netid>/runs/<run name>` | `config.yaml`, `metrics.jsonl`, `checkpoints/`, `summary.json` |
| your home | `~` (25 GB quota) | notebooks, small files only |"""),
    ("code", '''for p in [GW1B_HOME, GW1B_GROUP, GW1B_SCRATCH, USER_DIR]:
    print(f"{'exists' if os.path.exists(p) else 'MISSING':8s} {p}")
data_dir = os.path.join(GW1B_SCRATCH, "data")
print("\\ndatasets available:", os.listdir(data_dir) if os.path.isdir(data_dir) else "(none yet)")'''),
    ("md", """## 3. Compute is the experimental variable

Every GW1B experiment reports *performance, tokens, FLOPs, GPU-hours and dollars*. The budget calculator
tells you what a run costs **before** you submit it."""),
    ("code", '''from gw1b.budget import estimate, format_estimate
from gw1b.config import load_config
for cfg_name in ["50m", "350m", "gw1b_1p15b"]:
    cfg = load_config(os.path.join(GW1B_HOME, "configs", cfg_name + ".yaml"))
    tokens = 20 * cfg.model.n_params                         # Chinchilla-optimal
    e = estimate(cfg.model.n_params, tokens, gpu="a100", n_gpus=8, mfu=0.35,
                 n_layers=cfg.model.n_layers, seq_len=cfg.data.seq_len, d_model=cfg.model.d_model)
    print(cfg_name, "->", format_estimate(e).splitlines()[1].strip())'''),
    ("md", """## 4. The workflow in one picture

```
laptop (browser / Colab UI) ──ssh tunnel──▶ Pegasus login node ──Slurm──▶ GPU node: JupyterLab + your kernel
                                                                    └──▶ batch jobs: gw1b run -m gw1b.train ...
```

* Notebooks are for **exploring**: tokenizers, data, small models, plots, checkpoints.
* Anything longer than ~1 hour goes to a **batch job** (`gw1b run` / `gw1b train`) so it survives your laptop going to sleep,
  and resumes automatically from its last checkpoint if it hits the time limit (`--chain N`).

Next: `01_tokenizer.ipynb`."""),
], "00 · Hello Pegasus: is everything working?")

# ---------------------------------------------------------------------------------------------
nb01 = nb([
    ("md", """Team 1 owns the tokenizer, but everyone should understand it. Here we train two small
SentencePiece models (BPE and unigram) on a text sample and compare how efficiently they encode text.

The class tokenizer is trained the same way with `vocab_size=32000` on ~2 GB of FineWeb-Edu text:
```
python -m gw1b.prepare_data sample-text --sample 10BT --out $GW1B_SCRATCH/data/tokenizer_corpus.txt --mb 2000
python -m gw1b.tokenizer train --input $GW1B_SCRATCH/data/tokenizer_corpus.txt --out $GW1B_GROUP/tokenizer/gw1b-32k --vocab 32000 --type bpe
```"""),
    ("code", SETUP),
    ("md", "## 1. A text sample\nUse real FineWeb-Edu text if the class corpus is on scratch, otherwise a small synthetic corpus."),
    ("code", '''from gw1b.prepare_data import parquet_files, iter_documents
sample_path = os.path.join(USER_DIR, "tokenizer_sample.txt")
try:
    files = parquet_files("10BT")[:1]
    with open(sample_path, "w") as f:
        for i, doc in enumerate(iter_documents(files)):
            f.write(doc.replace("\\n", " ") + "\\n")
            if i >= 20000: break
    print("using FineWeb-Edu text from", files[0])
except FileNotFoundError:
    import random
    random.seed(0)
    # a synthetic "language": 6000 pseudo-words built from syllables, Zipf-distributed like real text
    syll = [c + v for c in "bcdfghjklmnprstvwz" for v in ["a", "e", "i", "o", "u", "ai", "ou"]]
    words = ["".join(random.choices(syll, k=random.randint(1, 4))) for _ in range(6000)]
    weights = [1.0 / (i + 1) for i in range(len(words))]
    with open(sample_path, "w") as f:
        for _ in range(20000):
            f.write(" ".join(random.choices(words, weights=weights, k=random.randint(8, 30))) + ".\\n")
    print("FineWeb-Edu not found on scratch; using a synthetic corpus")
print(open(sample_path).read(300))'''),
    ("md", "## 2. Train BPE vs unigram"),
    ("code", '''from gw1b.tokenizer import train_sentencepiece, Tokenizer
toks = {}
for kind in ["bpe", "unigram"]:
    path = train_sentencepiece(sample_path, os.path.join(USER_DIR, f"tok-{kind}-4k"), vocab_size=4000,
                               model_type=kind, input_sentence_size=20000)
    toks[kind] = Tokenizer(path)
    print(kind, "vocab:", toks[kind].vocab_size, "special ids (unk,bos,eos,pad):",
          toks[kind].unk_id, toks[kind].bos_id, toks[kind].eos_id, toks[kind].pad_id)'''),
    ("md", "## 3. Compare compression (bytes per token) — the metric that decides how many tokens your 100B-token budget really buys"),
    ("code", '''text = open(sample_path).read(200_000)
for kind, tok in toks.items():
    s = tok.stats(text)
    print(f"{kind:8s} tokens={s['tokens']:>7,d}  bytes/token={s['bytes_per_token']:.2f}  words/token={s['words_per_token']:.2f}")
print()
sentence = "Pegasus trains the GW1B language model with JAX on eight A100 GPUs."
for kind, tok in toks.items():
    print(f"{kind:8s}", tok.pieces(sentence))'''),
    ("md", """## 4. Questions Team 1 can answer with this
* 16K vs 32K vs 50K vocabulary: bytes/token vs. embedding size (vocab × d_model parameters!)
* BPE vs unigram on code / math / non-English text
* `byte_fallback=True` means no `<unk>`: every byte is representable. Try `tok.pieces("日本語 ∑ 🎓")`.

Next: `02_data_pipeline.ipynb`."""),
], "01 · Tokenizers: BPE vs unigram")

# ---------------------------------------------------------------------------------------------
nb02 = nb([
    ("md", """The trainer reads **token shards**: flat `uint16` arrays with an EOS token between documents.
A training example is a contiguous window of `seq_len + 1` tokens; batches are drawn deterministically
from a seeded permutation, so a resumed job continues on exactly the batch it would have seen.

The class corpus is built once with `python -m gw1b.prepare_data tokenize ...` (a CPU batch job, see
`slurm/tokenize.sbatch`). Here we build a miniature version to see the moving parts."""),
    ("code", SETUP),
    ("md", "## 1. Tokenize documents into shards"),
    ("code", '''from gw1b.tokenizer import Tokenizer
from gw1b.prepare_data import tokenize_documents
tok_path = os.path.join(USER_DIR, "tok-bpe-4k.model")
if not os.path.exists(tok_path):
    print("run 01_tokenizer.ipynb first (it trains tok-bpe-4k) — using the class tokenizer instead")
    tok_path = os.path.join(GW1B_GROUP, "tokenizer", "gw1b-32k.model")
sample_path = os.path.join(USER_DIR, "tokenizer_sample.txt")
docs = (line.strip() for line in open(sample_path) if line.strip())
out_dir = os.path.join(USER_DIR, "data", "mini")
manifest = tokenize_documents(docs, tok_path, out_dir, val_docs=1000, workers=4, shard_size=200_000, chunk_docs=200)
{k: manifest[k] for k in ["documents_seen", "train_tokens", "val_tokens", "train_shards", "val_shards"]}'''),
    ("md", "## 2. Look inside a shard"),
    ("code", '''from gw1b.data import TokenDataset, BatchLoader
tok = Tokenizer(tok_path)
ds = TokenDataset(os.path.join(out_dir, "train"), seq_len=128)
print(f"{ds.total_tokens:,} tokens in {len(ds.files)} shard(s) -> {ds.n_windows:,} windows of 128")
w = ds.window(0)
print("window 0 ids:", w[:20], "...")
print("decoded    :", repr(tok.decode(w[:60].tolist())))
print("EOS id", tok.eos_id, "appears", int((w == tok.eos_id).sum()), "times in this window (document boundaries)")'''),
    ("md", "## 3. Deterministic, resumable batches"),
    ("code", '''loader = BatchLoader(ds, batch_size=8, seed=0, start_step=0)
b0, b1 = next(loader), next(loader); loader.close()
loader2 = BatchLoader(ds, batch_size=8, seed=0, start_step=1)     # "resume" at step 1
b1_again = next(loader2); loader2.close()
print("inputs shape", b0["inputs"].shape, "| targets are inputs shifted by one:", bool((b0["inputs"][:, 1:] == b0["targets"][:, :-1]).all()))
print("resumed batch identical:", bool((b1["inputs"] == b1_again["inputs"]).all()))
# in a multi-GPU job, each process gets its slice of the same global batch:
parts = []
for rank in range(4):
    l = BatchLoader(ds, batch_size=8, seed=0, rank=rank, world=4); parts.append(next(l)["inputs"]); l.close()
print("4 ranks concatenate to the global batch:", bool((np.concatenate(parts) == b0["inputs"]).all()))'''),
    ("md", """## 4. Where Team 1 plugs in
* `tokenize_documents(docs, ...)` accepts **any iterator of strings** — your filtered / re-mixed / curriculum-ordered corpus.
* `--filter my_module:keep_fn` applies a document filter inside the tokenization workers.
* The `manifest.json` next to the shards records the tokenizer hash and source files = the *data manifest* we release.

Next: `03_train_proxy_model.ipynb`."""),
], "02 · Data pipeline: documents → token shards → batches")

# ---------------------------------------------------------------------------------------------
nb03 = nb([
    ("md", """We train a small model **in the notebook** to see every part of the loop, then look at the same
run the way a batch job produces it. The real proxy runs (50M–350M, 1–7B tokens) are batch jobs:

```
gw1b train --gpus 1 --time 12:00:00 --config configs/50m.yaml --set run.name=team2-baseline
```"""),
    ("code", SETUP),
    ("md", "## 1. A configuration\nEverything about a run is one YAML file (+ command-line overrides). Let's start from `tiny_debug.yaml` and point it at the mini dataset from notebook 02."),
    ("code", '''from gw1b.config import load_config, config_summary
mini = os.path.join(USER_DIR, "data", "mini")
cfg = load_config(os.path.join(GW1B_HOME, "configs", "tiny_debug.yaml"), [
    f"data.train_dir={mini}/train", f"data.val_dir={mini}/val", f"data.tokenizer={USER_DIR}/tok-bpe-4k.model",
    "model.vocab_size=4000", "run.name=nb03-tiny", f"run.out_dir={USER_DIR}/runs",
    "run.total_steps=300", "run.log_every=25", "run.eval_every=100", "run.ckpt_every=150",
    "model.dtype=auto",   # bfloat16 on A100/L40S/GH200, float32 on V100 (no bf16 tensor cores) and CPU
])
print(config_summary(cfg))'''),
    ("md", "## 2. Build the model (sharded over all GPUs of the node) and look at it"),
    ("code", '''from gw1b.sharding import make_mesh, create_model
from gw1b.model import count_params
mesh = make_mesh()
model = create_model(cfg, mesh)
print(f"{count_params(model)/1e6:.2f}M parameters, dtype = {model.cfg.dtype}, attention impl = {model.attn_impl}")
print("mesh:", mesh)
k = model.blocks[0].mlp.up_proj.kernel[...]
print("one weight matrix:", k.shape, k.dtype, "sharding:", k.sharding.spec)'''),
    ("md", "## 3. Train\n`gw1b.train.train(cfg)` is exactly what a batch job runs. It logs to `metrics.jsonl` + TensorBoard, checkpoints with Orbax, and resumes if you call it again."),
    ("code", '''from gw1b.train import train
summary = train(cfg)              # ~1-2 minutes on CPU, seconds on a GPU
{k: summary[k] for k in ["step", "tokens", "train_loss", "val_loss", "gpu_hours", "usd_equivalent", "flops"]}'''),
    ("md", "## 4. Plot the loss curve from metrics.jsonl (this is what you put in the paper)"),
    ("code", '''import pandas as pd, matplotlib.pyplot as plt
run_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
m = pd.DataFrame([json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
m.dropna(subset=["train/loss"]).plot(x="step", y="train/loss", ax=ax[0], legend=False, title="train loss")
if "val/loss" in m: m.dropna(subset=["val/loss"]).plot(x="tokens_seen", y="val/loss", ax=ax[1], marker="o", legend=False, title="val loss vs tokens")
plt.tight_layout(); plt.show()
print("TensorBoard: run  %tensorboard --logdir", os.path.join(run_dir, "tb"), " (port 6006 is in your tunnel)")'''),
    ("md", "## 5. Generate text from the checkpoint"),
    ("code", '''from gw1b.evaluate import load_run
from gw1b.generate import generate
from gw1b.tokenizer import Tokenizer
model, cfg2, step = load_run(run_dir)
tok = Tokenizer(cfg2.data.tokenizer)
for t in generate(model, tok, ["the", "we can"], max_new_tokens=20, temperature=0.8, top_k=20, seed=0):
    print(repr(t))'''),
    ("md", """## 6. Try an ablation (Team 2 / Team 3 style)
Change one thing, keep everything else fixed, compare loss at equal tokens **and** equal GPU-hours:
```python
cfg_b = load_config(..., ["model.n_kv_heads=4", "run.name=nb03-mha"])      # GQA -> MHA
cfg_c = load_config(..., ["model.activation=gelu", "model.d_ff=768", "run.name=nb03-gelu"])
cfg_d = load_config(..., ["optim.schedule=wsd", "optim.lr=2e-3", "run.name=nb03-wsd"])
```
As a batch job the same thing is  `gw1b train --config configs/50m.yaml --set model.n_kv_heads=8 --set run.name=mha`.

Next: `04_evaluate_and_export.ipynb`."""),
], "03 · Train a proxy model end-to-end")

# ---------------------------------------------------------------------------------------------
nb04 = nb([
    ("md", """Three ways to evaluate a checkpoint, and how to release it:

1. **Validation loss / perplexity** on held-out shards (cheap; the number for scaling laws)
2. **lm-evaluation-harness** (HellaSwag, ARC, PIQA, Winogrande, MMLU, GSM8K, ...) directly on the JAX model
3. **Export to Hugging Face format** and verify that `transformers` reproduces the JAX logits — that is the artifact we publish."""),
    ("code", SETUP),
    ("code", '''from gw1b.evaluate import load_run, eval_loss
run_dir = os.path.join(USER_DIR, "runs", "nb03-tiny")        # from notebook 03 (or any run directory)
model, cfg, step = load_run(run_dir, shard_params=False)
res = eval_loss(model, cfg.data.val_dir, cfg.data.seq_len, batch_size=8, n_batches=10)
print(f"step {step}: val loss {res['loss']:.3f}, perplexity {res['ppl']:.1f} on {res['tokens_evaluated']:,} tokens")'''),
    ("md", """## 2. lm-evaluation-harness
Works on any checkpoint, no conversion. Benchmarks are downloaded from the Hugging Face Hub the first time
(do that once on a login node so the compute nodes find them in `$HF_HOME`)."""),
    ("code", '''from gw1b.lm_eval_adapter import GW1BLM
from gw1b.tokenizer import Tokenizer
lm = GW1BLM(model, Tokenizer(cfg.data.tokenizer), batch_size=8)
try:
    import lm_eval
    out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=50, log_samples=False)
    results = out["results"] if isinstance(out, dict) else out.results
    print({k: v for k, v in results["hellaswag"].items() if isinstance(v, float)})
except Exception as e:
    print("lm-eval could not run here (offline or task cache missing):", str(e)[:200])
    print("batch version:  gw1b run --gpus 1 -m gw1b.lm_eval_adapter --run", run_dir, "--tasks hellaswag,arc_easy,piqa --limit 500")'''),
    ("md", "## 3. Export to Hugging Face format and verify"),
    ("code", '''from gw1b.export_hf import export, verify
out_dir = os.path.join(USER_DIR, "release", "nb03-tiny-hf")
export(model, out_dir, tokenizer_path=cfg.data.tokenizer, dtype="float32")
diff = verify(model, out_dir)          # loads with transformers on CPU and compares log-probs
print(sorted(os.listdir(out_dir)))'''),
    ("md", "## 4. The numbers every experiment reports"),
    ("code", '''import pandas as pd
s = json.load(open(os.path.join(run_dir, "summary.json")))
pd.DataFrame([{ "run": s["run"], "params (M)": s["n_params"]/1e6, "tokens (M)": s["tokens"]/1e6, "FLOPs": f"{s['flops']:.2e}",
                "GPU-hours": round(s["gpu_hours"], 3), "USD": round(s["usd_equivalent"], 2),
                "val loss": round(res["loss"], 3), "val ppl": round(res["ppl"], 1)}]).T'''),
    ("md", """## 5. Releasing GW-1B (end of semester)
```
python -m gw1b.export_hf --run $GW1B_SCRATCH/checkpoints/gw1b-base --out $GW1B_GROUP/release/GW-1B-Base --verify
huggingface-cli upload gwu-gw1b/GW-1B-Base $GW1B_GROUP/release/GW-1B-Base      # from a login node
```
plus the tokenizer, the data manifest (`manifest.json`), configs, `metrics.jsonl` and the model card."""),
], "04 · Evaluate, benchmark and export")

for name, n in [("00_hello_pegasus", nb00), ("01_tokenizer", nb01), ("02_data_pipeline", nb02),
                ("03_train_proxy_model", nb03), ("04_evaluate_and_export", nb04)]:
    nbf.write(n, os.path.join(HERE, name + ".ipynb"))
    print("wrote", name + ".ipynb")
