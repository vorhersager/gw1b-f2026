# Team playbook — how each team uses the toolchain

Every experiment is a run directory with `config.yaml`, `metrics.jsonl` and `summary.json`
(loss/benchmarks, tokens, FLOPs, GPU-hours, $). Name runs `team<N>-<hypothesis>-<variant>` and keep
them in `$GW1B_SCRATCH/teams/team<N>/runs` (`--set run.out_dir=…`).

## Team 1 — Data, tokenization & curriculum

* **Tokenizer experiments**: `python -m gw1b.tokenizer train --vocab {16000,32000,50000} --type {bpe,unigram}`;
  compare `Tokenizer.stats()` (bytes/token) on held-out text, then train identical 50M models on shards made with each
  tokenizer (`prepare_data tokenize --tokenizer …`) and compare loss **per byte**, not per token.
* **Quality filtering / mixtures**: write `keep(text) -> bool` in your module and tokenize with
  `--filter team1.filters:keep`; or build your own document iterator (parquet, HF datasets, SmolLM-Corpus subsets)
  and call `gw1b.prepare_data.tokenize_documents(docs, …)`. Each corpus → its own shard directory + `manifest.json`.
* **Curriculum**: order documents in the iterator (the loader shuffles windows within an epoch; for strict curricula set
  `data.shuffle=false`, or write stage-wise shard directories and chain runs with `--set data.train_dir=stage2`).
* Deliverable: `$GW1B_SCRATCH/data/gw-100B/` shards + manifest + the final `gw1b-32k.model` in `$GW1B_GROUP/tokenizer/`.

## Team 2 — Architecture & scaling laws

* Ladder configs `configs/{50m,100m,200m,350m}.yaml`; every knob is a YAML/CLI override:
  `model.n_kv_heads` (MHA↔GQA↔MQA), `model.activation=gelu|swiglu`, `model.norm=layernorm|rmsnorm`,
  `model.pos=learned|rope`, `model.tie_embeddings`, `model.d_model/n_layers/d_ff` (depth/width), `data.seq_len`, `model.vocab_size`.
  `gw1b python -m gw1b.train --config … --dry-run` prints the exact parameter count and cost before you submit.
* Iso-FLOP comparisons: hold `total_tokens × n_params` fixed; `summary.json["flops"]` is what you plot.
* Fit scaling curves from `metrics.jsonl` (`val/loss` vs `tokens_seen`) across the ladder; the notebook `03` shows the plotting.
* Deliverable: the frozen `configs/gw1b_1p15b.yaml` (or its successor) with an ablation table.

## Team 3 — Optimization & pretraining

* Optimizer/schedule knobs: `optim.lr`, `optim.warmup_steps`, `optim.schedule=cosine|wsd|linear|constant`,
  `optim.min_lr_ratio`, `optim.decay_fraction` (WSD), `optim.beta1/beta2/eps`, `optim.weight_decay`, `optim.grad_clip`,
  `run.batch_size` (the global batch; `optim.grad_accum` is chosen automatically from the GPU memory, `run.remat` and
  `run.loss_chunk` trade compute for memory), `data.seq_len`, `model.dtype`. New optimizers: one line in
  `train.py:build_optimizer` (anything in `optax`).
* Throughput work: `run.profile=true` writes a JAX profiler trace (`<run>/profile`, open in TensorBoard); watch `perf/mfu`.
  Compare `attn_implementation`, batch sizes, `XLA_FLAGS`, single vs multi-node.
* The production run: `gw1b train --gpus 8 --gpu-type a100 --time 1-00:00:00 --chain 4 --config configs/gw1b_1p15b.yaml --set run.out_dir=$GW1B_SCRATCH/checkpoints`.
  Pilot first (`--set run.total_tokens=5e9`), then the real run; it resumes across chained jobs. Optional: MaxText
  comparison in the same container.
* Deliverable: the trainer recipe (final YAML), the base checkpoint, `metrics.jsonl` of the run.

## Team 4 — Post-training & evaluation

* Benchmarks on any checkpoint without conversion:
  `gw1b run --gpus 1 -m gw1b.lm_eval_adapter --run <run> --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu --limit 1000`
  (download task data once on a login node; `gsm8k` uses generation — slower). Results → `<run>/lm_eval/*.json`.
* SFT: build instruction shards (`prompt + response + EOS`) with `tokenize_documents`, then continue training from the
  base checkpoint: copy `checkpoints/<step>` into a new run dir and `gw1b train --config configs/gw1b_1p15b.yaml --set run.name=sft --set data.train_dir=<sft shards> --set optim.lr=2e-5 --set optim.warmup_steps=100 --set run.total_tokens=…`
  (loss on all tokens; prompt-masking is a small change to `cross_entropy(mask=…)` in `train.py`). Preference optimization
  (DPO) fits in the same loop as a second loss term — a good ICML-paper-sized contribution.
* Held-out GW evaluation set: plain text + `python -m gw1b.evaluate --data <shards of your set>` for perplexity, or an
  lm-eval task YAML for accuracy; contamination checks by n-gram overlap against the token shards.
* Release: `python -m gw1b.export_hf --run <run> --out $GW1B_GROUP/release/GW-1B-Base --verify` → `huggingface-cli upload`.
  The `verify` step proves the published weights reproduce the JAX logits.

## Team 5 — Interpretability, safety & robustness

* Checkpoint ladder: the production config keeps every `run.ckpt_keep_every` steps forever (5000 steps ≈ 5B tokens);
  set it to what you need before the run starts. `gw1b.evaluate.load_run(run_dir, step=…)` loads any of them.
* Hidden states: call the blocks yourself —
  ```python
  x = model.embed(tokens); cos, sin = rope_tables(jnp.arange(T), cfg.head_dim, cfg.rope_theta, x.dtype)
  for i, blk in enumerate(model.blocks): x, _ = blk(x, cos, sin, impl=model.attn_impl); hs.append(x)
  ```
  then train linear probes (scikit-learn is installed) per layer and per checkpoint.
* Calibration / factuality classifiers: `lm_eval_adapter.GW1BLM.loglikelihood(...)` gives per-continuation log-probs.

## Everyone

* `gw1b budget` before every submission; `summary.json` in every table; `metrics.jsonl` → plots in the paper.
* Push configs and analysis code to the class GitHub org; never push checkpoints or data.
