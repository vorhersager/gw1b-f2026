# Architecture and design decisions

## The one-sentence design

*One read-only directory on group storage (`gw1b-f2026`) + one container image built from a lock file;
students `source activate.sh` and use `gw1b …`, which submits Slurm jobs that run every command
through `bin/gw1b-exec` inside that image.*

```
 laptop ──ssh──▶ login node                          GPU node (Slurm job)
                 ├─ activate.sh  (PATH, env vars)     ├─ gw1b-exec ─▶ apptainer exec --nv gw1b.sif python …
                 ├─ gw1b jupyter ─▶ sbatch jupyter.sbatch ─▶ JupyterLab (Colab-compatible flags)
                 ├─ gw1b train   ─▶ sbatch run.sbatch / train_multinode.sbatch ─▶ python -m gw1b.train
                 └─ gw1b status / cancel / doctor / kernel / budget
 storage:  $GW1B_GROUP (persistent: env, code, tokenizer, releases)   $GW1B_SCRATCH (Lustre: data, checkpoints, caches)
```

## Decisions and why

| decision | alternatives considered | why this one |
|---|---|---|
| **Apptainer image built from a pip lock** (`env/gw1b.def`) | conda env per student; module system; NVIDIA JAX-Toolbox nightly containers | Bit-identical for all 24 students and unaffected by OS/module changes on the cluster; the lock file is the reproducibility recipe we publish. JAX-Toolbox is a moving nightly target. A shared `uv` venv from the same lock is the no-root fallback. |
| **`jax[cuda12]` pip wheels, not CUDA 13, not the cluster's CUDA module** | `jax[cuda13]`, `cuda12-local` | CUDA 13 dropped Volta — the V100 nodes are most of the cluster. Pip wheels carry their own CUDA libraries, so only the driver (≥ 525) matters. |
| **Flax NNX model written from scratch (~250 lines)** rather than MaxText as the class trainer | MaxText, Levanter, EasyLM | Students must be able to read and change every line (Team 2 ablations, Team 5 probes). The model uses HF Llama conventions so the release is a standard `LlamaForCausalLM`. MaxText remains an option for Team 3's production run if they want to compare throughput — it fits in the same container (`pip install` in a writable overlay). |
| **FSDP via one mesh axis + GSPMD** (`sharding.py`) | pure data parallel; tensor parallel | Simplest thing that fits 1.15B on 16 GB V100s and scales to 16 A100s; XLA does the collectives; zero code changes between 1 and 16 GPUs. |
| **One process per GPU for multi-node** | one process per node | What `jax.distributed.initialize()` auto-detects under Slurm (`SLURM_LOCALID` = GPU index). Single-node jobs use one process for all GPUs (no `srun`). |
| **uint16 token shards + memmap loader** (`data.py`) | Grain / tf.data / HF datasets streaming | Trivial format (llm.c/nanoGPT style), no dependency on the compute node reaching the internet, deterministic and resumable by construction, fast on Lustre. Grain is installed for teams who want its pipelines. |
| **Orbax `CheckpointManager` with `keep_period`** | Flax serialization, manual pickles | Async, sharded-aware saves; `ckpt_keep_every` gives Team 5 its checkpoint ladder for free. |
| **JupyterLab launched by a Slurm job + ssh tunnel** (`jupyter.sbatch`) | Open OnDemand only; JupyterHub | Works on any Slurm cluster with nothing installed centrally; the same server serves Colab (local-runtime flags). `gw1b kernel` adds an OnDemand/VS Code kernel if those exist. |
| **Everything through `gw1b-exec`** | separate scripts for container vs venv | One place to set caches, `PYTHONPATH`, XLA flags and choose the runtime; sbatch templates and kernelspecs stay tiny. |
| **`gw1b.env` as the single config** | hard-coded paths | The only file that changes between clusters/semesters; `discover_cluster.sh` tells the instructor what to put in it. |

## Data flow for the semester

```
HF Hub ──(login node)──▶ $HF_HOME/fineweb-edu/10BT/*.parquet ──sample-text──▶ tokenizer corpus ──▶ gw1b-32k.model (group dir)
                                                              └──tokenize (CPU job, 40 cores)──▶ $GW1B_SCRATCH/data/fineweb-edu-10B/{train,val}/*.bin + manifest.json
training job ──▶ $GW1B_SCRATCH/users|teams/…/runs/<name>/{config.yaml, metrics.jsonl, tb/, checkpoints/<step>/, summary.json}
release ──export_hf──▶ $GW1B_GROUP/release/GW-1B-Base/{model.safetensors, config.json, tokenizer.json} ──▶ Hugging Face Hub
```

## Numerics

* Master weights f32, matmuls in bf16 (`model.dtype: auto` = bf16 on GPUs that have it, f32 on V100/CPU), logits
  and loss in f32, AdamW moments f32.
* Attention: cuDNN flash attention on Ampere+ (`attn_implementation=auto`), XLA elsewhere; identical math. On a
  V100 the XLA path computes the scores in f32 (XLA:GPU has no bf16 dot algorithm before Ampere).
* Memory: `run.batch_size` is the global batch; the trainer picks the per-GPU micro-batch from the GPU's memory
  (`optim.grad_accum: auto`, `budget.choose_micro_batch`) and accumulates the rest. The loss is computed `run.loss_chunk`
  positions at a time (`model.loss`), so the f32 `[batch, T, vocab]` logits never exist in full (1 GB per sequence
  otherwise); the XLA attention path rematerialises its `[N, T, T]` scores in the backward pass; `run.remat=true` adds
  block-level gradient checkpointing (the setting for 200m+ on 16 GB V100s).
* RoPE uses the Hugging Face "rotate-half" layout, so exported weights need no permutation; `export_hf.verify`
  checks log-prob agreement with `transformers` on CPU (the tests show a max difference of 0.0000 in f32).
* Weight decay is applied to matrices only (norm scales / biases excluded), init N(0, 0.02) with 1/√(2L) scaling
  on residual projections — the standard GPT-2/Llama recipe, all overridable in YAML.

## Upgrading

1. Edit `env/requirements.in` → `env/lock.sh` (needs `uv`) → review the diff of `env/requirements.txt`.
2. Bump `GW1B_ENV_VERSION` in `gw1b.env` and in `env/gw1b.def`.
3. `env/build_sif.sh` builds `gw1b-<version>.sif` next to the old one and moves the `gw1b.sif` symlink; running jobs
   keep the old image open. Roll back by moving the symlink.
4. `python -m pytest tests` inside the new image, then `gw1b doctor --gpu`.
