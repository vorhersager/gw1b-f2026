# Primer: the GPUs on Pegasus and the tools around them

A plain-language map of the hardware and software GW1B runs on. Read it once; after that the
[student guide](STUDENT_GUIDE.md) and [cluster facts](CLUSTER_FACTS.md) will make sense without
further explanation. Each section ends with what it means for this project.

Contents: [1. What a GPU does for us](#1-what-a-gpu-does-for-us) · [2. The GPUs on Pegasus](#2-the-gpus-on-pegasus) ·
[3. Memory, precision, interconnects](#3-memory-precision-and-interconnects--the-three-things-that-decide-what-fits-and-how-fast) ·
[4. The cluster and how work gets onto it](#4-the-cluster-and-how-work-gets-onto-it) · [5. The software environment](#5-the-software-environment) ·
[6. The JAX stack](#6-the-jax-stack-what-each-library-does) · [7. Data, metrics, evaluation](#7-data-metrics-and-evaluation) ·
[8. How it all fits together](#8-how-it-all-fits-together) · [Glossary](#glossary)

## 1. What a GPU does for us

Training a language model is, computationally, one thing repeated billions of times: multiplying
large matrices. A CPU core does a handful of multiply-adds per clock tick; a GPU does tens of
thousands at once, and its *tensor cores* are circuits built for nothing but matrix multiplication
at reduced precision. That is why a 1-billion-parameter model that would take a CPU years trains on
eight A100s in about two days.

Three numbers describe a GPU for our purposes:

* **Throughput** — how many trillion floating-point operations per second (TFLOP/s) its tensor cores
  deliver at the precision we train in (bf16, see §3). This sets how fast a step runs.
* **Memory** — how many gigabytes sit next to the chip (HBM or GDDR). This sets how large a model
  (plus its optimizer state and activations) fits without tricks.
* **Interconnect** — how fast GPUs exchange data with each other (NVLink, PCIe) and with other nodes
  (InfiniBand). This sets how well many GPUs work on one model together.

Training cost is predictable from first principles: a forward+backward pass costs about
**6 × parameters × tokens** floating-point operations. For GW-1B, 6 × 1.15e9 × 20e9 ≈ 1.4e20 FLOPs.
Divide by the GPUs' throughput and by the fraction of peak you actually reach (the **MFU**, model
FLOP utilization — 30–40 % is typical) and you have the wall-clock time. `gw1b budget` does exactly
this arithmetic; its table is in [CLUSTER_FACTS.md](CLUSTER_FACTS.md).

## 2. The GPUs on Pegasus

| GPU | generation (year) | memory | bf16 tensor TFLOP/s (dense) | fp32 TFLOP/s | GPU-to-GPU link | on Pegasus | `--gpu-type` |
|---|---|---|---|---|---|---|---|
| **V100 16 GB** | Volta (2017) | 16 GB HBM2, 0.9 TB/s | — (fp16 125; **no bf16**) | 15.7 | NVLink on the 4-GPU nodes | 16 nodes × 2, 22 nodes × 4 | `v100` (default) |
| **A100 80 GB PCIe** | Ampere (2020) | 80 GB HBM2e, 1.9 TB/s | 312 | 19.5 | NVLink bridges **in pairs**, PCIe otherwise | 2 nodes × 8 (gpu050, gpu051), 1 TB RAM | `a100` |
| **L40S 48 GB** | Ada Lovelace (2023) | 48 GB GDDR6, 0.86 TB/s | 362 | 91.6 | PCIe only | a few nodes | `l40s` |
| **GH200 Grace Hopper** | Hopper (2023) + Grace ARM CPU | 96 GB HBM3, 4 TB/s (+480 GB CPU memory over NVLink-C2C) | 989 | 67 | one GPU per node; InfiniBand between nodes | 8 nodes × 1 (`superChip`) | `gh200` |
| **RTX PRO 6000 Blackwell 96 GB** | Blackwell (2025) | 96 GB GDDR7, 1.6 TB/s | ~500 (spec: 1 PFLOP/s with sparsity) | 120 | PCIe 5 only | 2 nodes × 4 + 4 nodes × 2, arriving fall 2026 | `rtx6000` (name TBD) |

How to read the table:

* **V100** is the oldest and the most numerous (~120 GPUs). It is fine for notebooks and for the 50M–100M
  proxy models, but it has no bf16 tensor cores: training runs in `float32` there (`model.dtype: auto`
  picks it), which is roughly eight times slower per GPU than bf16 on an A100. It also only supports CUDA 12, which
  is why the whole class uses the CUDA 12 build of JAX.
* **A100** is the workhorse: bf16, 80 GB, flash attention. There are only 16 on the whole campus and they
  run around the clock. The eight on a node are NVLinked in pairs (0–3, 1–2, 4–7, 5–6); the rest of the
  traffic goes over PCIe, which is slower but adequate for a 1B model (§3).
* **L40S** is a cheaper data-center card: bf16 at A100-class throughput but less memory and bandwidth,
  no NVLink. Good for 100M–350M experiments.
* **GH200** is the fastest chip here (≈ 3 A100s), one per node, and the least busy according to HPC.
  Its catch: the CPU is an ARM (aarch64) Grace, so the x86 container cannot run there — the environment
  has a separate arm64 image for it (`gw1b-admin build arm64`).
* **RTX PRO 6000 Blackwell** nodes are coming online during the semester: more memory than an A100 and
  probably faster, but no NVLink and their own network fabric. A 4-GPU node is a serious option for the
  production run once they exist; measure first with `gw1b doctor --gpu --gpu-type rtx6000`.

For GW1B: students default to a V100; the 200M–350M ladder runs on A100/L40S; the 1.15B production run
targets 8 A100s, with 4 A100s on one socket (`--one-socket`) or one GH200 as fallbacks.

## 3. Memory, precision and interconnects — the three things that decide what fits and how fast

**Precision.** Numbers in a network can be stored with 32 bits (`float32`, the safe default), 16 bits
(`float16` or `bfloat16`), or fewer. Tensor cores are 8–16× faster at 16 bits than at 32. `bfloat16`
keeps float32's exponent range with fewer mantissa bits, so training in bf16 "just works" where fp16
needs loss scaling to avoid overflow — which is why modern LLM training uses bf16 for the matmuls
and keeps a float32 master copy of the weights for the optimizer update ("mixed precision"). V100s
predate bf16; everything newer has it. (`tf32` is a float32-in/float32-out mode that rounds internally to
10 mantissa bits — Ampere and later use it automatically for fp32 matmuls; fp8 and fp4 are inference
formats on Hopper/Blackwell that we do not use.)

**Memory.** With AdamW in mixed precision each parameter costs about **16 bytes** during training: 2
(bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master weight) + 8 (two fp32 Adam moments). A 1.15B model is
therefore ~18 GB before a single activation is stored; activations add memory proportional to batch ×
sequence length × depth, which is why training needs far more memory than inference (2 bytes per
parameter). When activations are what does not fit, the standard trick is to recompute them during the
backward pass instead of storing them (`jax.checkpoint`, "rematerialization") — not needed for GW-1B on
80 GB GPUs at the default micro-batch, but a one-line change in `gw1b/model.py` if a team needs it.
Fitting a model that does not fit on one GPU is what **FSDP** (fully sharded data parallel) is for: every
GPU holds a 1/N slice of the parameters and optimizer state and gathers the full layer only while it is
needed. The GW1B trainer does this over any number of GPUs with one mesh axis.

**Interconnect.** FSDP and plain data parallelism exchange gradients (and, for FSDP, parameters) every
step. How fast that is depends on the link: NVLink ≈ 300–600 GB/s between a pair, PCIe 4 ≈ 25 GB/s per
direction, InfiniBand EDR between nodes ≈ 12 GB/s. For a 1.15B model at one million tokens per step the
exchange is a few GB per step against ~8 s of compute, so even PCIe keeps the communication under 10 %
of the step and most of it overlaps with compute. Tensor parallelism (splitting single matmuls across
GPUs) would not tolerate PCIe; we do not use it. **NCCL** is NVIDIA's library that implements these
collective operations (all-reduce, all-gather, reduce-scatter); JAX calls it under the hood, and on
Pegasus it runs over the EDR InfiniBand fabric (`ib0`/`mlx5_0`) between nodes.

## 4. The cluster and how work gets onto it

**Pegasus** is GWU's shared cluster: a few *login nodes* you ssh into, hundreds of *compute nodes* you
never log into directly, shared file systems, and a scheduler that hands compute nodes to jobs.

| piece | what it is | what it means for you |
|---|---|---|
| **ssh + key + 2FA** | you log in with an SSH key (sent with the HPC Access Request form) and a 6-digit code from an authenticator app; off campus, the GW VPN first | `ssh <netid>@pegasus.arc.gwu.edu`; the first login uses the one-time code HPC emails, then you set up 2FA at once |
| **login node** | a small shared machine for editing, submitting jobs, light scripts | never train here; `gw1b python` is for dry runs and `gw1b budget` only |
| **compute node** | a machine with the GPUs/CPUs your job asked for, yours for the job's duration | everything heavy runs here, via Slurm |
| **home** `~` | your private, small, backed-up directory | code, notebooks, settings — not datasets |
| **group storage** `/SEAS/groups/gw1b` (or the Research NAS path HPC assigns) | persistent, shared by the class | the environment, tokenizer, released checkpoints |
| **scratch** `/scratch/gw1b-class` | fast GPFS file system, 2 PB shared, not backed up, purged by age with notice | datasets, caches, checkpoints, your runs (`users/<netid>`, `teams/teamN`) |
| **Globus** | a web service for moving large data between Pegasus, GW Box/Drive and your laptop | the 100B-token dataset, release artifacts |
| **module** | `module load apptainer` makes optional software visible on a node | the environment loads what it needs; you rarely type it |

**Slurm** is the scheduler. You describe a job (how many GPUs of which kind, cores, memory, how long)
and Slurm queues it until that much hardware is free, runs it, and takes the node back when the time
limit ends. Vocabulary you will see:

* **partition** — a named group of nodes with its own limits. Pegasus has `gpu` (all GPU nodes, 7-day
  limit), `cpu` (14 days), `nano` (30-minute tests), `superChip` (the GH200 nodes).
* **GRES** — "generic resource", Slurm's name for GPUs. `--gres=gpu:a100:8` means "8 GPUs of type
  a100". The type names are what `sinfo -o "%N %G"` prints; the `gw1b` command fills them in for you
  from `--gpu-type`.
* **job, job id, sbatch/srun/squeue/scancel/sinfo** — submit a script, run a command interactively, list
  the queue, cancel, list partitions. `gw1b run/shell/status/cancel` wrap these.
* **wall time** — the time limit you ask for. Shorter jobs start sooner ("backfill"); the GW1B trainer
  checkpoints every 500 steps and `--chain N` resubmits automatically, so ask for a day at a time.
* **fair-share** — the priority rule: the more you have run recently, the lower your next job's
  priority. No per-user caps, no QOS, no preemption on Pegasus: once a job runs, nothing stops it.
* **reservation** — hardware set aside for a group at a fixed time. HPC does single-day ones (we have
  asked about the Friday labs); multi-day reservations need a proposal.

**Open OnDemand** (https://ood.arc.gwu.edu) is Pegasus's web portal: a *Jupyter Notebook Pegasus* app
that starts a notebook server on a compute node from a form, without ssh. After `gw1b kernel` the
**GW1B (JAX)** kernel appears in its kernel menu.

## 5. The software environment

A Python environment is a particular set of library versions. 27 students each installing their own
would mean 27 different sets of bugs, so the class uses one frozen environment, built once and shared.

| tool | what it is | role here |
|---|---|---|
| **container image (Apptainer, `.sif`)** | one file holding a whole Linux userland: Python 3.12, JAX with its CUDA libraries, every pinned package. Apptainer is the HPC flavor of Docker: no root, runs as you, sees your files, `--nv` passes the GPU through | `gw1b-exec` starts every command inside it; the `.sif` lives in the group directory |
| **Docker + GitHub Actions + GHCR** | Pegasus cannot *build* images (no fakeroot), so GitHub builds `env/Dockerfile` in the cloud (the `build-image` workflow) and publishes it to the GitHub Container Registry; Pegasus only *pulls* it | instructor runs the workflow once per environment version; `gw1b-admin build` pulls |
| **venv / uv** | the fallback: the same pinned packages installed into a shared directory with the fast installer `uv`, no container | used only if Apptainer were unavailable |
| **lock file** `env/requirements.txt` | exact versions of every package (JAX 0.10.2, Flax 0.12.8, …) | the single source of truth for "the environment"; bump `GW1B_ENV_VERSION` when it changes |
| **NVIDIA driver / CUDA** | the driver is the kernel component on each node (≥ 570 on Pegasus); CUDA libraries ship inside our image (CUDA 12 build, because V100 cannot run CUDA 13) | nothing to install; `gw1b doctor --gpu` checks the pairing |
| **`gw1b.env`** | one shell file with every cluster-specific value: paths, partition, GPU type names, defaults | the instructor edits it; everything else reads it |
| **`gw1b` CLI** | `setup`, `jupyter`, `run`, `train`, `shell`, `status`, `cancel`, `doctor`, `kernel`, `budget` | the student's whole interface to Slurm and the environment |
| **`gw1b-admin`** | `init → discover → build → test → data → students` | the instructor's one-command setup |
| **JupyterLab / Colab / ssh tunnel** | `gw1b jupyter` starts JupyterLab on a GPU node; an ssh tunnel (`ssh -N -L …`) makes its port appear on your laptop, where the browser — or Google Colab's *Connect to a local runtime* — talks to it | the everyday way to work; `laptop/gw1b-connect.sh` does it in one command |

## 6. The JAX stack: what each library does

JAX is a numerical library with four superpowers: `jit` (compile a Python function to fast GPU code
through XLA), `grad` (automatic differentiation — the backpropagation of Lecture 10), `vmap` (vectorize
over a batch), and sharding (split arrays over many devices). Everything in the model and trainer is
plain JAX arrays and functions; the libraries below add structure on top.

| library | purpose | where you meet it |
|---|---|---|
| **JAX** (`jax`, `jax.numpy`) | arrays, autodiff, compilation, multi-device execution | every file in `gw1b/` |
| **XLA** | the compiler behind `jit`: fuses operations, chooses GPU kernels, runs collectives through NCCL | invisible; `JAX_COMPILATION_CACHE_DIR` stores its output so re-runs start fast |
| **Flax NNX** | the neural-network layer on JAX: modules (`nnx.Linear`, `nnx.Module`), parameters as objects, sharding annotations | `gw1b/model.py` — the Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU) |
| **Optax** | optimizers and learning-rate schedules as composable transforms: AdamW, Lion, clipping, warmup-cosine/WSD, gradient accumulation | `gw1b/train.py` `build_optimizer` and `lr_schedule` (Lecture 11) |
| **Orbax** | checkpointing: writes/reads sharded model + optimizer state, keeps the last N, auto-resume | every 500 steps; what makes chained 1-day jobs one continuous run |
| **Grain** | Google's deterministic, resumable data-loading library (pinned for teams that want to experiment with it) | optional; the default loader is `gw1b/data.py` (memory-mapped uint16 token shards) |
| **SentencePiece** | trains and runs the tokenizer (BPE or unigram) that turns text into the 32,000 token ids the model sees | `gw1b/tokenizer.py`, `$GW1B_GROUP/tokenizer/gw1b-32k.model`, notebook 01 |
| **Hugging Face `datasets`** | downloads and reads FineWeb-Edu (parquet files) | `gw1b/prepare_data.py`, the `gw1b-admin data` jobs |
| **`transformers` + `safetensors`** | the Hugging Face model format the world can load; our exporter writes a `LlamaForCausalLM` and verifies it bit-for-bit | `gw1b/export_hf.py`, the public release |
| **lm-evaluation-harness** (`lm_eval`) | the standard benchmark runner (HellaSwag, ARC, PIQA, Winogrande, MMLU, GSM8K…) | `gw1b/lm_eval_adapter.py` lets it drive our JAX model |
| **CPU-only PyTorch** | present only because `lm_eval` and `transformers` import it | never used for compute |
| **TensorBoard / Weights & Biases** | dashboards of loss, learning rate, tokens/s, MFU; `metrics.jsonl` is the raw log | TensorBoard needs no account (port 6006 is in the tunnel); W&B is opt-in |
| **NCCL** | NVIDIA's collective-communication library (all-reduce etc.) used by JAX for multi-GPU and multi-node jobs | automatic; `slurm/train_multinode.sbatch` points it at InfiniBand |
| **pytest** | the CPU test suite (`tests/`) that runs in GitHub Actions on every push | `python -m pytest tests` before you change the trainer |

Why JAX for this class: `grad` and `jit` are the two ideas from Lectures 10–11 made executable, the
sharding model makes FSDP a handful of lines instead of a framework, and the same code runs on a laptop
CPU (the tests), a V100, eight A100s or a GH200 without change.

## 7. Data, metrics and evaluation

* **FineWeb-Edu** — a filtered web corpus released by Hugging Face (ODC-By license) with ready-made
  10B-, 100B- and 350B-token samples. We tokenize the 10B sample into **token shards**: flat binary
  files of `uint16` token ids (2 bytes per token; 20 GB for 10B tokens) that the loader memory-maps and
  slices into 2048-token windows. `manifest.json` records exactly which files went in — the data card
  of the release.
* **Loss / perplexity** — the next-token cross-entropy in nats per token, and `exp(loss)`. The number
  every experiment reports; `gw1b.evaluate` computes it on the held-out split for any checkpoint.
* **Tokens/s, MFU, GPU-hours, $** — logged every `log_every` steps so every run states what it cost
  (`run.gpu_hour_price_usd`, default $2, turns GPU-hours into the "$10,000 budget" framing).
* **Benchmarks** — `lm_eval` tasks on exported checkpoints; Team 4 decides which ones discriminate at
  sub-1B scale and checks contamination against the token shards.

## 8. How it all fits together

```
laptop ──ssh (key + 2FA, VPN off campus)──▶ login node ──sbatch/srun──▶ compute node (V100 / A100 / L40S / GH200)
                                              │  gw1b jupyter / run / train               │
                                              │  reads gw1b.env, picks partition + GRES   │  apptainer exec --nv gw1b.sif …
                                              │                                           │     └─ JAX + Flax + Optax + Orbax
   JupyterLab / Colab ◀──ssh tunnel──────────┘                                           │        reads token shards on /scratch
   (or the Open OnDemand Jupyter app)                                                     │        writes checkpoints on /scratch
                                                                                          └─ NCCL over InfiniBand to other nodes
GitHub: code + tests (Actions) + container image (GHCR) ──▶ pulled once onto group storage by gw1b-admin
```

A training run, end to end: `gw1b train --gpus 8 --gpu-type a100 --time 1-00:00:00 --chain 4 --config
configs/gw1b_1p15b.yaml` asks Slurm for a node with eight A100s for a day, four times in a row; each job
starts the container, JAX shards the model over the eight GPUs, Optax runs AdamW with a cosine schedule,
Orbax writes a checkpoint every 500 steps, and the next job in the chain resumes from the last one.
`gw1b status` shows where the job is; TensorBoard through the tunnel shows the loss.

## Glossary

| term | meaning |
|---|---|
| **aarch64 / arm64** | the ARM processor architecture of the Grace CPU in GH200 nodes; software must be built for it separately from x86-64 |
| **activation** | an intermediate result of the forward pass kept for backpropagation |
| **bf16 / fp16 / fp32 / tf32** | number formats; see §3 |
| **CUDA** | NVIDIA's programming platform; "CUDA 12 wheels" = a JAX build linked against CUDA 12 libraries |
| **FSDP** | fully sharded data parallel: parameters, gradients and optimizer state split across GPUs, gathered on demand |
| **GRES** | Slurm's "generic resource" — how GPUs are requested, by type and count |
| **HBM / GDDR** | GPU memory technologies; HBM is the fast stacked kind on data-center cards |
| **InfiniBand (IB), EDR/NDR** | the low-latency network between nodes; EDR = 100 Gb/s (V100/A100 nodes), NDR = 400 Gb/s (Blackwell nodes) |
| **MFU** | model FLOP utilization: achieved useful FLOP/s divided by the GPU's peak |
| **NCCL** | NVIDIA Collective Communications Library — multi-GPU all-reduce/all-gather |
| **node** | one computer in the cluster; a job may span several |
| **NVLink** | NVIDIA's fast GPU-to-GPU link, much faster than PCIe |
| **partition** | a named set of nodes in Slurm with shared limits |
| **PCIe** | the general-purpose bus GPUs plug into; the slow path when there is no NVLink |
| **RoPE, GQA, SwiGLU, RMSNorm** | the Llama-family architecture choices in `gw1b/model.py` (positions, attention heads, feed-forward, normalization) — Lectures 13–14 |
| **SIF** | Singularity/Apptainer Image Format — the single-file container |
| **tensor core** | the matrix-multiply unit inside modern NVIDIA GPUs |
| **token shard** | a binary file of token ids that the data loader memory-maps |
| **wall time** | elapsed clock time a job may run before Slurm ends it |
| **XLA** | the compiler JAX uses to turn Python-level array code into GPU kernels |
