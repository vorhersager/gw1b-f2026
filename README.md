# GW1B — Fall 2026 · The $10,000 Language Model Challenge on GWU Pegasus

[![tests](https://github.com/vorhersager/gw1b-f2026/actions/workflows/tests.yml/badge.svg)](https://github.com/vorhersager/gw1b-f2026/actions/workflows/tests.yml)
![JAX](https://img.shields.io/badge/JAX-0.10.2%20%7C%20CUDA%2012-blue)
![Flax NNX](https://img.shields.io/badge/Flax%20NNX-0.12.8-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

The shared environment and toolchain for the GWU Machine Learning class project: design, train,
evaluate and openly release a ~1-billion-parameter language model **from random initialization**,
in JAX, on the university's Pegasus cluster. Every team and every student uses exactly this
environment — one pinned container image, one command-line tool, JupyterLab or Google Colab as the
front end, Slurm batch jobs for anything long.

```
laptop (browser / Colab UI) ──ssh tunnel──▶ Pegasus login node ──Slurm──▶ GPU node: JupyterLab + your kernel
                                                                    └──▶ batch jobs: gw1b train --config configs/100m.yaml
```

| I am a… | start here |
|---|---|
| **student** | [Install & use on Pegasus](#students-install--use-on-pegasus) → `docs/STUDENT_GUIDE.md`, `docs/COLAB.md`, notebooks `00`–`04` |
| **instructor / TA** | [Install on Pegasus](#instructors-install-the-environment-on-pegasus) → `docs/INSTRUCTOR_SETUP.md` |
| **team lead** | `docs/TEAM_PLAYBOOK.md` — where Teams 1–5 plug into the toolchain |
| **curious** | `docs/ARCHITECTURE.md`, `docs/CLUSTER_FACTS.md` (what Pegasus can do for a 1B model) |

---

## Students: install & use on Pegasus

There is nothing to install. The environment lives on the cluster; you connect to it.

### 0. Access
* A Pegasus account (requested for the class by the instructor) and the **GW VPN** when off campus.
* A terminal with `ssh` (macOS/Linux: built in; Windows 10+: built in, or use PowerShell).

### 1. One-time setup (5 minutes)
```bash
ssh <netid>@pegasus.arc.gwu.edu
source /SEAS/groups/gw1b/gw1b-f2026/activate.sh     # exact path: see the onboarding message from your instructor
gw1b setup           # adds the environment to ~/.bashrc, creates your folders, copies starter notebooks to ~/gw1b
gw1b doctor          # every line should say [ok]
```
Optional: `ssh-copy-id <netid>@pegasus.arc.gwu.edu` from your laptop so tunnels don't ask for a password.

### 2. Every day: a GPU JupyterLab (also usable from Google Colab)
```bash
gw1b jupyter                                  # 1 GPU, 4 h on the debug partition (defaults)
gw1b jupyter --gpus 2 --time 3:00:00 --partition small-gpu
```
When the job starts it prints:
```
  1) On YOUR LAPTOP, in a new terminal (keep it open):
       ssh -N -L 8888:gpu017:8891 -L 6006:gpu017:6006 <netid>@pegasus.arc.gwu.edu
  2) JupyterLab:   http://localhost:8888/lab?token=…
     Colab:        Connect ▾ -> "Connect to a local runtime" -> http://localhost:8888/?token=…
  3) Finished?     gw1b cancel 123456
```
Run line 1 on your laptop, then open the URL in a browser — or in Colab choose *Connect ▾ → Connect to a
local runtime* and paste it. The notebook UI is wherever you like; **the kernel runs on a Pegasus GPU node**.
One command from the laptop that does both steps and opens the browser:
`laptop/gw1b-connect.sh <netid>` (macOS/Linux) or `laptop\gw1b-connect.ps1 <netid>` (Windows).

Start with `~/gw1b/notebooks/00_hello_pegasus.ipynb` → `01_tokenizer` → `02_data_pipeline` →
`03_train_proxy_model` → `04_evaluate_and_export`.

### 3. Anything longer than an hour: a batch job
```bash
gw1b train --gpus 1 --time 12:00:00 --config configs/50m.yaml --set run.name=team2-gqa-vs-mha
gw1b train --gpus 4 --time 1-00:00:00 --config configs/350m.yaml --set run.name=team3-wsd --set optim.schedule=wsd
gw1b run   --gpus 1 --time 2:00:00 -m gw1b.lm_eval_adapter --run $GW1B_SCRATCH/users/$USER/runs/team2-gqa-vs-mha --tasks hellaswag,arc_easy --limit 500
gw1b run   --cpu --cpus 16 --time 4:00:00 my_script.py --arg value       # CPU-only job
gw1b status            # your jobs + free GPUs per partition
gw1b cancel <jobid>
```
Runs land in `$GW1B_SCRATCH/users/<netid>/runs/<run.name>/` with `config.yaml`, `metrics.jsonl`, `tb/`,
`checkpoints/` and `summary.json` (loss, tokens, FLOPs, GPU-hours, $ — the numbers every experiment reports).
A job that hits its time limit **resumes from its last checkpoint** when resubmitted; `--chain 4` queues four
dependent jobs up front. V100 nodes: add `--set model.dtype=float32` (no bf16 tensor cores).

### 4. Know the cost before you submit
```bash
gw1b budget --config configs/350m.yaml --tokens 7e9 --gpu a100 --n-gpus 4
gw1b python -m gw1b.train --config configs/100m.yaml --dry-run      # resolved config + parameter count + cost
```

### Cheat sheet
```
gw1b jupyter [--gpus N --time T --partition P]   gw1b jupyter --info | --stop
gw1b train [job opts] --config C [--set k=v]     gw1b run [job opts] <python args>     gw1b shell [job opts]
gw1b status        gw1b cancel <id|all>          gw1b doctor [--gpu]      gw1b budget …      gw1b kernel
python -m gw1b.evaluate --run R      python -m gw1b.generate --run R --prompt "…"      python -m gw1b.export_hf --run R --out D --verify
```
Full guide: [`docs/STUDENT_GUIDE.md`](docs/STUDENT_GUIDE.md) · Colab: [`docs/COLAB.md`](docs/COLAB.md) ·
problems: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

---

## Instructors: install the environment on Pegasus

One command on a login node does it all — asks for four paths, discovers the cluster facts, builds the
image, tests it, starts the data pipeline and writes the onboarding text for the students:

```bash
ssh <netid>@pegasus.arc.gwu.edu
git clone https://github.com/vorhersager/gw1b-f2026.git
bash gw1b-f2026/bin/gw1b-admin all         # "bash …" the first time: a Windows checkout may lack the executable bit
```

| step (`gw1b-admin <step>`, all re-runnable; `gw1b-admin status` shows progress) | what it does |
|---|---|
| `init` | paths + class Unix group → `gw1b.env`; copies the repo to `$GW1B_GROUP/gw1b-f2026` (students read it from there); creates the storage layout with group permissions |
| `discover` | `sinfo` / `sacctmgr` / `sbatch --test-only` + a 3-minute GPU probe → A100/L40S partition names, `--account` requirement, GRES syntax, wall-time limits, apptainer, NVIDIA driver version (≥ 525 required), internet — written into `gw1b.env` and `docs/CLUSTER_FACTS.generated.md` |
| `build` | the container `$GW1B_GROUP/sif/gw1b-2026.09.sif` from `env/gw1b.def` (≈ 7 GB, ≈ 10 min), or the shared `uv` venv if there is no Apptainer |
| `test` | imports, the CPU test suite inside the environment, `gw1b doctor --gpu` (a real GPU job) |
| `data` | FineWeb-Edu `sample-10BT` download (≈ 28 GB, login node) → tokenizer corpus → tokenizer job → tokenization job (chained) → `$GW1B_SCRATCH/data/fineweb-edu-10B` |
| `students` | bakes your paths into the laptop scripts and writes `ONBOARDING.md` — the message you send to the class |

Before you start, line up with HPC support: the class Unix group with all students in it, group storage
(`/<SCHOOL>/groups/gw1b`) and Lustre scratch (`/lustre/groups/gw1b`, with a purge extension), Pegasus accounts for
the students, and — nice to have — a reservation of one A100 node for the final run. If `apptainer build` is
refused on the login node, build the identical image with Docker on a laptop (`env/Dockerfile`) and copy the `.sif`.
Everything else: [`docs/INSTRUCTOR_SETUP.md`](docs/INSTRUCTOR_SETUP.md).

**Updating mid-semester**: edit `env/requirements.in` → `env/lock.sh` → bump `GW1B_ENV_VERSION` → `gw1b-admin build --force`.
The new image lands next to the old one; running jobs keep the old image.

---

## The toolchain

The `gw1b` Python package (JAX 0.10 · Flax NNX · Optax · Orbax · SentencePiece · Hugging Face datasets · lm-eval):

| module | what it does |
|---|---|
| `gw1b.model` | Llama-style decoder in Flax NNX: RMSNorm/LayerNorm, RoPE/learned positions, GQA/MHA/MQA, SwiGLU/GELU, tied embeddings, KV cache. Hugging Face conventions, so checkpoints export as a standard `LlamaForCausalLM`. |
| `gw1b.config` | one YAML per run (`model / data / optim / run`), `--set a.b=c` overrides, `extends:` inheritance |
| `gw1b.data` | uint16 token shards + memory-mapped, deterministic, resumable, data-parallel batch loader |
| `gw1b.tokenizer` | SentencePiece (BPE/unigram) training + wrapper + HF tokenizer export |
| `gw1b.prepare_data` | FineWeb-Edu download → tokenizer corpus → token shards (multiprocess), data manifest; document filters plug in |
| `gw1b.train` | FSDP training on 1–16 GPUs (multi-node through Slurm), AdamW/Lion, cosine/WSD/linear schedules, gradient accumulation, Orbax checkpoints, auto-resume, `metrics.jsonl` + TensorBoard (+ W&B), MFU / GPU-hours / $ accounting |
| `gw1b.evaluate` | validation loss / perplexity of any checkpoint |
| `gw1b.generate` | greedy / sampling generation with a KV cache (batched) |
| `gw1b.lm_eval_adapter` | HellaSwag, ARC, PIQA, Winogrande, MMLU, GSM8K, … on the JAX model through lm-evaluation-harness |
| `gw1b.export_hf` | checkpoint → `LlamaForCausalLM` safetensors, verified against `transformers` |
| `gw1b.budget` | FLOPs → GPU-hours → wall-clock → $ for any config and GPU type |

Configs: `configs/tiny_debug.yaml` (runs in a minute anywhere), the scaling ladder `50m` / `100m` / `200m` / `350m`,
and `gw1b_1p15b.yaml` — the design-document architecture (32 layers, d = 1792, 28 heads / 7 KV heads, SwiGLU 4864,
32K vocab, 2048 context; 1,151M parameters).

### What Pegasus can do for a 1B model (35 % MFU)

| config | tokens | 1×V100 (f32) | 8×A100 | GPU-hours |
|---|---|---|---|---|
| 50M | 1B | 23 h | 0.1 h | 1 |
| 350M | 7B | 951 h | 6 h | 48 |
| 1.15B | 20B | — | **53 h** | 423 |
| 1.15B | 100B | — | **264 h (11 days)** | 2115 |

The proxy ladder is cheap; the final run is the scarce resource (16 A100s for the whole university). Details and
the V100/A100/L40S/GH200 specifics: [`docs/CLUSTER_FACTS.md`](docs/CLUSTER_FACTS.md).

---

## Repository layout

```
gw1b.env            the ONE config file (paths, partitions, account, defaults)     ← filled in by gw1b-admin
activate.sh         students `source` this
bin/gw1b            student CLI: setup jupyter run train shell python budget status cancel doctor kernel
bin/gw1b-admin      instructor CLI: init discover build test data students status all
bin/gw1b-exec       runs any command inside the environment (container or venv) — used by everything
env/                gw1b.def (Apptainer), Dockerfile, requirements.in/.txt (lock), requirements-eval.txt, build scripts
slurm/              job templates: jupyter, single-node run, multi-node train, tokenize, GPU smoke test, discovery
laptop/             one-command connect scripts for macOS/Linux and Windows
gw1b/               the Python package
configs/            tiny_debug, 50m, 100m, 200m, 350m, gw1b_1p15b
notebooks/          00 hello → 01 tokenizer → 02 data → 03 train → 04 evaluate/export (+ make_notebooks.py)
tests/              CPU smoke tests (model, KV cache, sharded training, loader, checkpoint/resume, generation, export)
docs/               STUDENT_GUIDE · COLAB · INSTRUCTOR_SETUP · TEAM_PLAYBOOK · CLUSTER_FACTS · ARCHITECTURE · TROUBLESHOOTING
```

## Developing on a laptop (no GPU needed)

```bash
git clone https://github.com/vorhersager/gw1b-f2026.git && cd gw1b-f2026
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                     # CPU JAX; add ".[eval]" for lm-eval + transformers + torch
python -m pytest tests -q                   # ~2 minutes
python -m gw1b.train --config configs/tiny_debug.yaml --dry-run
```
The notebooks and the whole training loop run on CPU (the tests simulate 8 devices for the FSDP path), so
students can develop on their own machines and submit to Pegasus when it works.

## Contributing (class workflow)

* Branch per team (`team1-data`, `team2-arch`, …), pull requests into `main`; the CI runs the CPU tests.
* Commit configs, analysis code, notebooks (cleared outputs) and papers — **never** datasets, checkpoints or `.sif` images
  (`.gitignore` blocks the common ones).
* Environment changes (new packages) go through `env/requirements.in` + `env/lock.sh` in a PR, then the instructor
  rebuilds the image so everyone stays identical.

## Versions

Python 3.12 · `jax[cuda12]==0.10.2` · `flax==0.12.8` · `optax==0.2.8` · `orbax-checkpoint==0.12.6` · `grain==0.2.18` ·
`sentencepiece==0.2.2` · `tokenizers==0.23.2` · `datasets==5.0.1` · `jupyterlab==4.6.4` · `lm-eval==0.4.13` ·
`transformers==5.17.0` · CPU-only `torch==2.14.0` — fully pinned in `env/requirements.txt`. Environment version **2026.09**.

## License

Code in this repository is released under the Apache License 2.0 (see `LICENSE`). Model weights, the tokenizer and the
data manifest produced by the class are released separately with the model card at the end of the semester.

## Acknowledgements

Computing resources are provided by GW Information Technology's High Performance Computing group (Pegasus).
The pretraining corpus is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By).
