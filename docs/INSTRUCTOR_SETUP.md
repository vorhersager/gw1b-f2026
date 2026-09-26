# Instructor / TA setup

Everything below is done by one command on a Pegasus login node — `gw1b-admin` — which asks for
the few things it cannot find out by itself and stops where it needs you. Budget: ~30 minutes of your
attention, plus unattended time for the container build (~10 min) and the FineWeb-Edu download and
tokenization (~2–3 h in Slurm jobs).

## Before you start: what to get from HPC support

From the meeting, confirm or request:

* a **Unix group** for the class (the proposal said `gw1b-class`) with all 24 students in it;
* **group storage** `/<SCHOOL>/groups/gw1b` (persistent, not purged) and **Lustre scratch**
  `/lustre/groups/gw1b` (ask for the purge extension through the end of the semester);
* Pegasus accounts for every student (HPC access request form);
* whether jobs need a Slurm **account** for the class, and the names of the **A100 / L40S partitions**
  (`gw1b-admin discover` finds these itself if you can see them in `sinfo`);
* whether **Apptainer** is installed on the compute nodes and whether `apptainer build --fakeroot` is allowed
  on login nodes (if not: build the image on a laptop with Docker, or use the venv fallback — both are automated);
* optional: Open OnDemand for Pegasus, and a reservation of one A100 node for the final run.

## The fast path

```bash
ssh <netid>@pegasus.arc.gwu.edu
git clone https://github.com/vorhersager/gw1b-f2026.git   # or scp / Globus the folder from your laptop
bash gw1b-f2026/bin/gw1b-admin all                # asks: group dir, scratch dir, class group, login host
```

`gw1b-admin all` runs six steps; each is also available on its own (`gw1b-admin <step>`), is idempotent
(re-run any time), and `gw1b-admin status` shows what is done:

| step | what it does | needs you for |
|---|---|---|
| `init` | writes paths + class group into `gw1b.env`; copies the repo to `$GW1B_GROUP/gw1b-f2026` (students must read it from there); creates the directory layout on group storage and scratch; sets group permissions; fixes executable bits / CRLF if the checkout came from Windows | confirming four paths |
| `discover` | `sinfo`/`sacctmgr`/`sbatch --test-only` → finds the A100/L40S partitions, whether `--account` is required, the GRES syntax, the A100 partition's wall-time limit, apptainer (module?), login-node internet; a 3-minute `srun` on the debug GPU partition reads the **driver version** (must be ≥ 525) and checks compute-node internet; writes everything into `gw1b.env` and `docs/CLUSTER_FACTS.generated.md` | nothing (asks only if a default partition name does not exist) |
| `build` | builds the container `$GW1B_GROUP/sif/gw1b-2026.09.sif` from `env/gw1b.def` (or the shared venv when there is no apptainer) | ~10 min wait |
| `test` | import check, the CPU test suite inside the environment, then `gw1b doctor --gpu` (a real GPU job: JAX sees the GPU, bf16 matmul TFLOP/s, a 60-step training run) | waiting for a GPU |
| `data` | downloads FineWeb-Edu `sample-10BT` (~28 GB) on the login node into `$HF_HOME` on scratch, writes a 2 GB tokenizer corpus, submits the tokenizer-training job (32 cores) and the tokenization job (40 cores) chained after it → `$GW1B_SCRATCH/data/fineweb-edu-10B/{train,val}` + `manifest.json` | nothing; jobs run for 2–3 h |
| `students` | bakes your paths into `laptop/gw1b-connect.sh|.ps1` and writes `ONBOARDING.md` — the text you send to the class | copy-paste |

Options: `--yes` (no prompts), `--sample 100BT` (the 100B-token corpus later in the semester),
`--vocab 32000 --type bpe|unigram`, `--skip-download` (parquet files already on scratch, e.g. via Globus),
`--force` (redo a step whose output exists), `--no-gpu-test`.

When `discover` cannot determine something it says so, and `gw1b.env` is a plain shell file you can edit
(`VERIFY` marks the lines in question). `gw1b doctor` — the students' check — flags what is still missing.

## If the container build is refused on the login node

`apptainer build` needs `--fakeroot` (or root). If HPC support will not enable it, build the identical image
anywhere with Docker and copy it over:

```bash
# on a laptop / any Linux box with Docker
docker build -t gw1b -f env/Dockerfile env
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD":/output --privileged \
    quay.io/singularity/docker2singularity:v4.1.0 --name gw1b-2026.09.sif gw1b
scp gw1b-2026.09.sif <netid>@pegasus.arc.gwu.edu:/SEAS/groups/gw1b/sif/ && ssh pegasus 'ln -sfn gw1b-2026.09.sif /SEAS/groups/gw1b/sif/gw1b.sif'
```
or fall back to the shared venv (`gw1b-admin build venv`): same lock file, same versions, no container.

## Running the semester

* **Interactive sessions** default to the debug GPU partition, 1 GPU, 4 h (the proposal's 2–4 h turnover);
  change `GW1B_DEFAULT_*` / `GW1B_JUPYTER_TIME` in `gw1b.env`.
* **Proxy experiments** (50M–350M, 1–7B tokens): `gw1b train --config configs/100m.yaml --set run.name=…`.
  V100 nodes: `--set model.dtype=float32` (no bf16 tensor cores); A100/L40S: bf16 by default. Everyone runs
  `gw1b budget` before submitting.
* **The GW-1B run**: `gw1b train --gpus 8 --partition $GW1B_PART_A100 --time $GW1B_MAX_TIME --chain 8 --config configs/gw1b_1p15b.yaml`.
  Chained jobs resume from the last Orbax checkpoint (every 500 steps ≈ 0.5B tokens ≈ 40 min on 8 A100s).
  Two nodes: `--nodes 2`. Expect ~55 h per 20B tokens on one 8×A100 node at 35 % MFU; 100B tokens is ~11 days
  on one node, 5.5 on two — see [CLUSTER_FACTS.md](CLUSTER_FACTS.md) before promising 100B tokens.
* **Scratch purge**: everything under `$GW1B_SCRATCH` can disappear at the start of a month. Milestone checkpoints
  (`ckpt_keep_every`) and the release go to `$GW1B_GROUP/release/` (a 1.15B bf16 checkpoint is 2.3 GB; with
  optimizer state 14 GB; the group quota is ~250 GB).
* **Updating the environment** mid-semester: edit `env/requirements.in` → `env/lock.sh` → bump `GW1B_ENV_VERSION`
  in `gw1b.env` and `env/gw1b.def` → `gw1b-admin build --force`. The new `.sif` lands next to the old one and the
  `gw1b.sif` symlink moves; running jobs keep the old image. `gw1b-admin test` afterwards.
* **W&B**: compute nodes are assumed offline (`WANDB_MODE=offline`); students `wandb sync` from a login node.
  TensorBoard needs no internet (port 6006 is in the tunnel).
* **Where each team plugs in**: [TEAM_PLAYBOOK.md](TEAM_PLAYBOOK.md).

## Manual reference (what `gw1b-admin` does, for the record)

```bash
# init
export GW1B_GROUP=/SEAS/groups/gw1b GW1B_SCRATCH=/lustre/groups/gw1b
mkdir -p $GW1B_GROUP/{sif,tokenizer,release} $GW1B_SCRATCH/{data,hf_cache,users,teams,checkpoints,tmp}
chgrp -R gw1b-class $GW1B_GROUP $GW1B_SCRATCH; chmod -R g+rX,o-rwx $GW1B_GROUP; chmod g+s $GW1B_GROUP/gw1b-f2026
chmod 2775 $GW1B_SCRATCH/{users,teams,checkpoints,tmp}; chmod 2755 $GW1B_SCRATCH/{data,hf_cache}
# discover
bash slurm/discover_cluster.sh > docs/CLUSTER_FACTS.generated.md     # then edit the VERIFY lines of gw1b.env
# build
env/build_sif.sh          # or env/build_venv.sh
# test
gw1b exec python -m pytest tests -q && gw1b doctor --gpu
# data
gw1b python -m gw1b.prepare_data download --sample 10BT
gw1b python -m gw1b.prepare_data sample-text --sample 10BT --out $GW1B_SCRATCH/data/tokenizer_corpus.txt --mb 2000
gw1b run --cpu --cpus 32 --mem 64G --time 3:00:00 -m gw1b.tokenizer train --input $GW1B_SCRATCH/data/tokenizer_corpus.txt --out $GW1B_GROUP/tokenizer/gw1b-32k --vocab 32000 --type bpe
sbatch --partition=$GW1B_PART_CPU slurm/tokenize.sbatch --sample 10BT --tokenizer $GW1B_GROUP/tokenizer/gw1b-32k.model --out $GW1B_SCRATCH/data/fineweb-edu-10B
```
