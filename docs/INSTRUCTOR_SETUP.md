# Instructor / TA setup

Everything below is done by one command on a Pegasus login node — `gw1b-admin` — which asks for
the few things it cannot find out by itself and stops where it needs you. Budget: ~30 minutes of your
attention, plus unattended time for the container build (~10 min) and the FineWeb-Edu download and
tokenization (~2–3 h in Slurm jobs).

## Before you start: what HPC support has confirmed, and what is still open

Confirmed by HPC staff 9/28–10/1 (details in [CLUSTER_FACTS.md](CLUSTER_FACTS.md)): the class Unix group `MG-gw1b-class`
(`groups` shows it; your account is in it since 10/1), the group directory `/SEAS/groups/gw1b` (not created yet on 10/2 —
HPC may move the class to their new "Research NAS" layout; use `/scratch/gw1b-class/group` until they send the path) and
the scratch directory `/scratch/gw1b-class` (GPFS — **Lustre is
gone**; purges are age-based and announced); one `gpu` partition for all GPU nodes with the GPU model chosen by GRES
type (`--gres=gpu:a100:8`), 7-day wall time, **no `--account`, no QOS, no per-user limits, no preemption**; `module
load apptainer` (1.3.0) but **no fakeroot**, so the image is pulled from GHCR instead of built on Pegasus; drivers ≥ 570
everywhere (CUDA 12 wheels fine); compute nodes have internet; Open OnDemand (ood.arc.gwu.edu) has a Jupyter app that
reads `~/.local/share/jupyter/kernels`, and SSH tunnels work; InfiniBand (`ib0`/`mlx5_0`) between the GPU nodes; Globus
is available. Reservations: single-day only (a class reservation for the Friday labs is under review); for the final
run plan on shared, flexible jobs — and look at the Grace Hopper nodes, which HPC says are the least contended.

**Student accounts** are created one at a time from each student's HPC Access Request form (with their SSH public
key); a third of the class had applied by 10/1. Push everyone to apply now, and schedule HPC's onboarding session
before the first Pegasus lab (10/17). ONBOARDING.md (written by `gw1b-admin students`) tells them exactly what to do,
including the 2FA setup at first login.

Publish the container image once before running the setup: GitHub → Actions → **build-image** → Run workflow
(or push a tag `env-2026.09`), then make the package public (profile → Packages → gw1b-f2026 → Package settings →
Change visibility). `gw1b-admin build` pulls `ghcr.io/vorhersager/gw1b-f2026:2026.09` (the `GW1B_IMAGE` line of `gw1b.env`).

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
| `init` | writes paths + class group into `gw1b.local.env` (git-ignored; `gw1b.env` keeps the defaults, so `git pull` never conflicts); copies the repo to `$GW1B_GROUP/gw1b-f2026` (students must read it from there); creates the directory layout on group storage and scratch; sets group permissions; fixes executable bits / CRLF if the checkout came from Windows | confirming four paths |
| `discover` | `sinfo`/`sacctmgr`/`sbatch --test-only` → confirms the `gpu`/`cpu` partitions, reads the GPU **GRES type names** (v100/a100/l40s/rtx6000) and the `gpu` wall-time limit, whether `--account` is required, apptainer (module), login-node internet; a 3-minute `srun` on one V100 reads the **driver version** (must be ≥ 525) and checks compute-node internet; writes everything into `gw1b.local.env` and `docs/CLUSTER_FACTS.generated.md` | nothing |
| `build` | pulls the GitHub-built image into `$GW1B_GROUP/sif/gw1b-2026.09.sif` (`apptainer build … docker://`, no privileges needed); builds from `env/gw1b.def` where fakeroot exists; or the shared venv when there is no apptainer | 10–20 min wait |
| `test` | import check, the CPU test suite inside the environment, then `gw1b doctor --gpu` (a real GPU job: JAX sees the GPU, matmul TFLOP/s in the dtype that GPU supports, a 60-step training run with `model.dtype: auto`) | waiting for a GPU |
| `data` | submits a chain of `cpu` jobs: download FineWeb-Edu `sample-10BT` (~28 GB) into `$HF_HOME` on scratch → 2 GB tokenizer corpus → tokenizer training (32 cores) → tokenization (40 cores) → `$GW1B_SCRATCH/data/fineweb-edu-10B/{train,val}` + `manifest.json` (on the login node instead if `GW1B_COMPUTE_INTERNET=no`) | nothing; jobs run for 3–4 h |
| `students` | writes `onboarding/ONBOARDING.md` (the text you send to the class) and copies of `gw1b-connect.sh|.ps1` with your paths baked in, next to it (`onboarding/` is git-ignored; `laptop/` in the repo already carries the Pegasus defaults) | copy-paste |

Options: `--yes` (no prompts), `--sample 100BT` (the 100B-token corpus later in the semester),
`--vocab 32000 --type bpe|unigram`, `--skip-download` (parquet files already on scratch, e.g. via Globus),
`--force` (redo a step whose output exists), `--no-gpu-test`.

When `discover` cannot determine something it says so, and `gw1b.local.env` is a plain shell file you can edit
(`VERIFY` marks the lines in question). `gw1b doctor` — the students' check — flags what is still missing.

## If the image cannot be pulled from GHCR

Pegasus cannot run `apptainer build --fakeroot` (no subuid mapping), so the normal path is the GHCR pull above. If the
package is private or the workflow has not run, either make it public / `apptainer registry login ghcr.io` with a
GitHub token, or build the identical image anywhere with Docker and copy it over:

```bash
# on a laptop / any Linux box with Docker
docker build -t gw1b -f env/Dockerfile env
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD":/output --privileged \
    quay.io/singularity/docker2singularity:v4.1.0 --name gw1b-2026.09.sif gw1b
# copy with scp or Globus, then:
ssh pegasus 'ln -sfn gw1b-2026.09.sif /SEAS/groups/gw1b/sif/gw1b.sif'
```
HPC staff also offered to build containers on request (send them `env/gw1b.def`). Last resort: the shared venv
(`gw1b-admin build venv`): same lock file, same versions, no container.

## Running the semester

* **Interactive sessions** default to 1 V100 on the `gpu` partition for 4 h (`GW1B_GPU_DEFAULT`, `GW1B_JUPYTER_TIME`
  in `gw1b.env`). There is no debug partition: on lab days 27 concurrent sessions compete with everyone else's jobs,
  so ask HPC for a single-day reservation of a few V100 nodes for each Friday lab (they do those), or have students
  share sessions.
* **Proxy experiments** (50M–350M, 1–7B tokens): `gw1b train --config configs/100m.yaml --set run.name=…`.
  `model.dtype: auto` (the default) trains in float32 on V100s (no bf16 tensor cores) and in bf16 on A100/L40S/GH200.
  Everyone runs `gw1b budget` before submitting.
* **The GW-1B run**: `gw1b train --gpus 8 --gpu-type a100 --time 1-00:00:00 --chain 4 --config configs/gw1b_1p15b.yaml`.
  Chained 1-day jobs (HPC: 8-GPU jobs dequeued within 2 days in September; longer requests wait longer) resume from
  the last Orbax checkpoint (every 500 steps ≈ 0.5B tokens ≈ 40 min on 8 A100s), so the run can also continue on
  half a node when a full one is not free: `gw1b train --gpus 4 --gpu-type a100 --one-socket --set optim.grad_accum=2
  …` (same global batch; `--one-socket` = HPC's recipe for the four GPUs of one CPU socket). Expect ~55 h per 20B
  tokens on 8 A100s at 35 % MFU, 106 h on 4. Alternatives: one Grace Hopper node (`--gpu-type gh200`, ~5.5 days per
  20B tokens, least contended, needs the arm64 image) or a 4-GPU Blackwell node once online. 100B tokens is ~11
  A100-node-days — see [CLUSTER_FACTS.md](CLUSTER_FACTS.md) before promising it. There is no QOS or priority boost on
  Pegasus; a written proposal for a partial reservation can be submitted but HPC prefers standard scheduling.
* **Scratch**: `/scratch/gw1b-class` is not backed up; purges are age-based, announced in advance, with exceptions on
  request — still, milestone checkpoints (`ckpt_keep_every`) and the release go to `$GW1B_GROUP/release/` (a 1.15B
  bf16 checkpoint is 2.3 GB; with optimizer state 14 GB; the group quota is shared school-wide and not a concern).
* **Updating the environment** mid-semester: edit `env/requirements.in` → `env/lock.sh` → bump `GW1B_ENV_VERSION`
  in `gw1b.env` and `env/gw1b.def` → `gw1b-admin build --force`. The new `.sif` lands next to the old one and the
  `gw1b.sif` symlink moves; running jobs keep the old image. `gw1b-admin test` afterwards.
* **W&B**: compute nodes have internet, but `WANDB_MODE=offline` stays the default (no accounts needed); teams that
  want live dashboards export `WANDB_MODE=online WANDB_API_KEY=…`. TensorBoard needs no internet (port 6006 is in the tunnel).
* **Where each team plugs in**: [TEAM_PLAYBOOK.md](TEAM_PLAYBOOK.md).

## Manual reference (what `gw1b-admin` does, for the record)

```bash
# init
export GW1B_GROUP=/SEAS/groups/gw1b GW1B_SCRATCH=/scratch/gw1b-class
mkdir -p $GW1B_GROUP/{sif,tokenizer,release} $GW1B_SCRATCH/{data,hf_cache,users,teams,checkpoints,tmp}
chgrp -R MG-gw1b-class $GW1B_GROUP $GW1B_SCRATCH; chmod -R g+rX,o-rwx $GW1B_GROUP; chmod g+s $GW1B_GROUP/gw1b-f2026
chmod 2775 $GW1B_SCRATCH/{users,teams,checkpoints,tmp}; chmod 2755 $GW1B_SCRATCH/{data,hf_cache}
# discover
bash slurm/discover_cluster.sh > docs/CLUSTER_FACTS.generated.md     # then put the VERIFY values into gw1b.local.env
# build
env/build_sif.sh --from-image ghcr.io/vorhersager/gw1b-f2026:2026.09    # or env/build_venv.sh
# test
gw1b exec python -m pytest tests -q && gw1b doctor --gpu
# data
gw1b run --cpu --cpus 8 --mem 32G --time 6:00:00 -m gw1b.prepare_data download --sample 10BT
gw1b run --cpu --cpus 4 --mem 32G --time 2:00:00 -m gw1b.prepare_data sample-text --sample 10BT --out $GW1B_SCRATCH/data/tokenizer_corpus.txt --mb 2000
gw1b run --cpu --cpus 32 --mem 64G --time 3:00:00 -m gw1b.tokenizer train --input $GW1B_SCRATCH/data/tokenizer_corpus.txt --out $GW1B_GROUP/tokenizer/gw1b-32k --vocab 32000 --type bpe
sbatch --partition=$GW1B_PART_CPU slurm/tokenize.sbatch --sample 10BT --tokenizer $GW1B_GROUP/tokenizer/gw1b-32k.model --out $GW1B_SCRATCH/data/fineweb-edu-10B
```
