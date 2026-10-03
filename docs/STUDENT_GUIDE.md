# Student guide

You never install anything. Every team uses the same environment on Pegasus; you reach it from
JupyterLab in your browser, from Google Colab, or as batch jobs.

New to GPUs, Slurm or JAX? Read [PRIMER.md](PRIMER.md) first — it explains every name used below.

## 1. First time (5 minutes)

Before anything else you need a Pegasus account, and HPC creates them one at a time from your **HPC Access Request
form**, which needs your **SSH public key**: on your laptop run `ssh-keygen -t ed25519` (press Enter at the prompts),
then paste the contents of `~/.ssh/id_ed25519.pub` into the form. When the account email arrives, log in once with the
single-use code HPC sends ("HPC Pegasus 2FA") and **set up 2FA immediately** (step "02 Initial Login + 2FA Setup" at
https://github.com/gwuniversity/hpc-onboarding) — without it you cannot log in a second time. Off campus, connect to
the GW VPN first. Every new ssh connection asks for your 2FA code.

```bash
ssh <netid>@pegasus.arc.gwu.edu
source /SEAS/groups/gw1b/gw1b-f2026/activate.sh      # the path your instructor gave you
gw1b setup           # once: adds the line above to ~/.bashrc, creates your scratch folders, copies the notebooks to ~/gw1b
gw1b doctor          # every line [ok]?
gw1b kernel          # once: makes the environment selectable in the Open OnDemand Jupyter app
gw1b notebooks       # later in the semester: refresh ~/gw1b/notebooks from the class repo (your edits are kept as .bak)
```
Help from HPC: Zoom office hours Tue/Thu 12:30–14:30 (https://gwu-edu.zoom.us/j/91295945575) or hpchelp@gwu.edu.

## 2. Every day: a GPU JupyterLab

On Pegasus:
```bash
gw1b jupyter                       # 1 GPU, 4 h on the debug partition (defaults)
gw1b jupyter --gpus 2 --time 3:00:00               # 2 V100s (4 = a NVLink node); --gpu-type a100 for an A100
```
Wait until it prints:
```
  1) On YOUR LAPTOP, in a new terminal (keep it open):
       ssh -N -L 8888:gpu017:8891 -L 6006:gpu017:6006 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 jsmith@pegasus.arc.gwu.edu
  2) JupyterLab:   http://localhost:8888/lab?token=…
     Colab:        Connect ▾ -> "Connect to a local runtime" -> http://localhost:8888/?token=…
  3) Finished?     gw1b cancel 123456
```
Copy line 1 into a terminal on your laptop, then open the JupyterLab URL in your browser **or** connect
Colab to it ([COLAB.md](COLAB.md)). Your notebooks live in your Pegasus home (`~/gw1b/notebooks`), the
code runs on the GPU node, and the job ends at its time limit — save your notebooks; checkpoints are on disk anyway.

Shortcut from the laptop (does both steps and opens the browser):
`laptop/gw1b-connect.sh <netid> [--gpus 1 --time 2:00:00]` (macOS/Linux) or `laptop\gw1b-connect.ps1 <netid>` (Windows).

Useful:
* `gw1b jupyter --info` — print the connection lines again; `gw1b jupyter --stop` — end the session.
* `%load_ext tensorboard` / `%tensorboard --logdir <run>/tb` inside a notebook (port 6006 is tunnelled).
* A JupyterLab *Terminal* runs inside the environment on the GPU node — handy for `nvidia-smi`, quick scripts.


### Two ways to get a notebook on a GPU node

| | `gw1b jupyter` + ssh tunnel | Open OnDemand Jupyter app (https://ood.arc.gwu.edu) |
|---|---|---|
| start | `gw1b jupyter` on the login node, then the printed `ssh -N -L …` on your laptop (the laptop script does both with one 2FA prompt) | portal → *Jupyter Notebook Pegasus* → pick partition `gpu`, GPU type, time → review every field → Launch |
| notebook UI | JupyterLab in your browser **or Google Colab** (*Connect to a local runtime*) | classic Jupyter in the portal |
| the GW1B environment | automatic | run `gw1b kernel` once; then select the **GW1B (JAX)** kernel — the app reads `~/.local/share/jupyter/kernels` |
| good for | Colab users, TensorBoard (port 6006 is in the tunnel) | no ssh, no tunnel |


## 3. Longer work: batch jobs

Anything longer than an hour runs as a Slurm job so it does not depend on your laptop or your Jupyter time limit.
```bash
gw1b train --gpus 1 --time 12:00:00 --config configs/50m.yaml --set run.name=team2-gqa-vs-mha
gw1b train --gpus 4 --time 1-00:00:00 --config configs/350m.yaml --set run.name=team3-wsd --set optim.schedule=wsd
gw1b run   --gpus 1 --time 2:00:00 -m gw1b.lm_eval_adapter --run $GW1B_SCRATCH/users/$USER/runs/team2-gqa-vs-mha --tasks hellaswag,arc_easy --limit 500
gw1b run   --cpu  --cpus 16 --time 4:00:00 my_script.py --arg value        # CPU-only job (data work)
gw1b status                                                                 # your jobs, free GPUs
gw1b cancel <jobid>
```
* Output goes to `$GW1B_SCRATCH/users/<netid>/jobs/<name>-<jobid>.out`; the run itself to
  `$GW1B_SCRATCH/users/<netid>/runs/<run.name>/` (`config.yaml`, `metrics.jsonl`, `tb/`, `checkpoints/`, `summary.json`).
* **Time limits**: if the job hits its limit, just resubmit the same command — it resumes from the last checkpoint.
  For runs longer than a limit submit a chain: `--chain 4` queues four dependent jobs.
* **Which GPUs**: Pegasus has one `gpu` partition; the model is chosen with `--gpu-type v100|a100|l40s` (default
  V100; `--gpus 5–8` implies A100). V100s have no bf16: `model.dtype: auto` trains in float32 there. `gw1b status` shows how
  many nodes of each type are idle right now. There are no per-user limits and no QOS: priority is fair-share, so
  ask for what you need and no more — 1–2-GPU jobs start fastest, and nothing is ever preempted once it runs.
* **Wall time**: the `gpu` partition allows 7 days, but short jobs start sooner; for anything over a day use
  `--time 1-00:00:00 --chain N` and let the checkpoint/resume logic do the rest.
* Interactive shell on a GPU node (for debugging): `gw1b shell --gpus 1 --time 1:00:00`.

## 4. Where files live

| path | what | notes |
|---|---|---|
| `~` | notebooks, small files | 25 GB quota, backed up-ish; **never** run jobs here |
| `$GW1B_SCRATCH/users/<netid>/` | your runs, checkpoints, JAX compile cache | fast GPFS scratch, **not backed up** |
| `$GW1B_SCRATCH/teams/team<N>/` | team-shared runs and data | same |
| `$GW1B_SCRATCH/data/` | the class corpora (token shards) and HF cache | read-only |
| `$GW1B_GROUP/` | the environment, tokenizer, released models | read-only |

Copy anything you want to keep (figures, `summary.json`, `metrics.jsonl`, final checkpoints) to `~` or your team's
GitHub repo. A checkpoint of a 350M model is ~1.4 GB (bf16) / 4.2 GB with optimizer state.

## 5. Running your own code

The class package is importable everywhere (`import gw1b`). Your own code goes in your home or team folder:
```python
import sys; sys.path.insert(0, "/scratch/gw1b-class/teams/team2/ourcode")     # in a notebook
```
or `cd` to that folder before `gw1b run script.py`. To modify the package itself, copy `gw1b/` into your
folder and put it first on `PYTHONPATH` (`export PYTHONPATH=/path/to/yourcopy:$PYTHONPATH`) — your copy wins.
Need a package that is not installed? Ask the instructor to add it to `env/requirements.in` (everyone gets the
same environment; `pip install --user` is disabled on purpose).

## 6. Etiquette (the cluster is shared with the whole university)

* Do not run computations on the login node — `gw1b python` is only for `--dry-run`, `gw1b budget`, tiny checks.
* Ask for the time you need, not the maximum. Cancel Jupyter jobs you are done with (`gw1b jupyter --stop`).
* One Jupyter job per person; team experiments as batch jobs with `run.name=team<N>-<what>`.
* Report every experiment with the numbers `summary.json` gives you: loss/benchmarks, tokens, FLOPs, GPU-hours, $.

## 7. Cheat sheet

```
gw1b jupyter [--gpus N --time T --partition P]   gw1b jupyter --info | --stop
gw1b train [job opts] --config C [--set k=v]     gw1b run [job opts] <python args>     gw1b shell [job opts]
gw1b status        gw1b cancel <id|all>          gw1b doctor [--gpu]      gw1b budget --config C --tokens N --gpu a100 --n-gpus 8
gw1b python -m gw1b.train --config C --dry-run   (prints the resolved config + cost estimate)
python -m gw1b.evaluate --run R      python -m gw1b.generate --run R --prompt "…"      python -m gw1b.export_hf --run R --out D --verify
```
