# Troubleshooting

**`gw1b: command not found`** — `source /SEAS/groups/gw1b/gw1b-f2026/activate.sh` (or `gw1b setup` once, then open a new shell).

**`gw1b-exec: no environment found`** — neither the container (`$GW1B_SIF`) nor the venv (`$GW1B_VENV`) exists or
`apptainer` is not available. Instructor: `env/build_sif.sh` or `env/build_venv.sh`; if apptainer is a module, set
`GW1B_APPTAINER_MODULE` in `gw1b.env`.

**`Permission denied (publickey)` when you ssh** — your SSH public key is not on your Pegasus account yet (it is
registered through the HPC Access Request form), or ssh is offering a different key: `ssh -i ~/.ssh/id_ed25519 …`.
Password login does not exist.

**`Verification code:` prompt** — that is Pegasus 2FA. The first time, use the single-use code HPC emailed you
("HPC Pegasus 2FA") and set up the authenticator immediately; afterwards every new ssh connection asks for the
6-digit code. The laptop script opens one connection and reuses it for the tunnel, so you type it once.

**`git pull` says `Your local changes to the following files would be overwritten by merge: gw1b.env`** — an older
`gw1b-admin init`/`discover` edited `gw1b.env` in place. Move those values to the git-ignored `gw1b.local.env` (which
`gw1b.env` now sources first), restore the tracked file, and pull:
```bash
git diff -U0 gw1b.env | grep '^+export' | sed 's/^+//' > gw1b.local.env   # keep what init/discover wrote
git checkout -- gw1b.env && git pull
gw1b-admin status      # rewrites the copied lines as plain values and shows what is set
```
From now on init/discover write `gw1b.local.env` only (and `gw1b-admin` does this migration itself when it finds an
edited `gw1b.env`), so pulls stay clean. `gw1b-admin students` likewise writes to `onboarding/` instead of `laptop/`.

**`gw1b doctor` shows the shipped defaults (`group dir /SEAS/groups/gw1b`, `no container image at /SEAS/groups/gw1b/sif/…`)**
— its first line says which files it read. `gw1b.env only — no gw1b.local.env` means the site values are missing: the
instructor runs `gw1b-admin init` (and `discover`). If `gw1b.local.env` is listed but the values are still the defaults,
it holds `export KEY="${KEY:-…}"` lines (copied by hand) and a stale value exported by an earlier `source activate.sh`
in the same shell won: `gw1b-admin status` rewrites the file as plain `export KEY="…"` lines, after which every shell
picks the site values up again.

**`sbatch: error: invalid partition` / `Invalid generic resource` / `Invalid account`** — the `VERIFY` values (`GW1B_PART_*`, `GW1B_GPU_*`,
`GW1B_ACCOUNT`, `GW1B_GRES`) are wrong for this cluster. Run `gw1b-admin discover` (writes `gw1b.local.env`) or set them there by hand.

**JAX sees no GPU (`jax.devices()` → CpuDevice) inside a GPU job**
1. `nvidia-smi` in the job output shows the driver: it must be **≥ 525** for the CUDA 12 wheels. If it is older, ask HPC
   staff for a driver update or to install the `cuda-compat` forward-compatibility package (datacenter GPUs support it);
   alternatively add `nvidia-cuda-compat` handling to the image — ask the instructor.
2. The container must be started with `--nv` (`gw1b-exec` adds it when `nvidia-smi` exists on the node).
3. Was a GPU actually allocated? `echo $CUDA_VISIBLE_DEVICES` inside the job; check `--gres` syntax.
4. `JAX_PLATFORMS` must not be set to `cpu` (some login-node profiles do that).

**`RESOURCE_EXHAUSTED: Out of memory`** — reduce `run.batch_size` (keep tokens/step by raising `optim.grad_accum`),
shorten `data.seq_len`, make sure `run.shard_params=true`, or lower `XLA_PYTHON_CLIENT_MEM_FRACTION`. On 16 GB V100s a
350M model needs batch ≤ 8 sequences of 2048 per GPU.

**Very slow first step** — XLA compilation (30 s for a 50M model, several minutes for 1.15B). It is cached in
`$JAX_COMPILATION_CACHE_DIR` on scratch, so the second run is fast. Do not put that cache in `$HOME` (25 GB quota).

**Slow training on V100** — bf16 is emulated on Volta (no bf16 tensor cores). `model.dtype: auto` (the default) already
picks float32 there; if a config forces `bfloat16`, override with `--set model.dtype=float32` and expect ~15 TFLOP/s per GPU.

**`UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32`** — a bf16 program on a V100
(or T4): JAX's attention asks XLA for a bf16 dot algorithm that only exists on Ampere and newer. The model now
computes attention in f32 on those GPUs and `model.dtype: auto` avoids bf16 there altogether; if you still see it,
`git pull` the repository (fixed 2026-10-03) or add `--set model.dtype=float32`.

**`cudnn` attention error** — `--set model.attn_implementation=xla` (auto-selection picked cuDNN on an unsupported
shape/GPU). Please report the shape to the instructor.

**Jupyter: "job is PENDING (Resources)"** — no idle GPU of that type; wait, check `gw1b status`, or try `--gpu-type any` / fewer GPUs /
shorter `--time`. `gw1b status` shows free GPUs per partition.

**Jupyter connects but kernels die / `import gw1b` fails** — you are probably in an Open OnDemand or hosted-Colab kernel,
not the GW1B one. Use the URL printed by `gw1b jupyter`, or `gw1b kernel` and select *GW1B (JAX)* as the kernel.

**The tunnel terminal prints `channel N: open failed: connect failed: Connection refused`** — the login node reached the
GPU node, but nothing is listening on the Jupyter port there (yet). JupyterLab needs 20–90 s to start inside the
container; `gw1b jupyter` now waits for it before printing the tunnel line, and `gw1b jupyter --info` says whether the
port is *answering*. If it stays refused: `tail -n 30 $GW1B_SCRATCH/users/$USER/jobs/gw1b-jupyter-<jobid>.out` — a
JupyterLab error there (port in use, unwritable `~/.jupyter`, …) means the job died; `gw1b cancel <jobid>` and start again.
The tunnel itself can stay open; just reload the browser tab / click Connect again once the port answers.

**Server running, tunnel open, browser still cannot connect** — nine times out of ten an *older* tunnel window is still
open on the laptop: it owns local port 8888 and forwards to the previous job's node/port, and a second `ssh -N -L 8888:…`
then prints `bind: Address already in use` / `cannot listen to port: 8888` (with `-o ExitOnForwardFailure=yes` it exits).
Close every tunnel window, run `gw1b jupyter --info` on Pegasus, start the printed line in a fresh laptop terminal, and
open the printed URL (its token belongs to the current job). `http://127.0.0.1:8888/lab?token=…` works where `localhost`
resolves oddly. Quick checks: on the laptop `curl -sI http://127.0.0.1:8888/lab | head -1` (an `HTTP/1.1 …` line = the
tunnel works, the problem is the URL/token/browser); on the login node `gw1b jupyter --info` says *answering*.

**Colab: "Unable to connect to the runtime"** — see [COLAB.md](COLAB.md): the tunnel terminal must stay open; paste the
URL with the token; try the JupyterLab URL in a browser tab first.

**`ssh: connect to host … port 22: Connection timed out`** — VPN (off campus). Tunnel dies after idle time →
`ssh -o ServerAliveInterval=60 …` (the laptop scripts already do this).

**`Disk quota exceeded` in home** — caches went to `$HOME`. `gw1b.env` sends `HF_HOME`, `WANDB_DIR`, the JAX cache and
`PYTHONPYCACHEPREFIX` to scratch/`/tmp`; check `du -sh ~/.cache ~/.local` and delete.

**My data/checkpoints disappeared** — scratch (`/scratch/gw1b-class`, GPFS) is not backed up and may be purged (policy: ask HPC). Keep milestone checkpoints and results in
`$GW1B_GROUP`/your home/GitHub; ask the instructor about the purge extension.

**Multi-node job hangs at start** — NCCL cannot find the interconnect. Set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` in
`slurm/train_multinode.sbatch` (ask HPC staff for the interface name), and make sure the sbatch used `--ntasks-per-node=<GPUs>`
without `--gpus-per-task`.

**Resumed run repeats/skip batches?** — it does not: the loader is deterministic in (seed, step). If you changed
`run.batch_size` or `data.seed` between runs, the sequence changes by design.

**Orbax: "checkpoint already exists" / partial checkpoint after a kill** — delete the incomplete `<step>` directory
(it has a `.orbax-checkpoint-tmp` marker) and resubmit.

**I need a package that is not installed** — everyone shares one environment; ask the instructor to add it to
`env/requirements.in` and rebuild. For a quick experiment, `pip install --target ~/mypkgs <pkg>` and
`sys.path.insert(0, "~/mypkgs")` works inside the container (user site-packages are disabled on purpose).
