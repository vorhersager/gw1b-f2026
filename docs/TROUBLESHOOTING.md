# Troubleshooting

**`gw1b: command not found`** — `source /SEAS/groups/gw1b/gw1b-f2026/activate.sh` (or `gw1b setup` once, then open a new shell).

**`gw1b-exec: no environment found`** — neither the container (`$GW1B_SIF`) nor the venv (`$GW1B_VENV`) exists or
`apptainer` is not available. Instructor: `env/build_sif.sh` or `env/build_venv.sh`; if apptainer is a module, set
`GW1B_APPTAINER_MODULE` in `gw1b.env`.

**`sbatch: error: invalid partition` / `Invalid generic resource` / `Invalid account`** — the `VERIFY` lines in `gw1b.env` (`GW1B_PART_*`, `GW1B_GPU_*`,
`GW1B_ACCOUNT`, `GW1B_GRES`). Run `bash slurm/discover_cluster.sh`.

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

**Slow training on V100** — bf16 is emulated on Volta. Use `--set model.dtype=float32` (and expect ~15 TFLOP/s per GPU).

**`cudnn` attention error** — `--set model.attn_implementation=xla` (auto-selection picked cuDNN on an unsupported
shape/GPU). Please report the shape to the instructor.

**Jupyter: "job is PENDING (Resources)"** — no idle GPU of that type; wait, check `gw1b status`, or try `--gpu-type any` / fewer GPUs /
shorter `--time`. `gw1b status` shows free GPUs per partition.

**Jupyter connects but kernels die / `import gw1b` fails** — you are probably in an Open OnDemand or hosted-Colab kernel,
not the GW1B one. Use the URL printed by `gw1b jupyter`, or `gw1b kernel` and select *GW1B (JAX)* as the kernel.

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
