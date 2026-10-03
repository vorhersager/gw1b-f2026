# Pegasus facts (as told by HPC staff, 2026-09-29) and what they mean for GW1B

The public documentation the first version of this repository was built from is outdated (HPC staff said so
themselves). Everything below comes from their written answers to our questions; `gw1b-admin discover` re-checks
the machine-readable parts and writes them into `gw1b.env`.

## Confirmed by HPC staff

| item | value |
|---|---|
| login host | `pegasus.arc.gwu.edu` (NetID; VPN off campus) |
| partitions (`sinfo -s`) | `gpu` — **all** GPU nodes, 40 nodes `gpu[001-037,042,050-051]`, 7-day limit · `cpu*` — 153 CPU nodes, 14-day limit · `nano` — 4 CPU nodes, 30 min · `superChip` — 8 Grace Hopper GH200 nodes, 7 days · `viz`, `deus` (not for us) |
| choosing a GPU model | by GRES type, e.g. `--gres=gpu:a100:8`; HPC's example: `sbatch --partition gpu --time 24:00:00 --cpus-per-task=20 --mem=100G --gres=gpu:a100:8 myjob.sh`; add `--nodes 2` for 2 × 8 A100 |
| A100 nodes | `gpu050`, `gpu051`: 8 × A100 80 GB PCIe each; NVLink in **pairs** (GPU0–3, 1–2, 4–7, 5–6), PCIe otherwise, two NUMA domains (CPU 0–25 / 26–51); NIC `mlx5_0` on NUMA 0 |
| V100 nodes | 2-GPU and 4-GPU nodes; the 4 × V100 nodes have NVLink between all four |
| L40S / new nodes | L40S: no NVLink. **Coming in October: 6 RTX PRO 6000 Blackwell 96 GB nodes (2 × 4-GPU, 4 × 2-GPU), no NVLink** |
| NVIDIA drivers | V100: proprietary branch 575.57.08 → 580. A100/L40S/GH200: open branch, minimum 570.133.20, drifting upward (6xx series appearing). All ≥ 525 → the CUDA 12 wheels run everywhere. V100 = CUDA 12 max; newer GPUs also CUDA 13 |
| Apptainer | `module load apptainer` (1.3.0, newer version coming); **no subuid/subgid → no `--fakeroot` builds**; `--nv` works; bring images as SIF, or ask HPC to build |
| scratch | **Lustre is retired.** Scratch is GPFS at `/scratch/<group>` (e.g. `/scratch/gw1b-class`). Purge policy and quota: not yet answered |
| group storage | `/SEAS/groups/gw1b` requested (persistent); quota not yet answered |
| Unix group | `gw1b-class` — creating it and the directory layout is "not an issue" |
| compute-node internet | yes, compute nodes download freely (so data jobs run on `cpu` nodes, W&B can be online) |
| inter-node network | "completely open" between compute nodes; interconnect type/interface names not yet answered |
| Jupyter | a Jupyter (classic) app exists on Pegasus (Open OnDemand-style), works on all x86 nodes; **it loads kernels from `~/.local/share/jupyter/kernels` (confirmed 10/1)**, which is exactly where `gw1b kernel` installs the GW1B kernel; SSH port forwarding "generally works" → `gw1b jupyter` + tunnel (needed for Colab) |
| reservations | only single-day (demo-style) reservations normally; multi-day A100 reservations need a written proposal; partial-node reservations are being worked on; advice: submit flexible jobs (8, 4, 2 GPUs) rather than wait for a whole node |
| Globus | yes (Pegasus storage ↔ GW Box ↔ GW Google Drive; Globus Personal Connect for laptops) |

## Still open (asked 2026-09-30)

* Quota and purge policy of `/scratch/gw1b-class` and `/SEAS/groups/gw1b` (we asked for 2 TB scratch, 1 TB group).
* Whether `--account`/`--qos` are required (HPC's example had neither) and per-user GPU/job limits on `gpu`.
* Exact GRES type names for V100, L40S and the Blackwell nodes (`sinfo -o "%N %G"` once we have an account).
* Interconnect between GPU nodes (InfiniBand vs Ethernet; interface names for `NCCL_SOCKET_IFNAME`).
* Student accounts; the Jupyter-app walkthrough meeting; HPC's second reply of 10/1 ("More answers below", items 4+) still to be folded in here.

## What the hardware means for the project

**V100 (16 GB)** — the default for students (`--gpu-type v100`, or nothing): plenty for the 50M–350M proxy ladder,
but no bf16 tensor cores (use `model.dtype=float32`; ~15 TFLOP/s), 16 GB per GPU (FSDP is on by default), and no
cuDNN flash attention (XLA attention is used automatically). Ask for 4 to get a NVLink node (`--gpus 4`).

**A100 80 GB PCIe (2 nodes × 8, `--gpu-type a100`)** — everything above 200M and GW-1B itself: bf16 (312 TFLOP/s),
flash attention, 80 GB. NVLink only within pairs, so cross-GPU traffic is mostly PCIe; for a 1.15B model at 1M tokens
per step the FSDP all-gathers/reduce-scatters are ~10 % of step time and largely overlapped — fine. Tensor
parallelism would not be; we do not use it. `shard_params: false` (plain data parallel; the model + optimizer state
fit in 80 GB) is a knob to test in the pilot. There are only 16 A100s for the whole university and they run around
the clock: queue early, checkpoint often (`ckpt_every: 500`), `--chain`, and be able to run on 4 GPUs.

**RTX PRO 6000 Blackwell 96 GB (from October; 2 × 4-GPU + 4 × 2-GPU nodes)** — spec sheet 1 PFLOP/s FP16 tensor
(sparse), so ~500 TFLOP/s dense ≈ 1.6 × A100, 96 GB, PCIe 5 only, GDDR7. Once online they are a real alternative
for the production run on a 4-GPU node (`--gpu-type rtx6000 --gpus 4`); measure with `gw1b doctor --gpu` first.
Requires the CUDA 12.8+ runtime in the wheels (JAX 0.10 has it).

**Grace Hopper GH200 (8 single-GPU nodes, `superChip`)** — H100 96 GB, fastest chips on the cluster, but the host is
aarch64: the x86 image does not run there. An arm64 build of `env/Dockerfile` (`platforms: linux/arm64` in the
build-image workflow; JAX ships aarch64 CUDA wheels) would make one GH200 ≈ 3 A100s. Stretch goal for Team 3.

**L40S (48 GB)** — bf16 + flash attention, roughly 60 % of an A100 for training; good for 100M–350M proxies.

### Wall-clock estimates (35 % MFU; FLOPs = 6·N·D + attention term)

| config | tokens | FLOPs | 1×V100 f32 | 4×V100 f32 | 1×A100 | 4×A100 | 8×A100 | 16×A100 | GPU-hours (A100) |
|---|---|---|---|---|---|---|---|---|---|
| 50m (50M) | 1B | 4.5e17 | 23 h | 6 h | 1.2 h | 0.3 h | 0.1 h | 0.1 h | 1 |
| 100m (100M) | 2B | 1.7e18 | 84 h | 21 h | 4.2 h | 1.1 h | 0.5 h | 0.3 h | 4 |
| 200m (201M) | 4B | 6.4e18 | 325 h | 81 h | 16 h | 4.0 h | 2.0 h | 1.0 h | 16 |
| 350m (354M) | 7B | 1.9e19 | 951 h | 238 h | 48 h | 12 h | 6.0 h | 3.0 h | 48 |
| gw1b_1p15b | 20B | 1.7e20 | — | — | 423 h | 106 h | **53 h** | 26 h | 423 |
| gw1b_1p15b | 100B | 8.3e20 | — | — | 2115 h | 529 h | **264 h (11 days)** | 132 h (5.5 days) | 2115 |

(`gw1b budget --config configs/<name>.yaml --tokens <N> --gpu a100 --n-gpus 8` reproduces any row.)

Reading of the table:
* The proxy ladder is cheap on A100s and feasible on V100s with patience; run V100 jobs for the 50M/100M points.
* GW-1B on the design-document budget of 80–100B tokens is **1–2 weeks of exclusive A100-node time** — HPC will not
  reserve that without a proposal. 20–30B tokens (≈ Chinchilla-optimal 23B) is a 2–3 day run on 8 A100s, 4–5 days on
  4, and the safe plan. The `gpu` partition allows 7-day jobs; chain 1–2 day jobs anyway (they backfill sooner and
  survive node failures). Decide at the Week-8 design freeze with `gw1b budget` numbers in hand.
* Checkpoints every 500 steps (0.5B tokens, ~40 min on 8 A100s) mean a killed job loses under an hour.

### Cloud-equivalent dollars

`run.gpu_hour_price_usd` (default $2/GPU-hour, roughly an on-demand A100) turns GPU-hours into the "$" every
experiment reports, so students can compare against the $10,000 framing even though Pegasus time is not billed.
