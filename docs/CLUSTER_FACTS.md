# Pegasus facts (what we know, what to verify) and what they mean for GW1B

## Verified from public GW IT documentation (September 2026)

| item | value | source |
|---|---|---|
| login host | `pegasus.arc.gwu.edu` (ssh with NetID; VPN off campus; 2FA available) | GW CBI HPC guide |
| scheduler / OS | Slurm; CentOS 8 (per IT page — may have moved to Rocky/RHEL 8/9) | it.gwu.edu/hpc-pegasus |
| GPU nodes | 16 nodes × 2 V100 (16 GB); 22 nodes × 4 V100 SXM2 16 GB; **2 nodes × 8 A100 80 GB PCIe**; 2 Grace Hopper (ARM CPU + H100 96 GB); L40S nodes added in the expansion | it.gwu.edu/hpc-pegasus, it.gwu.edu/hpc-expansion |
| CPU nodes | 164 × 40 cores/192 GB, 54 × 384 GB, 2 × 3 TB, 6 high-throughput | it.gwu.edu/hpc-pegasus |
| partitions (2020 slides) | `defq` (CPU), `small-gpu` (2×V100), `large-gpu` (4×V100), `debug-gpu` / `debug-cpu` (4 h), `short` (1 day), `tiny` (8 h), `highMem`, `highThru` | MIA lab HPC intro (2020) |
| storage | home `/<SCHOOL>/home/<netid>` 25 GB; group `/<SCHOOL>/groups/<group>` 250 GB; Lustre scratch `/lustre/groups/<group>` purged at the start of each month; apps in `/c1/apps` | GW CBI HPC guide |
| filesystems | NFS (Qumulo, 2 PB replicated) + Lustre scratch (Lenovo DSS, 2 PB); "neither is long-term storage" | it.gwu.edu/hpc-pegasus |

## Must be verified once (`slurm/discover_cluster.sh` → `gw1b.env`)

* Partition names for the **A100**, **L40S** and **GH200** nodes (not published) and their time limits.
* Whether `sbatch` needs `--account` (class allocation) and the GRES syntax (`gpu:1` vs `gpu:a100:1`).
* **NVIDIA driver version on the GPU nodes**: the CUDA 12 pip wheels need ≥ 525. If older, options are the
  `cuda-compat` forward-compatibility package inside the image (datacenter GPUs only) or a driver update by HPC staff.
* **Apptainer/Singularity** present on compute nodes (and whether `--fakeroot` builds are allowed on the login node).
  If absent: `env/build_venv.sh` (identical package set, no container).
* Outbound internet from **compute nodes** (assume none: data is downloaded on login nodes into `$HF_HOME`).
* Scratch purge policy / extension for the class, and the actual quota of the group directory.
* Whether Open OnDemand exists for Pegasus (`gw1b kernel` makes the environment selectable there if so).

## What the hardware means for the project

**V100 (16 GB, 2019)** — plenty for the 50M–350M proxy ladder, but: no bf16 tensor cores (use
`model.dtype=float32`; ~15 TFLOP/s), 16 GB per GPU (FSDP is on by default, so a 350M model with AdamW fits on
1–2 V100s), and cuDNN flash attention is unavailable (XLA attention is used automatically).

**A100 80 GB PCIe (2 nodes × 8)** — the machines for everything above 200M and for GW-1B itself: bf16 (312 TFLOP/s),
flash attention, 80 GB. PCIe (no NVLink) makes all-gathers slower than on SXM boxes, so expect 25–40 % MFU with FSDP.
There are only 16 of them for the whole university: queue early, use checkpoints and `--chain`.

**Grace Hopper (H100 96 GB, ARM host)** — fastest chips on the cluster, but the host is aarch64: the x86 container
does not run there. A second, arm64 build of `env/gw1b.def` would be needed (JAX ships aarch64 CUDA wheels). Not part of
the standard environment; a possible stretch goal for Team 3.

**L40S (48 GB)** — bf16 + flash attention, roughly 60 % of an A100 for training; good for 100M–350M proxies once the
partition name is known.

### Wall-clock estimates (35 % MFU; FLOPs = 6·N·D + attention term)

| config | tokens | FLOPs | 1×V100 f32 | 4×V100 f32 | 1×A100 | 8×A100 | 16×A100 | GPU-hours (A100) |
|---|---|---|---|---|---|---|---|---|
| 50m (50M) | 1B | 4.5e17 | 23 h | 6 h | 1.2 h | 0.1 h | 0.1 h | 1 |
| 100m (100M) | 2B | 1.7e18 | 84 h | 21 h | 4.2 h | 0.5 h | 0.3 h | 4 |
| 200m (201M) | 4B | 6.4e18 | 325 h | 81 h | 16 h | 2.0 h | 1.0 h | 16 |
| 350m (354M) | 7B | 1.9e19 | 951 h | 238 h | 48 h | 6.0 h | 3.0 h | 48 |
| gw1b_1p15b | 20B | 1.7e20 | — | — | 423 h | **53 h** | 26 h | 423 |
| gw1b_1p15b | 100B | 8.3e20 | — | — | 2115 h | **264 h (11 days)** | 132 h (5.5 days) | 2115 |

(`gw1b budget --config configs/<name>.yaml --tokens <N> --gpu a100 --n-gpus 8` reproduces any row.)

Reading of the table:
* The proxy ladder is cheap on A100s and feasible on V100s with patience; run V100 jobs for the 50M/100M points.
* GW-1B on the design-document budget of 80–100B tokens is **1–2 weeks of exclusive A100-node time** — realistic
  only if HPC staff reserve a node. 20–30B tokens (≈ Chinchilla-optimal 23B) is a 2–3 day run and the safer plan;
  the design doc's own budget ("100–175 TPU wall-clock hours") assumed a much faster TPU v6e host. Decide at the
  Week-8 design freeze with `gw1b budget` numbers in hand.
* Checkpoints every 500 steps (0.5B tokens, ~40 min) mean a killed job loses under an hour.

### Cloud-equivalent dollars

`run.gpu_hour_price_usd` (default $2/GPU-hour, roughly an on-demand A100) turns GPU-hours into the "$" every
experiment reports, so students can compare against the $10,000 framing even though Pegasus time is not billed.
