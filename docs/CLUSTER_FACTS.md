# Pegasus facts (as told by HPC staff, 2026-09-28 … 10-01) and what they mean for GW1B

The public documentation the first version of this repository was built from is outdated (HPC staff said so
themselves). Everything below comes from their written answers to our questions; `gw1b-admin discover` re-checks
the machine-readable parts and writes them into `gw1b.env`.

## Confirmed by HPC staff

| item | value |
|---|---|
| login host | `pegasus.arc.gwu.edu` (NetID; GW network or VPN; **SSH public key** registered via the HPC Access Request form; **2FA** code on every login, set up at the first login with the single-use code HPC emails) |
| accounts | created one by one from each student's access request (no batch process; a third of the class had applied by 10/1); HPC runs an onboarding session for the class; accounts last until the end of the semester unless told otherwise; instructor + TA accounts exist since 10/1 |
| partitions (`sinfo -s`) | `gpu` — **all** GPU nodes, 40 nodes `gpu[001-037,042,050-051]`, 7-day limit · `cpu*` — 153 CPU nodes, 14-day limit · `nano` — 4 CPU nodes, 30 min · `superChip` — 8 Grace Hopper GH200 nodes, 7 days · `viz`, `deus` (not for us) |
| choosing a GPU model | by GRES type, e.g. `--gres=gpu:a100:8`; HPC's example: `sbatch --partition gpu --time 24:00:00 --cpus-per-task=20 --mem=100G --gres=gpu:a100:8 myjob.sh`; add `--nodes 2` for 2 × 8 A100 |
| A100 nodes | `gpu050`, `gpu051`: 8 × A100 80 GB PCIe each; NVLink in **pairs** (GPU0–3, 1–2, 4–7, 5–6), PCIe otherwise, two NUMA domains (CPU 0–25 / 26–51); NIC `mlx5_0` on NUMA 0 |
| V100 nodes | 2-GPU and 4-GPU nodes; the 4 × V100 nodes have NVLink between all four |
| L40S / new nodes | L40S: no NVLink. **Coming in October: 6 RTX PRO 6000 Blackwell 96 GB nodes (2 × 4-GPU, 4 × 2-GPU), no NVLink** |
| NVIDIA drivers | V100: proprietary branch 575.57.08 → 580. A100/L40S/GH200: open branch, minimum 570.133.20, drifting upward (6xx series appearing). All ≥ 525 → the CUDA 12 wheels run everywhere. V100 = CUDA 12 max; newer GPUs also CUDA 13 |
| Apptainer | `module load apptainer` (1.3.0, newer version coming); **no subuid/subgid → no `--fakeroot` builds**; `--nv` works; bring images as SIF, or ask HPC to build |
| scratch | **Lustre is retired.** Scratch is GPFS at `/scratch/<group>` (e.g. `/scratch/gw1b-class`), 2 PB shared (79 % used on 9/30). Purges are age-based, ad hoc when usage grows, always announced in advance, exceptions reviewed — our files are "unlikely to be impacted" |
| group storage | **`/SEAS/groups/gw1b-class`** (created by HPC on 10/3; before that `/scratch/gw1b-class/group` was the stand-in). Quotas are per school (SEAS, shared), not per group; our ~1 TB "should not be problematic". Home quotas are per user |
| Unix group | **`MG-gw1b-class`** (what `groups` prints on Pegasus; HPC called it `gw1b-class` in email) — owns `/scratch/gw1b-class` (`root:MG-gw1b-class`, mode 2770, created 10/1) |
| compute-node internet | yes, compute nodes download freely (so data jobs run on `cpu` nodes, W&B can be online) |
| inter-node network | V100/A100 nodes: **100 Gb/s EDR InfiniBand** (HCA `mlx5_0`, IPoIB `ib0`) + 10 GbE (`eno0`); traffic between compute nodes open; **GPUDirect RDMA not loaded by default**, no GPUDirect Storage; the Blackwell nodes get 25 GbE + a separate 400 Gb/s NDR fabric — mixed old/new jobs fall back to Ethernet |
| Jupyter | Open OnDemand at **https://ood.arc.gwu.edu** → *Jupyter Notebook Pegasus* app (classic Jupyter; review every field before Launch — "the branching logic is a little odd"); works on all x86 nodes; **loads kernels from `~/.local/share/jupyter/kernels`** (10/1) = where `gw1b kernel` installs ours; SSH port forwarding "generally works" → `gw1b jupyter` + tunnel (needed for Colab) |
| Slurm policy | **no `--account`, no QOS, no manual priority changes, no per-user limits, no preemption**; fair-share lowers the priority of heavy recent users; September dequeue times for 8-GPU jobs stayed under 2 days; most users submit 1–2-GPU jobs; 2–3-day jobs dequeue more slowly |
| A100 nodes, continued | 1 TB RAM; GPUs 0–3 on socket 0, 4–7 on socket 1; HPC's recipe for half a node with two NVLink pairs: `--sockets-per-node=1 --gpus-per-socket=a100:4 --gres-flags=enforce-binding` (= `gw1b run --gpus 4 --gpu-type a100 --one-socket`) |
| Grace Hopper | 8 `superChip` nodes, one GH200 each (H100 96 GB + Grace CPU, shared memory), **less used than the A100s** — HPC suggests them; aarch64 host → needs the arm64 image (`gw1b-admin build arm64`); the OOD Jupyter app does not support them yet |
| reservations | only single-day (demo-style) reservations normally; a proposal "can be taken under consideration" but the goal is standard scheduling; partial-node reservations are being worked on; advice: flexible jobs (8, 4, 2 GPUs), 1-day chunks; **class reservations for the Friday interactive sessions "may be feasible" — under review** |
| support | https://github.com/gwuniversity/hpc-onboarding (tutorials); Zoom office hours Tue/Thu 12:30–14:30; tickets hpchelp@gwu.edu |
| Globus | yes (Pegasus storage ↔ GW Box ↔ GW Google Drive; Globus Personal Connect for laptops) |

## Still open

* GRES type names: `v100` is confirmed (`gw1b doctor`, 10/2: the smoke test ran on `gpu023`, a Tesla V100-SXM2-16GB with driver
  575.57.08, compute capability 7.0). Still to read off `sinfo -o "%N %G"`: L40S, GH200 and the Blackwell nodes (and their arrival date).
* HPC's decision on Friday-lab reservations, and whether a written proposal for a partial A100 reservation is worth sending.
* Student accounts: everyone must submit the access request form with an SSH key; HPC's onboarding session before the first Pegasus lab.

## What the hardware means for the project

**V100 (16 GB)** — the default for students (`--gpu-type v100`, or nothing): plenty for the 50M–350M proxy ladder,
but no bf16 tensor cores (`model.dtype: auto` picks float32 there; ~15 TFLOP/s), 16 GB per GPU (FSDP is on by default), and no
cuDNN flash attention (XLA attention is used automatically). Ask for 4 to get a NVLink node (`--gpus 4`).

**A100 80 GB PCIe (2 nodes × 8, `--gpu-type a100`)** — everything above 200M and GW-1B itself: bf16 (312 TFLOP/s),
flash attention, 80 GB, 1 TB host RAM. NVLink only within pairs, so cross-GPU traffic is mostly PCIe; for a 1.15B
model at 1M tokens per step the FSDP all-gathers/reduce-scatters are ~10 % of step time and largely overlapped —
fine. Tensor parallelism would not be; we do not use it. `shard_params: false` (plain data parallel; the model +
optimizer state fit in 80 GB) is a knob to test in the pilot. There are only 16 A100s for the whole university and
they run around the clock: queue early, checkpoint often (`ckpt_every: 500`), chain **1-day** jobs (8-GPU jobs
dequeued within 2 days in September; longer jobs wait longer), and be able to run on 4 GPUs — `--gpus 4 --gpu-type
a100 --one-socket` gets the four GPUs of one socket (two NVLink pairs, one NUMA domain); `run.batch_size` is the global
batch, so the 1M-token step is kept automatically (`optim.grad_accum: auto` doubles the accumulation).

**RTX PRO 6000 Blackwell 96 GB (from October; 2 × 4-GPU + 4 × 2-GPU nodes)** — spec sheet 1 PFLOP/s FP16 tensor
(sparse), so ~500 TFLOP/s dense ≈ 1.6 × A100, 96 GB, PCIe 5 only, GDDR7. Once online they are a real alternative
for the production run on a 4-GPU node (`--gpu-type rtx6000 --gpus 4`); measure with `gw1b doctor --gpu` first.
Requires the CUDA 12.8+ runtime in the wheels (JAX 0.10 has it).

**Grace Hopper GH200 (8 single-GPU nodes, `superChip`, `--gpu-type gh200`)** — H100 96 GB on a Grace CPU with a
shared-memory design; the fastest chips on the cluster and, per HPC, the least contended. One GH200 ≈ 3 A100s:
20B tokens in ~5.5 days on one node at 35 % MFU (`gw1b budget --gpu gh200`), and the 7-day limit allows it in one
job. The host is aarch64, so it needs the arm64 image: run the build-image workflow with `arm64 = true`, set
`GW1B_IMAGE_ARM64` in `gw1b.env`, `gw1b-admin build arm64`, then `gw1b doctor --gpu --gpu-type gh200`. Multi-node
GH200 jobs would go over the same EDR fabric (`--nodes 2`), untested. The OOD Jupyter app cannot run there yet.

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

The "$" every experiment reports is GPU-hours × a notional public-cloud on-demand price per GPU type (V100 $1,
A100 $2, L40S $1.5, RTX 6000 $2.5, H100/GH200 $3.5 per GPU-hour; `run.gpu_hour_price_usd` overrides it), so
students can compare against the $10,000 framing even though Pegasus time is not billed. GPU-hours = wall-clock
time inside the training loop × GPUs in the job (queue time excluded; resumed runs carry their hours over).
