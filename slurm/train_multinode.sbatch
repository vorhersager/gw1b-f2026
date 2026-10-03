#!/bin/bash
# Multi-node training for the final GW-1B run (one JAX process per GPU, wired together by Slurm).
#   sbatch --nodes=2 --ntasks-per-node=8 --gres=gpu:a100:8 --partition=gpu --time=2-00:00:00 \
#          slurm/train_multinode.sbatch --config configs/gw1b_1p15b.yaml
# or:   gw1b run --nodes 2 --gpus 8 --gpu-type a100 --time 1-00:00:00 --chain 5 \
#              -m gw1b.train --config configs/gw1b_1p15b.yaml
#SBATCH --job-name=gw1b-train
#SBATCH --cpus-per-task=6
#SBATCH --signal=B:TERM@300
set -euo pipefail
source "$GW1B_HOME/gw1b.env"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[gw1b] job $SLURM_JOB_ID: $SLURM_JOB_NUM_NODES node(s) x $SLURM_NTASKS_PER_NODE task(s), nodes: $SLURM_JOB_NODELIST"
# NCCL over the cluster interconnect (HPC, 10/1): V100/A100 nodes share a 100 Gb/s EDR InfiniBand fabric,
# HCA mlx5_0 with IPoIB interface ib0 (10 GbE eno0 as fallback). GPUDirect RDMA's kernel module is not loaded
# by default, so NCCL stages through host memory — fine for a 1.15B model. Do not mix the new Blackwell nodes
# (separate NDR fabric) with V100/A100 nodes in one job: that traffic would go over Ethernet.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${GW1B_NCCL_SOCKET_IFNAME:-ib0}}
export NCCL_IB_HCA=${NCCL_IB_HCA:-${GW1B_NCCL_IB_HCA:-mlx5_0}}
export GW1B_DISTRIBUTED=1
# Each srun task = one process = one GPU (jax.distributed picks GPU #SLURM_LOCALID). Do NOT add
# --gpus-per-task here: every task must see all GPUs of its node.
srun --kill-on-bad-exit=1 "$GW1B_HOME/bin/gw1b-exec" python -m gw1b.train "$@"
