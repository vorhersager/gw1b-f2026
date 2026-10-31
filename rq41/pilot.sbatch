#!/usr/bin/env bash
# sbatch rq41/pilot.sbatch ABS_CHECKOUT NEW_ABS_OUTPUT
#SBATCH --job-name=team4-rq41-real-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
[[ $# = 2 ]] || { echo 'usage: pilot.sbatch ABS_CHECKOUT NEW_ABS_OUTPUT' >&2; exit 2; }
checkout="$(cd "$1" && pwd -P)"
[[ "$2" = /* ]] || { echo 'Output must be absolute' >&2; exit 2; }
cd "$checkout"
export GW1B_HOME="$checkout"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cuda
exec "$checkout/bin/gw1b-exec" python -u -m gw1b.rq41_pilot --checkout "$checkout" --out "$2"
