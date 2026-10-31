#!/usr/bin/env bash
# User submits manually after reviewing training status/logs; this script never submits jobs.
# sbatch rq41/smoke.sbatch CHECKOUT RUN NEW_OUTPUT [STEP]
#SBATCH --job-name=team4-rq41-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
[[ $# -ge 3 && $# -le 4 ]] || { echo 'usage: smoke.sbatch CHECKOUT RUN NEW_OUTPUT [STEP]' >&2; exit 2; }
: "${SLURM_JOB_ID:?Run this script through Slurm}"
checkout="$(cd "$1" && pwd -P)"
[[ -f "$checkout/gw1b/rq41_records.py" ]] || { echo 'RQ4.1 development checkout missing' >&2; exit 1; }
cd "$checkout"
# gw1b-exec PREPENDS GW1B_HOME, so override it as well as the working directory.
export GW1B_HOME="$checkout"
exec "$checkout/bin/gw1b-exec" bash "$checkout/rq41/eval_job.sh" "$2" "$3" "${4:-}"
