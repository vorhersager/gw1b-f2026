#!/bin/bash
# Generic single-node job — submitted by `gw1b run`. The command line to run is passed as arguments:
#   gw1b run --gpus 4 --time 12:00:00 -m gw1b.train --config configs/100m.yaml
#SBATCH --job-name=gw1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@120
set -euo pipefail
source "$GW1B_HOME/gw1b.env"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[gw1b] job $SLURM_JOB_ID on $(hostname -s), partition $SLURM_JOB_PARTITION, $(date -Is)"
echo "[gw1b] command: python $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "[gw1b] no GPU on this node"
# forward SIGTERM (time limit approaching) to python so the trainer checkpoints and exits cleanly
trap 'echo "[gw1b] SIGTERM: asking python to stop"; kill -INT "$child" 2>/dev/null || true' TERM
"$GW1B_HOME/bin/gw1b-exec" python "$@" &
child=$!
wait "$child"
status=$?
echo "[gw1b] finished with exit code $status at $(date -Is)"
exit $status
