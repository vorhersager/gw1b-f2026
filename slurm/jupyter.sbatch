#!/bin/bash
# JupyterLab on a GPU node — submitted by `gw1b jupyter` (partition/GPUs/time come from the wrapper).
#SBATCH --job-name=gw1b-jupyter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@60
set -euo pipefail
source "$GW1B_HOME/gw1b.env"

user_dir="$GW1B_SCRATCH/users/$USER"
mkdir -p "$user_dir/jobs" "$user_dir/runs"
pick_port() {  # a random port in 8800-9999 that nothing on this (shared) node listens on
    local p i
    for i in $(seq 1 30); do
        p=$((8800 + RANDOM % 1200))
        if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then echo "$p"; return; fi
    done
    echo $((8800 + RANDOM % 1200))
}
port="$(pick_port)"
token="$(python3 -c 'import secrets; print(secrets.token_hex(24))' 2>/dev/null || od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
node="$(hostname -s)"
info="$user_dir/jobs/jupyter-$SLURM_JOB_ID.json"
url="http://localhost:${GW1B_LOCAL_PORT:-8888}/lab?token=$token"
colab_url="http://localhost:${GW1B_LOCAL_PORT:-8888}/?token=$token"
tunnel="ssh -N -L ${GW1B_LOCAL_PORT:-8888}:$node:$port -L 6006:$node:6006 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 $USER@$GW1B_LOGIN_HOST"

cat > "$info" <<EOF
{"job_id": "$SLURM_JOB_ID", "node": "$node", "port": $port, "token": "$token",
 "local_port": ${GW1B_LOCAL_PORT:-8888}, "tunnel": "$tunnel", "url": "$url", "colab_url": "$colab_url",
 "started": "$(date -Is)", "gpus": "${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-?}}", "partition": "$SLURM_JOB_PARTITION"}
EOF

cat <<EOF
=====================================================================
 GW1B JupyterLab is starting on $node (job $SLURM_JOB_ID, port $port)

 1. On YOUR LAPTOP open a terminal and run (keep it open):
      $tunnel
 2. Then open in your browser:            $url
    or in Colab: Connect ▾ -> "Connect to a local runtime" -> paste:
      $colab_url
 3. When you are done:  gw1b cancel $SLURM_JOB_ID
=====================================================================
EOF

cd "${GW1B_JUPYTER_ROOT:-$HOME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

# Every notebook gets its own kernel and all of them share this job's GPU(s). JAX would otherwise grab 75 % of
# the GPU memory in the FIRST kernel that touches it, and the second notebook dies with CUDA_ERROR_OUT_OF_MEMORY.
# Allocate on demand instead (batch jobs keep JAX's default preallocation: one process per GPU there).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --ServerApp.allow_origin / allow_credentials: what Google Colab needs to talk to a local runtime.
# --ip=0.0.0.0 + random token: reachable from the login node through the ssh tunnel only.
# JupyterLab takes 20-90 s to come up in the container; `gw1b jupyter` waits for the port before printing the tunnel line.
echo "$(date +%T) starting JupyterLab (first start can take a minute or two) ..."
exec "$GW1B_HOME/bin/gw1b-exec" jupyter lab --no-browser --ip=0.0.0.0 --port="$port" \
    --ServerApp.token="$token" --ServerApp.port_retries=0 \
    --ServerApp.allow_origin='https://colab.research.google.com' --ServerApp.allow_credentials=True \
    --ServerApp.allow_remote_access=True --ServerApp.root_dir="${GW1B_JUPYTER_ROOT:-$HOME}" \
    --ServerApp.terminado_settings='{"shell_command": ["/bin/bash", "-l"]}'
