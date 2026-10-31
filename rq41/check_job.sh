#!/usr/bin/env bash
# Read-only login-node inspection; never submits/cancels a job or assumes completion.
set -euo pipefail
job=138631889
squeue -j "$job" -o '%.18i %.30j %.12T %.12M %.30R' || echo 'No active queue record; consult accounting below.'
sacct -j "$job" --format=JobID,JobName%35,State,Elapsed,ExitCode || echo 'Accounting unavailable; inspect logs below.'
scontrol show job "$job" || true  # finished jobs may have aged out
job_logs=/scratch/gw1b-class/users/g37693783/jobs
if [[ -d "$job_logs" ]]; then
    find "$job_logs" -maxdepth 1 -type f -name "*${job}*" -print -exec tail -n 60 {} \;
fi
run=/scratch/gw1b-class/users/g37693783/runs/team4-rq41-smoke-50m
if [[ -d "$run/checkpoints" ]]; then
    find "$run/checkpoints" -maxdepth 2 -type d -print
else
    echo "No checkpoint directory yet. Do not submit another training job."
fi
echo 'Directory names/log messages are provisional; the Slurm smoke job must verify an Orbax restore.'
