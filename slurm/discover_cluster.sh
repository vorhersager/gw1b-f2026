#!/usr/bin/env bash
# One-time discovery of Pegasus facts we could not get from public docs. Run on a login node:
#     bash slurm/discover_cluster.sh > docs/CLUSTER_FACTS.generated.md
# then fill in the VERIFY items of gw1b.env (partition names, gres syntax, account).
set -uo pipefail
GW1B_HOME="${GW1B_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$GW1B_HOME/gw1b.env" 2>/dev/null || true
section() { printf '\n## %s\n\n```\n' "$1"; }
endsection() { printf '```\n'; }

echo "# Pegasus facts discovered on $(date -Is) by $USER@$(hostname -f)"

section "OS / kernel / login node"
cat /etc/os-release 2>/dev/null | head -3; uname -r; echo "user: $USER  groups: $(id -Gn)"
endsection

section "Slurm partitions (name, avail, time limit, nodes, node list, GRES, memory MB, CPUs)"
sinfo -o "%-16P %-6a %-12l %-6D %-30N %-24G %-9m %-5c" 2>&1
endsection

section "GPU nodes by type (grep for a100 / l40 / h100 / v100 in GRES and features)"
sinfo -N -o "%-14N %-16P %-30G %-40f %-9m %-5c" 2>&1 | grep -iE "gpu|a100|l40|h100|v100|NODELIST" | sort -u
endsection

section "Partition details (time limits, default/max, QOS)"
scontrol show partition 2>&1 | grep -E "PartitionName|MaxTime|DefaultTime|MaxNodes|QoS|AllowAccounts|AllowGroups|State=" 
endsection

section "My Slurm associations (does sbatch need --account / --qos?)"
sacctmgr show assoc where user="$USER" format=cluster,account,partition,qos,maxjobs,grptres%30 2>&1
endsection

section "Container runtime"
for c in apptainer singularity; do printf '%-12s %s\n' "$c" "$(command -v $c 2>/dev/null || echo 'not in PATH')"; done
if ! command -v module >/dev/null 2>&1; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh 2>/dev/null
  [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash 2>/dev/null
fi
module avail 2>&1 | grep -iE "apptainer|singularity|cuda|cudnn|nccl|python|anaconda|miniconda|jupyter|openmpi|nvhpc" | head -40
for c in apptainer singularity; do
  if command -v $c >/dev/null 2>&1; then $c --version; $c build --help 2>&1 | grep -q fakeroot && echo "$c supports --fakeroot flag (may still need admin enablement)"; fi
done
endsection

section "Storage (group dir, scratch, home quota)"
for d in "$GW1B_GROUP" "$GW1B_SCRATCH" "$HOME" /lustre /scratch; do
  [[ -e "$d" ]] && { printf '%s: ' "$d"; df -h "$d" 2>/dev/null | tail -1; } || echo "$d: does not exist"
done
quota -s 2>/dev/null | head -20
command -v mmlsquota >/dev/null && mmlsquota -u "$USER" 2>/dev/null | head; command -v lfs >/dev/null && lfs quota -h -u "$USER" /lustre 2>/dev/null | head
endsection

section "Python / tools on the login node"
for t in python3 uv git curl tmux nvidia-smi; do printf '%-10s %s\n' "$t" "$(command -v $t 2>/dev/null || echo -)"; done
python3 --version 2>&1
endsection

section "Outbound internet from the login node (HF Hub, PyPI, GitHub)"
for u in https://huggingface.co https://pypi.org https://github.com https://download.pytorch.org/whl/cpu; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$u" 2>/dev/null) || code=ERR; echo "$u -> $code"
done
endsection

section "GPU node check (driver version decides CUDA 12 support: need >= 525) — via a 2-minute srun on the debug partition"
timeout 300 srun --partition="${GW1B_PART_GPU:-gpu}" ${GW1B_ACCOUNT:+--account=$GW1B_ACCOUNT} --gres="${GW1B_GRES:-gpu}:${GW1B_GPU_DEFAULT:-v100}":1 --time=00:03:00 --pty \
  bash -c 'hostname -s; cat /etc/os-release | head -2; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv; \
           for c in apptainer singularity; do printf "%-12s %s\n" $c "$(command -v $c 2>/dev/null || echo not-in-PATH)"; done; \
           for u in https://huggingface.co https://pypi.org; do printf "%s -> %s\n" $u "$(curl -s -o /dev/null -w %{http_code} --max-time 8 $u || echo ERR)"; done' 2>&1
endsection
