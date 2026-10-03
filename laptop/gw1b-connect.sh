#!/usr/bin/env bash
# One command from your laptop (macOS / Linux / WSL) to get a GPU JupyterLab on Pegasus + the tunnel:
#
#     ./gw1b-connect.sh <netid> [gw1b jupyter options]        e.g.  ./gw1b-connect.sh jsmith --gpus 1 --time 3:00:00
#
# It (1) asks Pegasus to start (or reuse) your Jupyter job, (2) opens the ssh tunnel and keeps it open,
# (3) prints the URL for JupyterLab and for Colab ("Connect to a local runtime").
# Press Ctrl-C to close the tunnel (the job keeps running until its time limit or `gw1b cancel`).
# Requires: ssh, and VPN if you are off campus. You will be asked for your GW password / 2FA.
set -euo pipefail
GW1B_HOME_REMOTE="${GW1B_HOME_REMOTE:-/scratch/gw1b-class/group/gw1b-f2026}"     # <- instructor: set once
LOGIN_HOST="${GW1B_LOGIN_HOST:-pegasus.arc.gwu.edu}"
LOCAL_PORT="${GW1B_LOCAL_PORT:-8888}"

[[ $# -ge 1 ]] || { echo "usage: $0 <netid> [gw1b jupyter options]"; exit 1; }
netid="$1"; shift
# Reuse one ssh connection for both steps (one password/2FA prompt instead of two)
ctl=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/gw1b-%r@%h:%p" -o ControlPersist=15m -o ServerAliveInterval=60)
mkdir -p "$HOME/.ssh"

echo "[gw1b] connecting to $LOGIN_HOST as $netid ..."
out="$(ssh -tt "${ctl[@]}" "$netid@$LOGIN_HOST" \
      "bash -lc 'source $GW1B_HOME_REMOTE/activate.sh >/dev/null 2>&1; gw1b jupyter --local-port $LOCAL_PORT $*'" | tr -d '\r')" || true
echo "$out"
read -r fwd1 fwd2 target <<< "$(printf '%s\n' "$out" | sed -n 's/.*ssh -N -L \([^ ]*\) -L \([^ ]*\) \([^ ]*\).*/\1 \2 \3/p' | head -1)"
[[ -n "${target:-}" ]] || { echo "[gw1b] could not find the tunnel command in the output above"; exit 1; }
url="$(printf '%s\n' "$out" | grep -m1 -oE 'http://localhost:[0-9]+/lab\?token=[a-f0-9]+' || true)"

echo
echo "[gw1b] opening tunnel:  ssh -N -L $fwd1 -L $fwd2 $target"
echo "[gw1b] JupyterLab: $url"
echo "[gw1b] Colab: Connect ▾ -> Connect to a local runtime -> ${url/\/lab?/\/?}"
echo "[gw1b] (Ctrl-C closes the tunnel; the job keeps running)"
( sleep 3; { command -v open >/dev/null && open "$url"; } || { command -v xdg-open >/dev/null && xdg-open "$url"; } || true ) >/dev/null 2>&1 &
exec ssh -N "${ctl[@]}" -L "$fwd1" -L "$fwd2" "$target"
