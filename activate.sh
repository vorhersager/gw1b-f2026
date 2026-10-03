# Source this on a Pegasus login node:   source /SEAS/groups/gw1b-class/gw1b-f2026/activate.sh
# (or run `gw1b setup` once to add that line to your ~/.bashrc)
_gw1b_activate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GW1B_HOME="$_gw1b_activate_dir"
source "$GW1B_HOME/gw1b.env"
case ":$PATH:" in *":$GW1B_HOME/bin:"*) ;; *) export PATH="$GW1B_HOME/bin:$PATH" ;; esac
export PYTHONPATH="$GW1B_HOME${PYTHONPATH:+:$PYTHONPATH}"
# per-user working directories on scratch
export GW1B_USER_DIR="$GW1B_SCRATCH/users/$USER"
mkdir -p "$GW1B_USER_DIR/runs" "$GW1B_USER_DIR/jobs" "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || true
unset _gw1b_activate_dir
if [[ $- == *i* ]]; then
    echo "GW1B environment active (v$GW1B_ENV_VERSION). Try:  gw1b help"
fi
