# Shared helpers for the Team 3 launch scripts. Sourced, not run.
# Every script prints its `gw1b train` commands; add --submit to actually queue them.
set -euo pipefail

SUBMIT=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--submit" ]]; then SUBMIT=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# Paired seeds: model init (run.seed) and data order (data.seed) move together.
seed_args() { echo "--set run.seed=$1 --set data.seed=$1"; }

emit() {  # emit <gw1b train args...>
  if [[ $SUBMIT -eq 1 ]]; then
    echo "+ gw1b train $*"
    # shellcheck disable=SC2068
    gw1b train $@
  else
    echo "gw1b train $*"
  fi
}

finish() {
  if [[ $SUBMIT -eq 0 ]]; then
    echo
    echo "# dry run: nothing submitted. Re-run with --submit to queue these jobs."
  fi
}

scale_lr() {  # scale_lr <lr> <factor>: e.g. 6e-4 x 0.5 -> 3e-4 (short form, used in run names)
  awk -v x="$1" -v f="$2" 'BEGIN { printf "%.2e\n", x * f }' | sed -E 's/\.?0+e/e/; s/e([-+]?)0*([0-9])/e\1\2/; s/e\+/e/'
}
