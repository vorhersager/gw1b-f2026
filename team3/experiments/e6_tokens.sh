#!/usr/bin/env bash
# Team 3, E6: the data axis. How much do 20 and 30 tokens/parameter improve on 10, at 50M and 100M, for each
# optimizer? Decides how long the production run trains. Also the one-run ladder of RQ 3.2 (H2).
#
#   bash team3/experiments/e6_tokens.sh trunks <50m|100m> <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>     # 2 runs
#   bash team3/experiments/e6_tokens.sh branches <50m|100m> <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>   # 4 runs
#   ... add --submit to queue.
#
# Trunk: one WSD run per optimizer to 30 tokens/parameter, cooling down over its last 20%. It keeps the checkpoints
# where a 10 and a 20 tokens/parameter WSD run start their cooldown (k and 2k steps; the trunk's own cooldown starts
# at 3k), so each branch copies one and runs only its cooldown (team3/experiments/branch.py). A branch then equals
# a standalone WSD run of its length, and three results cost about 1.2 runs. Branch once the trunk is past 2k steps.
# The trunk keeps every k/2 steps, so 5 and 15 (and 25 at 50M) tokens/parameter can be branched later too.
# The 10 tokens/parameter branch repeats a run of E1/E1b with the WSD schedule; cosine at the same length is E1/E1b.
source "$(dirname "$0")/_common.sh"
[[ $# -eq 6 ]] || { echo "usage: $0 trunks|branches 50m|100m <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2; }
STAGE="$1" SIZE="$2"
RUNS="$GW1B_SCRATCH/teams/team3/runs"
case "$SIZE" in   # tokens at 10/20/30 tokens per parameter; k = cooldown start of the 10 tokens/parameter run
  50m)  CONFIG=configs/team3/e1_50m.yaml;   TOKS=(0.5B 1B 1.5B); NTOK=(500000000 1000000000 1500000000)
        K=1526; TRUNK_TIME="--time 16:00:00";            BRANCH_TIME=4:00:00 ;;
  100m) CONFIG=configs/team3/e1b_100m.yaml; TOKS=(1B 2B 3B);     NTOK=(1000000000 2000000000 3000000000)
        K=3052; TRUNK_TIME="--time 1-00:00:00 --chain 2"; BRANCH_TIME=8:00:00 ;;
  *) echo "size must be 50m or 100m" >&2; exit 2 ;;
esac
COMMON="--gpus 4 --gpu-type v100"

name() { echo "team3-e6-$1-lr$2-wd$3-tok$4-s0"; }   # name <optimizer> <lr> <wd> <tokens>

trunk() {  # trunk <optimizer> <lr> <wd>
  emit $COMMON $TRUNK_TIME --config $CONFIG --set run.name="$(name "$1" "$2" "$3" "${TOKS[2]}")" \
    --set optim.name="$1" --set optim.lr="$2" --set optim.weight_decay="$3" $(seed_args 0) \
    --set run.total_tokens="${NTOK[2]}" --set optim.schedule=wsd --set optim.decay_fraction=0.2 \
    --set run.ckpt_every=$((K / 2)) --set run.ckpt_keep_every=$((K / 2))
}

branches() {  # branches <optimizer> <lr> <wd>
  local trunk_dir="$RUNS/$(name "$1" "$2" "$3" "${TOKS[2]}")" i out
  for i in 0 1; do
    out="$(name "$1" "$2" "$3" "${TOKS[$i]}")"
    local prep=(gw1b python team3/experiments/branch.py "$trunk_dir" --step $((K * (i + 1))) --tokens "${NTOK[$i]}" --name "$out")
    if [[ $SUBMIT -eq 1 ]]; then echo "+ ${prep[*]}"; "${prep[@]}"; else echo "${prep[*]}"; fi
    emit $COMMON --time $BRANCH_TIME --name gw1b-e6-branch --config "$RUNS/$out/config.yaml"
  done
}

case "$STAGE" in
  trunks)   trunk adamw "$3" "$4";    trunk muon "$5" "$6" ;;
  branches) branches adamw "$3" "$4"; branches muon "$5" "$6" ;;
  *) echo "usage: $0 trunks|branches 50m|100m <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2 ;;
esac
finish
