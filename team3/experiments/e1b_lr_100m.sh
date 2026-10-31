#!/usr/bin/env bash
# Team 3, E1b: peak learning rate at 100M on V100 (float32), both optimizers, 1B tokens each (10 tokens/parameter).
#
# Sweep: 0.5x / 1x / 2x a centre learning rate per optimizer, cosine schedule (6 runs)
#   bash team3/experiments/e1b_lr_100m.sh sweep <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>
# WSD: each optimizer's best sweep setting again with a warmup-stable-decay schedule (2 runs)
#   bash team3/experiments/e1b_lr_100m.sh wsd <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>
# ... add --submit to queue. If a winner sits at the edge of the sweep, add one more point (x2 or /2).
#
# Centres: each optimizer's 50M optimum from E1, moved along the trend if the class configs suggest one; weight
# decay: each optimizer's E1 winner. Tuning both optimizers at every size keeps the comparison fair: Muon's best
# learning rate moves little with size under update-RMS matching, AdamW's does, so a fixed value would favour one.
# The AdamW winner (with the 50M optimum) is the tuned learning rate per size delivered to Teams 1 and 2.
# WSD decays over the last 20% of steps to the same floor as cosine (10% of the peak), so the two schedules differ
# only in shape; E2 compares schedules (including decay to zero) in full.
source "$(dirname "$0")/_common.sh"
STAGE="${1:-}"
COMMON="--gpus 4 --gpu-type v100 --time 16:00:00 --config configs/team3/e1b_100m.yaml"
TOK=tok1B

run() {  # run <experiment> <optimizer> <lr> <wd> [extra --set args...]
  local exp="$1" opt="$2" lr="$3" wd="$4"; shift 4
  emit $COMMON --set run.name="team3-$exp-$opt-lr$lr-wd$wd-$TOK-s0" --set optim.name="$opt" \
    --set optim.lr="$lr" --set optim.weight_decay="$wd" $(seed_args 0) "$@"
}

[[ $# -eq 5 ]] || { echo "usage: $0 sweep|wsd <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2; }
case "$STAGE" in
  sweep)
    for f in 0.5 1 2; do run e1b adamw "$(scale_lr "$2" $f)" "$3"; done
    for f in 0.5 1 2; do run e1b muon "$(scale_lr "$4" $f)" "$5"; done ;;
  wsd)
    WSD="--set optim.schedule=wsd --set optim.decay_fraction=0.2"
    run e1bwsd adamw "$2" "$3" $WSD
    run e1bwsd muon "$4" "$5" $WSD ;;
  *)
    echo "usage: $0 sweep|wsd <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2 ;;
esac
finish
