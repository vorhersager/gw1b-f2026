#!/usr/bin/env bash
# Team 3, E5: the 200M point of the AdamW vs Muon scale ladder, V100 (float32), 2B tokens (10 tokens/parameter).
#
#   bash team3/experiments/e5_ladder_200m.sh <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>     # 6 runs
#   ... add --submit to queue.
#
# Like E1b, each optimizer gets its own sweep, 0.5x / 1x / 2x a centre learning rate, so both are tuned at every size.
# Centres: each optimizer's trend from 50M (E1) to 100M (E1b); weight decay: its E1/E1b winner.
# A run takes about 31-34 h on 4 V100s, so each is a chain of 1-day jobs (--chain 3): every job resumes the
# run from its last checkpoint, and a job that finds the run finished exits at once.
source "$(dirname "$0")/_common.sh"
[[ $# -eq 4 ]] || { echo "usage: $0 <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2; }
COMMON="--gpus 4 --gpu-type v100 --time 1-00:00:00 --chain 3 --config configs/team3/ladder_200m.yaml"
TOK=tok2B

run() {  # run <optimizer> <lr> <wd>
  emit $COMMON --set run.name="team3-e5-$1-lr$2-wd$3-$TOK-s0" --set optim.name="$1" \
    --set optim.lr="$2" --set optim.weight_decay="$3" $(seed_args 0)
}

for f in 0.5 1 2; do run adamw "$(scale_lr "$1" $f)" "$2"; done
for f in 0.5 1 2; do run muon "$(scale_lr "$3" $f)" "$4"; done
finish
