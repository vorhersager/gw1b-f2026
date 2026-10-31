#!/usr/bin/env bash
# Team 3, E8: the 350M point of the AdamW vs Muon scale ladder, V100 (float32), 3.5B tokens (10 tokens/parameter).
#
#   bash team3/experiments/e8_ladder_350m.sh <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>     # 2 runs
#   ... add --submit to queue.
#
# One run per optimizer at the learning rate predicted by the 50M-100M-200M trend (E1, E1b, E5); if time allows,
# a second learning rate per optimizer makes it a tuned point like the others. On V100s, like the smaller sizes, so
# every ladder loss shares one precision; the capped A100 hours go to the production run.
# A run takes about 4-4.5 days on 4 V100s, so each is a chain of 1-day jobs (--chain 6) that resume from the last
# checkpoint (every 250 steps, about 4 h); a job that finds the run finished exits at once.
source "$(dirname "$0")/_common.sh"
[[ $# -eq 4 ]] || { echo "usage: $0 <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2; }
COMMON="--gpus 4 --gpu-type v100 --time 1-00:00:00 --chain 6 --config configs/team3/ladder_350m.yaml"
TOK=tok3.5B

run() {  # run <optimizer> <lr> <wd>
  emit $COMMON --set run.name="team3-e8-$1-lr$2-wd$3-$TOK-s0" --set optim.name="$1" \
    --set optim.lr="$2" --set optim.weight_decay="$3" $(seed_args 0)
}

run adamw "$1" "$2"
run muon "$3" "$4"
finish
