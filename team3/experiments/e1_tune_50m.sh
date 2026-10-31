#!/usr/bin/env bash
# Team 3, E1: AdamW vs Muon tuning at 50M on V100 (float32), 0.5B tokens each (10 tokens/parameter, like every
# ladder size and the production run). Run names carry the token budget (tok0.5B); the first stage-1 sweep at 1B
# tokens (20 tokens/parameter) has names without it and stays as the 50M data-axis comparison.
#
# Stage 1: 5 learning rates at weight decay 0.1, both optimizers (10 runs)
#   bash team3/experiments/e1_tune_50m.sh stage1
# Stage 2: weight decay 0.3 and 1.0 at the best 3 learning rates of each optimizer (12 runs)
#   bash team3/experiments/e1_tune_50m.sh stage2 <adamw_lr1,lr2,lr3> <muon_lr1,lr2,lr3>
# Seeds: two more seeds (1, 2) of each optimizer's winner (4 runs)
#   bash team3/experiments/e1_tune_50m.sh seeds <adamw_lr> <adamw_wd> <muon_lr> <muon_wd>
# ... add --submit to queue. If a winner sits at the edge of the grid, extend the grid by one step (x2 or /2).
#
# Weight decay uses the optax convention (multiplied by the learning rate), for both optimizers.
# Muon uses update-RMS matching (optim.muon_consistent_rms=0.2), so the same learning-rate grid applies.
source "$(dirname "$0")/_common.sh"
STAGE="${1:-}"
LRS=(1.5e-4 3e-4 6e-4 1.2e-3 2.4e-3)     # centred on the class 50M default (6e-4), factor-2 steps
COMMON="--gpus 4 --gpu-type v100 --time 6:00:00 --config configs/team3/e1_50m.yaml"
TOK=tok0.5B

run() {  # run <optimizer> <lr> <wd> <seed>
  emit $COMMON --set run.name="team3-e1-$1-lr$2-wd$3-$TOK-s$4" --set optim.name="$1" \
    --set optim.lr="$2" --set optim.weight_decay="$3" $(seed_args "$4")
}

case "$STAGE" in
  stage1)
    for opt in adamw muon; do for lr in "${LRS[@]}"; do run "$opt" "$lr" 0.1 0; done; done ;;
  stage2)
    [[ $# -eq 3 ]] || { echo "usage: $0 stage2 <adamw_lrs,> <muon_lrs,> [--submit]" >&2; exit 2; }
    IFS=, read -r -a A_LRS <<< "$2"; IFS=, read -r -a M_LRS <<< "$3"
    for wd in 0.3 1.0; do
      for lr in "${A_LRS[@]}"; do run adamw "$lr" "$wd" 0; done
      for lr in "${M_LRS[@]}"; do run muon "$lr" "$wd" 0; done
    done ;;
  seeds)
    [[ $# -eq 5 ]] || { echo "usage: $0 seeds <adamw_lr> <adamw_wd> <muon_lr> <muon_wd> [--submit]" >&2; exit 2; }
    for s in 1 2; do run adamw "$2" "$3" "$s"; run muon "$4" "$5" "$s"; done ;;
  *)
    echo "usage: $0 stage1|stage2|seeds ... [--submit]" >&2; exit 2 ;;
esac
finish
