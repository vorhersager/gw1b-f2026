#!/usr/bin/env bash
# Team 3, E0: pilot and correctness checks on V100 (float32), run before any sweep.
#
#   bash team3/experiments/e0_pilot.sh smoke     # 1. tiny model on the class data, end to end (minutes)
#   bash team3/experiments/e0_pilot.sh adamw     # 2. 50M AdamW, 200 steps on 4 V100s: measures throughput
#   bash team3/experiments/e0_pilot.sh resume    # 3. after (2) finishes: same run extended to 300 steps
#   bash team3/experiments/e0_pilot.sh muon      # 4. 50M Muon, 200 steps: Muon's step-time overhead
#   ... add --submit to queue. Then price the sweep:
#   gw1b python team3/experiments/estimate_cost.py $GW1B_SCRATCH/teams/team3/runs/team3-e0-pilot-adamw-50m
source "$(dirname "$0")/_common.sh"
STAGE="${1:-}"
V100="--gpus 4 --gpu-type v100"
# run.total_tokens=null: the 50M config sets total_tokens (1B), which would override the per-stage step limits.
PILOT="--config configs/team3/e1_50m.yaml --set run.total_tokens=null --set run.log_every=10 --set run.eval_every=100 --set run.ckpt_every=100"

case "$STAGE" in
  smoke)
    # tiny_debug.yaml points at synthetic data that only the class smoke test creates; read the class corpus instead
    # (vocab 32000 to match its tokenizer), so the smoke test writes nothing outside the team runs folder.
    emit --gpus 1 --gpu-type v100 --time 0:30:00 --config configs/tiny_debug.yaml --set model.vocab_size=32000 \
      --set data.train_dir='$GW1B_SCRATCH/data/fineweb-edu-10B/train' --set data.val_dir='$GW1B_SCRATCH/data/fineweb-edu-10B/val' \
      --set run.name=team3-e0-smoke --set run.out_dir='$GW1B_SCRATCH/teams/team3/runs' --set run.eval_seqs=16 ;;
  adamw)
    emit $V100 --time 1:00:00 $PILOT --set run.name=team3-e0-pilot-adamw-50m --set run.total_steps=200 ;;
  resume)
    # Re-uses the run directory of stage 2: must print "resumed from step 200" and continue to 300.
    emit $V100 --time 1:00:00 $PILOT --set run.name=team3-e0-pilot-adamw-50m --set run.total_steps=300 ;;
  muon)
    emit $V100 --time 1:00:00 $PILOT --set run.name=team3-e0-pilot-muon-50m --set run.total_steps=200 \
      --set optim.name=muon ;;
  *)
    echo "usage: $0 smoke|adamw|resume|muon [--submit]" >&2; exit 2 ;;
esac
finish
