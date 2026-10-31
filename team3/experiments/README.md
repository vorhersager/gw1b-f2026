# Team 3 launch scripts

Each script prints the `gw1b train` commands it would run; add `--submit` to queue them. Run them from the class
checkout, which includes the Team 3 code (Muon, `run.eval_seqs`) since it was merged:

```bash
cd /SEAS/groups/gw1b-class/gw1b-f2026      # the scripts use paths relative to the checkout
```

To test a branch that is not merged yet, use your own checkout of it instead (setting `PYTHONPATH` alone is not
enough: the class copy is always placed first):

```bash
cp -n /SEAS/groups/gw1b-class/gw1b-f2026/gw1b.local.env <checkout>/   # once: site settings (git-ignored)
source <checkout>/activate.sh                                       # each session
cd <checkout>
```

Python on the login node runs inside the class environment: `gw1b python <script>`, not `python <script>`.

| Script | Experiment | Runs |
|---|---|---|
| `e0_pilot.sh smoke\|adamw\|resume\|muon` | E0: tiny model on the class data; 50M AdamW pilot (200 steps); resume to 300 steps; 50M Muon pilot | 1 each |
| `gw1b python estimate_cost.py <adamw pilot> --muon <muon pilot>` | Prices the V100 ladder (E1, E1b, E5, E8) from the pilots' measured throughput | none |
| `e1_tune_50m.sh stage1` | E1: AdamW and Muon, 5 learning rates at weight decay 0.1, 0.5B tokens | 10 |
| `e1_tune_50m.sh stage2 <adamw lrs> <muon lrs>` | E1: weight decay 0.3 and 1.0 at each optimizer's best 3 learning rates | 12 |
| `e1_tune_50m.sh seeds <adamw lr> <wd> <muon lr> <wd>` | E1: seeds 1 and 2 of each optimizer's winner | 4 |
| `e1b_lr_100m.sh sweep <adamw lr> <wd> <muon lr> <wd>` | E1b: both optimizers at 100M, 0.5x / 1x / 2x a centre learning rate each | 6 |
| `e1b_lr_100m.sh wsd <adamw lr> <wd> <muon lr> <wd>` | E1b: each optimizer's best 100M setting with a WSD schedule (decay over the last 20%) | 2 |
| `e5_ladder_200m.sh <adamw lr> <wd> <muon lr> <wd>` | E5: both optimizers at 200M, 2B tokens, 0.5x / 1x / 2x a centre learning rate each; chains of 1-day jobs | 6 |
| `e8_ladder_350m.sh <adamw lr> <wd> <muon lr> <wd>` | E8: one run each at 350M, 3.5B tokens, at the learning rate predicted by the 50M-200M trend; chains of 1-day jobs | 2 |

Order: `e0_pilot.sh` (all four stages) → `estimate_cost.py` → E1 stage1 → stage2 → seeds → E1b sweep → E1b wsd → E5 → E8.

Conventions used by every run:

- V100s, 4 GPUs per job, float32 for both optimizers at every size (`configs/team3/*.yaml`). The A100 hours are
  kept for the production run.
- 10 training tokens per parameter at every size (0.5B at 50M, 1B at 100M, 2B at 200M, 3.5B at 350M), the ratio the
  A100 budget allows for the 1.15B production run, so the ladder predicts that run. The first E1 stage-1 sweep ran
  at 20 tokens per parameter (1B tokens) and is kept as the data-axis comparison at 50M. Warmup is half the class
  recipe's, the same share of each (halved) run. Both optimizers receive equal tuning sweeps at 50M–200M.
  The 350M pair is an extrapolation check at predicted learning rates, not an independently tuned comparison;
  interpret its result with that limitation.
- Every run is scored on the same 512 validation windows (`run.eval_seqs=512`).
- Seeds are paired: `run.seed` (initialization) and `data.seed` (data order) take the same value.
- Weight decay uses the optax convention (multiplied by the learning rate).
- Muon uses update-RMS matching (`optim.muon_consistent_rms=0.2`), so it shares AdamW's learning-rate grid.
- Run names: `team3-<experiment>-<optimizer>-lr<lr>-wd<wd>-tok<tokens>-s<seed>`, in `$GW1B_SCRATCH/teams/team3/runs`
  (the first stage-1 sweep has no `tok` tag).

Results: `gw1b python team3/analysis/scale_report.py $GW1B_SCRATCH/teams/team3/runs --out <dir>` reads every run
(finished or still training) and writes tables and plots: each optimizer's best setting per size with its seed
spread, the share of AdamW's tokens Muon needs to reach AdamW's final loss, and the trend of that share with model
size. `--prefix team3-e1-` restricts it to one experiment. Live curves: `gw1b tensorboard $GW1B_SCRATCH/teams/team3/runs`.
