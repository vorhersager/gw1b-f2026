# RQ 2.1 analysis — preliminary results

## Findings for review

Ten completed 100M evaluations compare five depth/width configurations at approximately 20 and 60 training tokens per parameter. These are architecture configurations, not independent random seeds.

| Shape | Layers | Width | Loss t20 | Loss t60 | Compiled decode t60, cache capacity 512 (ms/token) |
|---|---:|---:|---:|---:|---:|
| r1 | 12 | 768 | 3.05249 | 2.93220 | 2.464 |
| r2 | 23 | 576 | 3.01515 | 2.90478 | 4.147 |
| r3 | 56 | 384 | 3.05074 | 2.93283 | 9.746 |
| r4 | 7 | 960 | 3.07153 | 2.95208 | 1.769 |
| r5 | 3 | 1344 | 3.15913 | 3.04564 | 0.985 |

The 23-layer model has the lowest held-out loss at both budgets. Moving to 56 layers does not improve this loss and increases measured decode latency. Shallower models offer lower latency at a quality cost. At t60, the 12-layer model has loss 2.93220 versus 2.90478 for 23 layers, while its compiled decode latency at cache capacity 512 is about 41% lower. This is an observed trade-off, not a universal architecture recommendation.

Every shape improves held-out loss at t60. Raw HellaSwag, ARC-Easy, PIQA and ARC-Challenge accuracy also improves in every t20-to-t60 comparison. WinoGrande and MMLU changes are mixed. MMLU subject-macro accuracy is approximately 23%; this alone does not establish useful subject knowledge.

One completed 350M training summary is present, but no 350M evaluation files were supplied. It appears in inventory and metadata tables with unavailable evaluation metrics left blank. No cross-scale quality conclusions are possible yet.

## Reproduce on Pegasus

Requires Python 3 with numpy, matplotlib and PyYAML. No JAX, checkpoints or GPU are needed. Use your existing environment if the system Python lacks these packages.

```bash
python3 analyze_rq21.py --root /scratch/gw1b-class/teams/team2 --model-size both --out ./rq21_analysis_results
```

Select `--model-size 100M`, `350M`, or `both`. The root contains `runs/` and `eval/eval_output/`. The script reads those directories and writes only to the selected separate output directory. Reusing an output directory replaces generated analysis files; use a fresh directory when retaining earlier analyses.

## Outputs and definitions

- results.csv: configuration, actual parameter/token counts, training costs, held-out loss/perplexity and downstream metrics. Scores are fractions, not percentages.
- downstream_all_metrics.csv: every task/metric with example counts. Raw and normalized accuracies are separate; absent metrics are not zero.
- inventory.csv: completion and available evaluation files, including missing 350M evaluations.
- latency_repeats.csv: each job/repeat independently, with median, mean, p95 and device metadata.
- 100M_loss.png: held-out loss by depth and token budget.
- 100M_quality_latency.png: four separate panels for implementation and cache capacity.

MMLU macro is the unweighted mean of subject accuracies; micro weights subjects by evaluated example counts. Neither is mixed into an unlabeled all-task average.

The loss evaluation uses the fixed validation selection described by the evaluator, 8,192 scored windows per run. Use this common evaluation rather than training-summary validation loss.

Latency: batch size 1, greedy decoding, 64-token prompt, 32 discarded warm-up steps, 256 timed steps, five repeats per job and two jobs. 512/2048 denote allocated cache capacity, not a 512/2048-token prompt. Both jobs report NVIDIA A100 80GB PCIe. As-shipped and compiled-only measurements stay separate. Plot points are medians of the ten repeat medians; horizontal bars span their minimum to maximum. These are variability ranges, not confidence intervals. No outliers were deleted. As-shipped timing has substantial variation.


