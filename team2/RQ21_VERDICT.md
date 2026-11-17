# RQ 2.1 — preliminary verdict for the available 100M results

## Scope
Five approximately 100M-parameter depth/width configurations, trained at
approximately 20 and 60 tokens per parameter. No 350M evaluation results
were included. Conclusions are descriptive point estimates; independent
training seeds and quality confidence intervals are not available.

## Claim 1: Deep-and-thin models win on loss at fixed training FLOPs
Verdict: partial support for a depth/width trade-off; the fixed-FLOP
claim is not yet established.

The 23-layer, width-576 model has the lowest held-out loss at both budgets.
The deepest configuration (56 layers, width 384) does not win.
Recorded training FLOPs must be compared before describing these runs
as compute-matched. Fixed parameter counts and token budgets alone
do not establish identical training FLOPs.

## Claim 2: The depth advantage shrinks above 40 tokens per parameter
Verdict: limited, comparator-dependent support.

The loss advantage of 23 over 12 layers decreases from 0.037345 at t20
to 0.027415 at t60. Against 3 layers, it changes only slightly:
0.143981 to 0.140853. Against 56 layers, the 23-layer advantage also
shrinks, but that is not evidence that deeper models generally win.

Only t20 and t60 were tested. These results cannot locate a threshold
at 40 tokens per parameter or establish a general interaction.

## Claim 3: The depth advantage reverses under a latency budget
Verdict: support for latency-dependent model selection; not a universal
quality reversal.

The unconstrained loss winner is the 23-layer model. When its measured
latency exceeds the allowed limit, shallower configurations become
the best feasible choices. See latency_budget_choices.csv for the
measured thresholds, separately by training budget, implementation,
cache capacity and hardware.

These thresholds use median latency estimates, not guarantees that
every decode step meets the limit. The results do not independently
establish the proposed causal mechanism of inference serialization.

## Empirical Pareto fronts
For both token budgets, both implementations and both cache capacities,
the loss-latency front contains:
- r5: 3 layers, width 1344
- r4: 7 layers, width 960
- r1: 12 layers, width 768
- r2: 23 layers, width 576

The 56-layer r3 configuration is dominated in every comparison.

Dominance uses point estimates: another model must have no higher loss
and no higher latency, with a strict improvement in at least one.
No statistical confidence in Pareto membership is claimed.

## Measurement definitions
Latency is the median of per-repeat median decode times.
Horizontal bars span the minimum and maximum repeat medians; they are
not confidence intervals. Tokens/second is 1000 divided by this latency
summary, rather than independently measured end-to-end throughput.

Cache capacities 512 and 2048 are allocated capacities, not prompt lengths.
As-shipped and compiled-only results are analyzed separately.
These fronts use held-out loss as quality; downstream-task fronts may differ.

## Overall conclusion
The hypothesis is partially supported by the available 100M experiments.
Moderate depth gives the lowest observed loss, while shallower models
provide useful latency-quality trade-offs. The extreme-depth model is
dominated. Fixed-FLOP, cross-scale and statistical claims remain unresolved.

## Training-FLOP audit — clarification of Claim 1
The summaries show identical training-token counts within each budget,
but recorded training FLOPs differ substantially across architectures:

| Shape | Layers | t20 FLOPs | t60 FLOPs |
|---|---:|---:|---:|
| r1 | 12 | 1.654011e18 | 4.962251e18 |
| r2 | 23 | 1.849327e18 | 5.548225e18 |
| r3 | 56 | 2.261789e18 | 6.785663e18 |
| r4 | 7 | 1.524789e18 | 4.574567e18 |
| r5 | 3 | 1.407950e18 | 4.224034e18 |

The maximum/minimum spread is 60.64% at both budgets. The definition of
the logged FLOP estimate should also be checked in the training code.

Updated Claim 1 verdict: UNTESTED at fixed training FLOPs.
The 23-layer model wins on observed loss under approximately fixed
parameters and fixed token counts, but that does not establish a
fixed-compute advantage.

The empirical loss-latency fronts describe these trained checkpoints;
they do not establish compute-optimal architecture under equal training
FLOPs. Testing that requires compute-matched checkpoints or experiments.
