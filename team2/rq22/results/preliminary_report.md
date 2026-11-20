# Preliminary GW-1B validation-loss estimate

Working target: 1.15B parameters trained on 20B tokens.
Target and validation comparability require confirmation from Team 3/4.

Inputs: six completed runs, covering approximately 50M and 100M
parameters at 10/20/30 tokens per parameter.

Method: fixed-alpha parametric fits, using actual parameter counts and
training tokens. Tested alpha values: 0.10, 0.20, 0.34, 0.50, 0.70.
Compared log-Huber, log-MSE, and raw-MSE objectives.

Results:
- At assumed alpha=0.34: predicted validation loss approximately 2.458.
- Across tested alpha values and objectives: approximately 2.350–2.556.
- Log-Huber and log-MSE agree to the displayed precision.
- Raw-MSE changes the alpha=0.34 prediction by approximately 0.0004.
- No fits hit parameter bounds; fixed-alpha Jacobian ranks were four.

Limitations:
- Alpha is assumed, not identified by these two model sizes.
- The range is assumption sensitivity, not a confidence interval.
- The range does not include all sources of uncertainty or model misspecification.
- No larger-model prediction has yet been validated on real held-out runs.
- Results predict validation loss, not downstream benchmark scores.
- Update with completed 200M/350M runs before the Sunday estimate.

Reproduction, from the repository root:
gw1b python team2/rq22_preliminary.py team2/rq22/results/preliminary_inputs.csv --target-params 1150000000 --target-tokens 20000000000
