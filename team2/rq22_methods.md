# RQ 2.2: scaling-law fitting

## Current implementation
- Inputs: completed-run summaries, using actual n_params, tokens, and val_loss.
- Model: L = E + A*(N/1e8)^(-alpha) + B*(D/1e9)^(-beta).
- A and B refer to normalized units, not raw parameter/token units.
- Objective: Huber loss on log-loss residuals, threshold 0.001.
- Optimizer: SciPy least_squares with nine initializations.
- This adapts Hoffmann Approach 3; it is not an exact reproduction.
- Parameter bounds are explicit numerical assumptions requiring sensitivity checks.

## Other methods in the research question
- Hoffmann Approach 1: construct the minimum-loss envelope across training
  curves and estimate compute-optimal model/data allocation.
- Hoffmann Approach 2: compare model sizes at fixed training FLOPs and
  locate the minimum of each IsoFLOP profile.
- Our 10/20/30 tokens-per-parameter grid is not itself an IsoFLOP design.
- Assess training-curve coverage before claiming either approach is supported.

## Current limitations
- Current snapshot: six completed runs at two distinct model sizes.
- These cannot uniquely identify all five free parameters.
- Intermediate checkpoints do not resolve missing model-size coverage.
- Additional sizes permit attempting the fit, but do not guarantee stability.
- Synthetic recovery and two-size rejection tests pass.
- No real-data final-model prediction or uncertainty interval has been produced.

## Remaining work
- Confirm comparable validation measurements across runs and target model.
- Confirm final-model parameter count and training-token budget.
- Compare fitting objectives, optimizer starts, and parameter bounds.
- Assess held-out prediction when sufficient model sizes are available.
- Assess intermediate checkpoints with training-schedule effects in mind.
- Distinguish fit sensitivity from variability across training seeds.
- Record the prediction before accessing the final model's measured loss.

## References
- Hoffmann et al., Training Compute-Optimal Large Language Models:
  https://arxiv.org/abs/2203.15556
- Choshen et al., A Hitchhiker's Guide to Scaling Law Estimation:
  https://proceedings.mlr.press/v267/choshen25a.html
