"""Exploratory scaling-law fit; requires NumPy and SciPy.

Uses Huber residuals on log loss with multiple initializations.
Uses SciPy least_squares, not an exact reproduction of Hoffmann's optimizer.
N and D are normalized to 100M parameters and 1B tokens.
Bounds are numerical assumptions; sensitivity must be checked before reporting.
"""
import argparse
import csv
import json

import numpy as np
from scipy.optimize import least_squares


def predict(theta, n, d):
    e, a, b, alpha, beta = theta
    return e + a * (n / 1e8) ** (-alpha) + b * (d / 1e9) ** (-beta)


def fit(n, d, loss, objective="log_huber"):
    if objective not in ("log_huber", "log_mse", "raw_mse"):
        raise ValueError("Unknown fitting objective: " + objective)

    def residuals(theta):
        predicted = predict(theta, n, d)
        if objective == "raw_mse":
            return predicted - loss
        return np.log(predicted) - np.log(loss)

    if len(np.unique(n)) < 3:
        raise ValueError(
            "Full fit needs at least three distinct model sizes. "
            "Current data cannot uniquely determine E, A, and alpha."
        )
    if len(n) <= 5:
        raise ValueError("Need more than five observations for this five-parameter fit.")

    best = None
    for alpha in (0.1, 0.3, 0.7):
        for beta in (0.1, 0.3, 0.7):
            result = least_squares(
                residuals,
                x0=[float(min(loss) / 2), 0.5, 0.5, alpha, beta],
                bounds=([1e-8, 1e-8, 1e-8, 0.01, 0.01],
                        [20, 100, 100, 2, 2]),
                loss="huber" if objective == "log_huber" else "linear",
                f_scale=1e-3,
                max_nfev=5000,
            )
            if result.success and (best is None or result.cost < best.cost):
                best = result

    if best is None:
        raise ValueError("No optimization start converged.")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument(
        "--objective",
        choices=["log_huber", "log_mse", "raw_mse"],
        default="log_huber",
    )
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.exit(2, "No completed runs in input.\n")

    n, d, loss = (
        np.array([float(row[key]) for row in rows])
        for key in ("n_params", "tokens", "val_loss")
    )
    if not all(np.all(np.isfinite(x) & (x > 0)) for x in (n, d, loss)):
        parser.exit(2, "All inputs must be finite and positive.\n")

    print(f"Observations: {len(rows)}; distinct model sizes: {len(np.unique(n))}")
    try:
        result = fit(n, d, loss, objective=args.objective)
    except ValueError as exc:
        parser.exit(2, f"Fit withheld: {exc}\n")

    report = {
        "status": "exploratory; not a validated final-model prediction",
        "objective": args.objective,
        "parameters_normalized": dict(zip(
            ["E", "A", "B", "alpha", "beta"], result.x.tolist()
        )),
        "log_loss_rmse": float(np.sqrt(np.mean(
            (np.log(predict(result.x, n, d)) - np.log(loss)) ** 2
        ))),
        "jacobian_rank": int(np.linalg.matrix_rank(result.jac)),
        "active_bounds": result.active_mask.tolist(),
        "n_observations": len(rows),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
