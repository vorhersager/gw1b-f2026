"""Fixed-alpha sensitivity analysis. Output is not a confidence interval."""
import argparse
import csv
import sys

import numpy as np
from scipy.optimize import least_squares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--target-params", type=float, required=True)
    parser.add_argument("--target-tokens", type=float, required=True)
    args = parser.parse_args()

    if min(args.target_params, args.target_tokens) <= 0:
        parser.error("Target parameters and tokens must be positive.")

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error("Input contains no observations.")

    n, d, y = (
        np.array([float(row[key]) for row in rows])
        for key in ("n_params", "tokens", "val_loss")
    )
    if not all(np.all(np.isfinite(x) & (x > 0)) for x in (n, d, y)):
        parser.error("Inputs must be finite and positive.")

    n, d = n / 1e8, d / 1e9
    target_n = args.target_params / 1e8
    target_d = args.target_tokens / 1e9

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "objective", "assumed_alpha", "beta", "E", "A_normalized",
        "B_normalized", "loss_rmse", "predicted_loss", "bound_hit",
        "jacobian_rank", "n_observations", "target_params", "target_tokens",
    ])

    for objective in ("log_huber", "log_mse", "raw_mse"):
        for alpha in (0.10, 0.20, 0.34, 0.50, 0.70):
            def model(t, nn, dd):
                e, a, b, beta = t
                return e + a * nn**(-alpha) + b * dd**(-beta)

            def residuals(t):
                estimated = model(t, n, d)
                if objective == "raw_mse":
                    return estimated - y
                return np.log(estimated) - np.log(y)

            best = None
            for beta_start in (0.1, 0.3, 0.7):
                for e_start in (0.5, 1.5, 2.5):
                    result = least_squares(
                        residuals,
                        [e_start, 0.5, 0.5, beta_start],
                        bounds=([1e-8, 1e-8, 1e-8, 0.01],
                                [20, 100, 100, 2]),
                        loss="huber" if objective == "log_huber" else "linear",
                        f_scale=1e-3,
                        max_nfev=5000,
                    )
                    if result.success and (
                        best is None or result.cost < best.cost
                    ):
                        best = result

            if best is None:
                raise RuntimeError(f"No converged fit: {objective}, alpha={alpha}")

            e, a, b, beta = best.x
            writer.writerow([
                objective, alpha, beta, e, a, b,
                float(np.sqrt(np.mean((model(best.x, n, d) - y)**2))),
                float(model(best.x, target_n, target_d)),
                bool(np.any(best.active_mask)),
                int(np.linalg.matrix_rank(best.jac)),
                len(rows), args.target_params, args.target_tokens,
            ])


if __name__ == "__main__":
    main()
