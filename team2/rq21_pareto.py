"""Build loss-latency Pareto fronts from existing analysis CSVs."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "analysis_results"

def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))

def save(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

results = {
    row["run"]: row for row in read("results.csv")
    if row.get("loss")
}
measurements = defaultdict(list)

for row in read("latency_repeats.csv"):
    key = (
        row["run"], row["mode"],
        row["cache_capacity"], row["device"]
    )
    measurements[key].append(float(row["median_ms"]))

groups = defaultdict(list)
for (run, mode, cache, device), values in measurements.items():
    if run not in results:
        continue
    r = results[run]
    latency = statistics.median(values)
    key = (r["size"], r["budget"], mode, cache, device)
    groups[key].append({
        "run": run,
        "layers": int(r["layers"]),
        "width": int(r["width"]),
        "loss": float(r["loss"]),
        "latency_ms": latency,
        "latency_min": min(values),
        "latency_max": max(values),
        "tokens_per_second_from_median": 1000 / latency,
    })

front_rows = []
budget_rows = []

for key, points in sorted(groups.items()):
    size, budget, mode, cache, device = key
    metadata = dict(
        size=size, training_budget=budget,
        mode=mode, cache_capacity=cache, device=device
    )

    front = []
    for point in points:
        dominators = [
            other["run"] for other in points
            if other["latency_ms"] <= point["latency_ms"]
            and other["loss"] <= point["loss"]
            and (
                other["latency_ms"] < point["latency_ms"]
                or other["loss"] < point["loss"]
            )
        ]
        front_rows.append({
            **metadata, **point,
            "pareto": not dominators,
            "dominated_by": ";".join(dominators),
        })
        if not dominators:
            front.append(point)

    front.sort(key=lambda x: x["latency_ms"])

    # Each frontier point defines a measured latency threshold.
    for point in front:
        budget_rows.append({
            **metadata,
            "latency_limit_ms": point["latency_ms"],
            "best_feasible_run": point["run"],
            "layers": point["layers"],
            "loss": point["loss"],
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    for point in points:
        x = point["latency_ms"]
        ax.errorbar(
            x, point["loss"],
            xerr=[[x-point["latency_min"]],
                  [point["latency_max"]-x]],
            fmt="o", color="gray", alpha=0.7
        )
        ax.annotate(
            "{}L / d{}".format(point["layers"], point["width"]),
            (x, point["loss"]), xytext=(5, 6),
            textcoords="offset points", fontsize=8
        )

    ax.plot(
        [p["latency_ms"] for p in front],
        [p["loss"] for p in front],
        "o--", color="tab:blue", label="Empirical Pareto front"
    )
    ax.set(
        xlabel="Decode latency (ms/token; lower is better)",
        ylabel="Held-out loss (lower is better)",
        title="{} t{} | {} | cache capacity {}".format(
            size, budget, mode, cache
        )
    )
    ax.margins(x=0.18, y=0.15)
    ax.legend()
    fig.tight_layout()
    filename = "pareto_{}_t{}_{}_{}.png".format(
        size, budget, mode, cache
    )
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

save("pareto_fronts.csv", front_rows)
save("latency_budget_choices.csv", budget_rows)

print("Created Pareto plots and tables in:", OUT)
for key, points in sorted(groups.items()):
    names = [
        row["run"] for row in front_rows
        if row["size"] == key[0]
        and row["training_budget"] == key[1]
        and row["mode"] == key[2]
        and row["cache_capacity"] == key[3]
        and row["device"] == key[4]
        and row["pareto"]
    ]
    print(key[:4], "front:", ", ".join(names))
