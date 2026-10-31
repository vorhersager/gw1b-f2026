"""Price the V100 scale ladder from measured pilot throughput (run after e0_pilot.sh adamw and muon).

    gw1b python team3/experiments/estimate_cost.py $GW1B_SCRATCH/teams/team3/runs/team3-e0-pilot-adamw-50m \\
        --muon $GW1B_SCRATCH/teams/team3/runs/team3-e0-pilot-muon-50m

Uses the median tokens/s of the pilot after its first log lines (compilation), scales to 100M, 200M and 350M by
FLOPs per token, and prints GPU-hours and wall-clock per run for the planned V100 ladder (E1, E1b, E5, E8).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gw1b.budget import flops_per_token  # noqa: E402
from gw1b.config import load_config  # noqa: E402


def pilot_throughput(run_dir: str) -> tuple[float, int, str]:
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    tps = [r["perf/tokens_per_s"] for r in rows if "perf/tokens_per_s" in r][2:]   # skip compile/warm-up
    if len(tps) < 3:
        raise SystemExit(f"{run_dir}: not enough logged steps to measure throughput")
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    return statistics.median(tps), summary["n_devices"], summary["gpu_kind"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adamw_pilot", help="run dir of the 50M AdamW pilot")
    ap.add_argument("--muon", help="run dir of the 50M Muon pilot (default: assume AdamW throughput)")
    a = ap.parse_args()

    tps_a, n_dev, kind = pilot_throughput(a.adamw_pilot)
    tps_m = pilot_throughput(a.muon)[0] if a.muon else tps_a
    cfg = {size: load_config(f"configs/team3/{name}.yaml") for size, name in
           ((50, "e1_50m"), (100, "e1b_100m"), (200, "ladder_200m"), (350, "ladder_350m"))}
    fpt = lambda c: flops_per_token(c.model.n_params, c.model.n_layers, c.data.seq_len, c.model.d_model)
    scale = {size: fpt(cfg[50]) / fpt(c) for size, c in cfg.items()}       # throughput relative to the 50M pilot
    tokens = {size: c.resolved_total_steps() * c.tokens_per_step for size, c in cfg.items()}

    print(f"pilot: {n_dev} x {kind}; AdamW {tps_a/1e3:,.1f}k tok/s, Muon {tps_m/1e3:,.1f}k tok/s "
          f"(Muon/AdamW step time {tps_a/tps_m:.2f}x)")
    plan = [  # (label, runs, tokens per run, tokens/s); every ladder size at 10 tokens/parameter
        ("E1 stage 1 at 20 tok/param, 50M (both)", 10, 2 * tokens[50], (tps_a + tps_m) / 2),
        ("E1 AdamW, 50M (11 tuning + 2 seeds)", 13, tokens[50], tps_a),
        ("E1 Muon, 50M (11 tuning + 2 seeds)", 13, tokens[50], tps_m),
        ("E1b AdamW, 100M (3 sweep + 1 WSD)", 4, tokens[100], tps_a * scale[100]),
        ("E1b Muon, 100M (3 sweep + 1 WSD)", 4, tokens[100], tps_m * scale[100]),
        ("E5 AdamW, 200M (3 sweep)", 3, tokens[200], tps_a * scale[200]),
        ("E5 Muon, 200M (3 sweep)", 3, tokens[200], tps_m * scale[200]),
        ("E8 AdamW, 350M (1 run)", 1, tokens[350], tps_a * scale[350]),
        ("E8 Muon, 350M (1 run)", 1, tokens[350], tps_m * scale[350]),
    ]
    total = 0.0
    for label, n, tokens, tps in plan:
        hours = tokens / tps / 3600
        gpu_h = n * hours * n_dev
        total += gpu_h
        print(f"{label:<40} {n:>2} runs x {hours:5.1f} h wall on {n_dev} GPUs = {gpu_h:8,.0f} GPU-hours")
    print(f"{'total':<40} {total:31,.0f} GPU-hours")


if __name__ == "__main__":
    main()
