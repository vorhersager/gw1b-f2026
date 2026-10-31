"""Bounded real-checkpoint pilot. All Pegasus execution must be inside Slurm."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .rq41_records import canonical, file_sha256, reserve_output, write_json


def verify_checkout(expected, imported_file=__file__):
    """Accept filesystem aliases/bind mounts while rejecting another checkout."""
    actual = Path(imported_file).resolve().parents[1]
    if not expected:
        raise RuntimeError("Pass --checkout with the intended development checkout")
    try:
        matches = os.path.samefile(actual, expected)
    except OSError:
        matches = False
    if not matches:
        raise RuntimeError(f"Imported package checkout {actual} differs from requested checkout {expected}")
    return actual


def choose_limit(seconds, remaining, target=500):
    for count in sorted({target, 100, 50}, reverse=True):
        if count <= target and 2 * seconds * count / 50 * 1.5 + 120 <= remaining:
            return count
    raise TimeoutError("Insufficient measured time remaining for a paired evaluation; validation outputs preserved")


def check_repeat(path_a, path_b, tolerance=1e-5):
    from .rq41_analysis import read_records
    a, b = read_records(path_a), read_records(path_b)
    if a.keys() != b.keys():
        raise ValueError("Determinism check: example sets differ")
    maximum = 0.0
    for key in a:
        for field in ("doc", "arguments", "gold", "protocol", "task_definition", "prediction", "metrics"):
            if a[key][field] != b[key][field]:
                raise ValueError(f"Determinism check: {field} differs for {key}")
        for x, y in zip(a[key]["candidates"], b[key]["candidates"], strict=True):
            maximum = max(maximum, abs(x["loglikelihood"] - y["loglikelihood"]))
    if maximum > tolerance:
        raise ValueError(f"Determinism check: maximum score difference {maximum} exceeds {tolerance}")
    return {"examples": len(a), "max_absolute_loglikelihood_difference": maximum,
            "tolerance": tolerance, "predictions_and_metrics": "exact match"}


def write_tables(report, out):
    """Percent accuracy and percentage-point differences; no significance labels."""
    rows = []
    for item in report["tasks"]:
        row = {"task": item["task"], "metric": item["metric"], "n": item["n"]}
        for name in ("score_a", "score_b", "difference_b_minus_a"):
            value = item[name]
            row[name] = 100 * value["estimate"]
            row[name + "_low"] = 100 * value["ci95"][0]
            row[name + "_high"] = 100 * value["ci95"][1]
        rows.append(row)
    with (out / "noise_table.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out / "noise_table.md").open("x", encoding="utf-8") as f:
        f.write("Exploratory 50M, one training seed. Individual, unadjusted 95% question-bootstrap intervals.\n\n")
        f.write("| Task | Metric | N | Earlier accuracy % [CI] | Later accuracy % [CI] | Change pp [CI] |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            cells = [f"{row[name]:.2f} [{row[name+'_low']:.2f}, {row[name+'_high']:.2f}]"
                     for name in ("score_a", "score_b", "difference_b_minus_a")]
            f.write(f"| {row['task']} | {row['metric']} | {row['n']} | " + " | ".join(cells) + " |\n")
        f.write("\nPositive change means improvement. No multiplicity correction or equivalence claim. "
                "This does not satisfy the three-seed 100M ablation.\n")


def pilot(out, budget_seconds=1620, checkout=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the real pilot through Slurm")
    # Explicit CLI path is independent of the container's environment variables.
    # /scratch and its /gpfs bind alias can name the same directory differently.
    root = verify_checkout(checkout or os.environ.get("GW1B_HOME"))
    out = reserve_output(out)
    start = time.monotonic()
    manifest = json.loads((root / "rq41/pilot_manifest.json").read_text())
    revisions = root / manifest["revisions_file"]
    run_dir = Path(manifest["run"])
    checks = {run_dir / "config.yaml": manifest["run_config_sha256"],
              Path(manifest["tokenizer"]): manifest["tokenizer_sha256"]}
    for checkpoint in manifest["checkpoints"]:
        checks[run_dir / "checkpoints" / str(checkpoint["step"]) / "meta/metadata"] = checkpoint["meta_sha256"]
    for path, expected in checks.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen pilot input changed: {path}")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pilot requires committed, clean code")
    write_json(out / "plan.json", {**manifest, "git_commit": commit, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                                   "package_path": str(root / "gw1b/__init__.py"),
                                   "dataset_revisions": json.loads(revisions.read_text())})
    print(canonical({"git_commit": commit, "package_path": str(root / "gw1b/__init__.py"),
                     "requested_checkout": checkout or os.environ.get("GW1B_HOME"), "out": str(out)}), flush=True)

    def execute(args, name):
        remaining = budget_seconds - (time.monotonic() - start)
        if remaining < 30:
            raise TimeoutError("Pilot time budget exhausted; outputs preserved")
        command = [sys.executable, *args]
        print(canonical({"stage": name, "command": command}), flush=True)
        stage_start = time.monotonic()
        with (out / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining)
        return time.monotonic() - stage_start

    execute(["-c", "import jax,gw1b; print(gw1b.__file__); print(jax.devices()); assert jax.default_backend() == 'gpu'"], "gpu_check")
    for checkpoint in manifest["checkpoints"]:
        step = checkpoint["step"]
        execute(["-m", "gw1b.rq41_checkpoint", "--run", str(run_dir), "--step", str(step),
                 "--out", str(out / f"verified-{step}.json")], f"restore-{step}")
        verified = json.loads((out / f"verified-{step}.json").read_text())
        if verified["training_tokens"] != checkpoint["training_tokens"] or verified["compute_dtype"] != "float32":
            raise ValueError("Checkpoint tokens/precision differ from frozen manifest")

    def evaluate(step, count, name):
        return execute(["-m", "gw1b.lm_eval_adapter", "--run", str(run_dir), "--step", str(step),
                        "--tasks", ",".join(manifest["tasks"]), "--num-fewshot", "0", "--limit", str(count),
                        "--batch-size", "2", "--rq41", "--eval-seed", "0", "--task-revisions", str(revisions),
                        "--out", str(out / name)], name)

    late = manifest["checkpoints"][-1]["step"]
    seconds = evaluate(late, 50, "validation")
    repeat_seconds = evaluate(late, 50, "validation-repeat")
    write_json(out / "determinism.json", check_repeat(out / "validation/records.jsonl", out / "validation-repeat/records.jsonl"))
    count = choose_limit(max(seconds, repeat_seconds), budget_seconds - (time.monotonic() - start), manifest["target_limit"])
    write_json(out / "selection.json", {"count_per_task": count, "validation_seconds": seconds,
                                       "repeat_seconds": repeat_seconds, "rule": manifest["count_rule"],
                                       "selection_uses_benchmark_scores": False})
    for checkpoint in manifest["checkpoints"]:
        evaluate(checkpoint["step"], count, f"step-{checkpoint['step']}")
    from .rq41_analysis import analyze
    early = manifest["checkpoints"][0]["step"]
    report = analyze(out / f"step-{early}/records.jsonl", out / f"step-{late}/records.jsonl", replicates=10000, seed=0)
    if any(item["n"] != count for item in report["tasks"]):
        raise ValueError("Actual sample counts differ from preselected pilot count")
    write_json(out / "comparison.json", report)
    write_tables(report, out)
    elapsed = time.monotonic() - start
    write_json(out / "cost.json", {"wall_seconds": elapsed, "gpus": 1, "gpu_type": "v100",
                                  "gpu_hours": elapsed / 3600, "notional_usd": elapsed / 3600,
                                  "measurement": "driver wall time; use Slurm sacct elapsed for full allocation including container startup",
                                  "price_source": "gw1b.budget.GPU_PEAK_TFLOPS v100 $1/GPU-hour; not a Pegasus invoice",
                                  "requested_allocation_gpu_hours": 0.5})
    write_json(out / "complete.json", {"completed": True, "git_commit": commit, "count_per_task": count,
                                      "checkpoints": [early, late], "exploratory": True})
    print((out / "noise_table.md").read_text(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkout", help="intended checkout; compared by filesystem identity, including GPFS aliases")
    args = parser.parse_args(argv)
    pilot(args.out, checkout=args.checkout)


if __name__ == "__main__":
    main()
