"""CPU percentile bootstrap for RQ4.1 independent answer-selection records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .rq41_records import METRICS, SCHEMA, SUPPORTED_TASKS, canonical, digest, write_json


def read_records(path) -> dict:
    records = {}
    common = None
    definitions = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                canonical(row)  # reject NaN/Infinity anywhere, not just in the selected metric
                if row["schema"] != SCHEMA:
                    raise ValueError("Unsupported record schema")
                if (row["task"] not in SUPPORTED_TASKS or row["sampling_unit"] != "independent_example"
                        or row["group_id"] is not None):
                    raise ValueError("Unsupported task/grouped data; document/subject resampling is deferred")
                if not isinstance(row["example_id"], str) or not row["example_id"]:
                    raise ValueError("Expected stable string example ID")
                if row["content_hash"] != digest(row["doc"]):
                    raise ValueError("Content hash does not match document")
                if row["split"] != row["task_definition"]["split"]:
                    raise ValueError("Split differs from task definition")
                if row["task_definition"]["config"]["output_type"] != "multiple_choice":
                    raise ValueError("Only independent multiple-choice data supported")
                identity = {k: row[k] for k in ("model", "protocol", "provenance", "synthetic")}
                if common is None:
                    common = identity
                if identity != common:
                    raise ValueError("Mixed runs/checkpoints/seeds/protocols in one input; never pool seed-by-question rows")
                if not isinstance(row["synthetic"], bool):
                    raise ValueError("Missing synthetic/real label")
                if not isinstance(row["model"]["loaded_step"], int):
                    raise ValueError("Missing actual checkpoint step")
                row["model"]["training_seeds"]
                key = (row["task"], row["split"], row["example_id"])
                if key in records:
                    raise ValueError(f"Duplicate example ID: {key}")
                definition = (row["task_definition"], row["metric_specs"])
                task_key = key[:2]
                if task_key in definitions and definitions[task_key] != definition:
                    raise ValueError("Conflicting task definitions or metrics within input")
                definitions[task_key] = definition
                if not row["metrics"] or set(row["metrics"]) != set(row["metric_specs"]):
                    raise ValueError("Missing metric/specification")
                for metric, value in row["metrics"].items():
                    base, filter_name = metric.split(",")
                    if base not in METRICS or filter_name != "none" or row["metric_specs"][metric] != METRICS[base]:
                        raise ValueError("Unsupported metric or normalization")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Non-finite/non-numeric metric")
                    if value not in (0, 1):
                        raise ValueError("Accuracy must be a binary per-example indicator")
                    if value != float(row["prediction"][base] == row["gold"]["index"]):
                        raise ValueError("Prediction and metric conflict")
                if any(not math.isfinite(c["loglikelihood"]) for c in row["candidates"]):
                    raise ValueError("Non-finite candidate score")
                records[key] = row
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: empty input")
    for task_key, (definition, _) in definitions.items():
        count = sum(k[:2] == task_key for k in records)
        if count != definition["expected_samples"]:
            raise ValueError(f"{task_key}: expected {definition['expected_samples']} records, got {count}")
    return records


def bootstrap_mean(values, replicates=10_000, seed=0):
    """Resample questions; for paired differences values are already B_i - A_i."""
    if replicates < 1 or seed < 0:
        raise ValueError("Replicates must be positive and seed nonnegative")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    # Bound temporary memory, including for future larger independent splits.
    chunk = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    return {"estimate": float(values.mean()),
            "ci95": np.percentile(means, [2.5, 97.5], method="linear").tolist()}


def analyze(path_a, path_b=None, *, replicates=10_000, seed=0):
    a = read_records(path_a)
    b = read_records(path_b) if path_b is not None else None
    first_a = next(iter(a.values()))
    if b is not None:
        if set(a) != set(b):
            raise ValueError(f"Missing counterparts: only A={len(set(a)-set(b))}, only B={len(set(b)-set(a))}")
        first_b = next(iter(b.values()))
        if first_a["synthetic"] != first_b["synthetic"]:
            raise ValueError("Cannot compare synthetic and real evaluations")
        if first_a["model"]["training_seeds"] != first_b["model"]["training_seeds"]:
            raise ValueError("Different training seeds: analyze within each seed separately")
        for key in a:
            # Candidate scores/predictions may change; rendered questions may not.
            for field in ("content_hash", "doc", "task_definition", "protocol", "arguments", "gold", "metric_specs"):
                if a[key][field] != b[key][field]:
                    raise ValueError(f"Incompatible {field} for {key}")
    report = {
        "synthetic": first_a["synthetic"], "model_a": first_a["model"],
        "model_b": next(iter(b.values()))["model"] if b else None,
        "uncertainty": "question sampling conditional on these checkpoints; training-seed variation is separate",
        "interval": "individual, unadjusted 95% percentile bootstrap; not simultaneous",
        "replicates": replicates, "bootstrap_seed": seed, "numpy_version": np.__version__,
        "resampling_unit": "independent_example", "tasks": [],
    }
    for task, split in sorted({k[:2] for k in a}):
        keys = sorted(k for k in a if k[:2] == (task, split))
        for metric, spec in sorted(a[keys[0]]["metric_specs"].items()):
            av = np.array([a[k]["metrics"][metric] for k in keys])
            item = {"task": task, "split": split, "metric": metric, "n": len(keys), **spec,
                    "score_a": bootstrap_mean(av, replicates, seed)}
            if b is not None:
                bv = np.array([b[k]["metrics"][metric] for k in keys])
                item["score_b"] = bootstrap_mean(bv, replicates, seed)
                item["difference_b_minus_a"] = {"unit": "fraction (1.0 = 100 percentage points)",
                    **bootstrap_mean(bv - av, replicates, seed)}
                item["improvement_direction"] = "positive B minus A is better"
            report["tasks"].append(item)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", help="evaluation records.jsonl for A")
    ap.add_argument("--compare", help="records.jsonl for B (difference is B minus A)")
    ap.add_argument("--replicates", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="new JSON report path; existing files are rejected")
    args = ap.parse_args(argv)
    try:
        report = analyze(args.records, args.compare, replicates=args.replicates, seed=args.seed)
        if args.out:
            write_json(args.out, report)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        ap.exit(2, f"rq41_analysis: {exc}\n")


if __name__ == "__main__":
    main()
