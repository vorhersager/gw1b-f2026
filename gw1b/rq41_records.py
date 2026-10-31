"""RQ4.1 v1 records for independent, single-answer multiple-choice examples.

No JAX or harness imports: the serializer is also exercised by CPU fixtures.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

SUPPORTED_TASKS = {"hellaswag", "arc_easy", "piqa"}
SCHEMA = "gw1b.rq41.v1"
METRICS = {
    "acc": {"unit": "fraction", "higher_is_better": True,
            "normalization": "none: argmax summed continuation log-likelihood"},
    "acc_norm": {"unit": "fraction", "higher_is_better": True,
                 "normalization": "argmax summed continuation log-likelihood / len(choice) in Python characters; excludes target delimiter"},
}


def jsonable(value):
    """Convert harness numpy scalars/arrays and tuples without stringifying data."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if callable(value):
        # Harness configs can retain callables in nested metadata. Never use repr(),
        # whose process-specific memory address would break reproducible pairing.
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None
        return {"__callable__": f"{value.__module__}.{value.__qualname__}", "source": source}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical(value) -> str:
    return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(jsonable(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def reserve_output(path):
    """Reserve an evaluation directory before doing expensive work; never reuse it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def normalize_samples(raw: dict, metadata: dict, task_definitions: dict,
                      choices_by_task: dict, expected_counts: dict) -> list[dict]:
    """Parse lm-eval 0.4.13 sample dictionaries (not invented prediction fields).

    choices_by_task maps task -> {string doc_id: task.doc_to_choice(sample['doc'])}.
    Harness stores filtered responses as [loglikelihood, is_greedy] per candidate.
    Predictions are derived from those scores and checked against observed metrics.
    """
    raw = jsonable(raw)
    if not expected_counts or any(n < 1 for n in expected_counts.values()):
        raise ValueError("Expected positive sample counts for every task")
    if set(raw["samples"]) != set(expected_counts):
        raise ValueError("Unexpected/missing sample tasks")
    records = []
    for task, expected in expected_counts.items():
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task/group: {task}; independent answer-selection only")
        samples = raw["samples"][task]
        if len(samples) != expected or raw["n-samples"][task]["effective"] != expected:
            raise ValueError(f"{task}: expected {expected} samples, got {len(samples)}")
        seen = set()
        for sample in samples:
            eid = str(sample["doc_id"])
            if eid in seen:
                raise ValueError(f"Duplicate example ID: {task}/{eid}")
            seen.add(eid)
            if sample["filter"] != "none":
                raise ValueError("Only the unmodified 'none' response filter is supported")
            choices = choices_by_task[task][eid]
            args, responses = sample["arguments"], sample["filtered_resps"]
            if not choices or any(not isinstance(c, str) or not c for c in choices):
                raise ValueError("Expected nonempty text choices")
            if len(choices) != len(args) or len(choices) != len(responses):
                raise ValueError("Unsupported multiple-input/mutual-information requests")
            delimiter = task_definitions[task]["config"]["target_delimiter"]
            if any(len(a) != 2 or a[1] != delimiter + c for a, c in zip(args, choices)):
                raise ValueError("Unexpected continuation scoring settings")
            if any(a[0] != args[0][0] for a in args):
                raise ValueError("Multiple-input scoring is unsupported")
            scores = [float(r[0]) for r in responses]
            if not all(math.isfinite(s) for s in scores):
                raise ValueError("Non-finite candidate score")
            # Same first-index tie breaking as numpy.argmax in ConfigurableTask.
            norm = [s / len(c) for s, c in zip(scores, choices)]
            predictions = {"acc": max(range(len(scores)), key=scores.__getitem__),
                           "acc_norm": max(range(len(norm)), key=norm.__getitem__)}
            gold = sample["target"]
            if isinstance(gold, str) and gold in choices:
                gold = choices.index(gold)
            if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(choices):
                raise ValueError(f"Unsupported single gold target: {sample['target']!r}")
            metrics, specs = {}, {}
            for metric in sample["metrics"]:
                if metric not in METRICS:
                    raise ValueError(f"Unsupported per-example metric: {metric}")
                value = sample[metric]  # absent metrics are errors, never replaced by zero
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Non-finite/non-scalar metric: {metric}")
                if value != float(predictions[metric] == gold):
                    raise ValueError(f"{task}/{eid}: derived {metric} disagrees with harness")
                key = f"{metric},none"
                metrics[key], specs[key] = float(value), METRICS[metric]
            if not metrics:
                raise ValueError("No supported per-example metrics")
            records.append({
                "schema": SCHEMA, "synthetic": metadata["synthetic"],
                "sampling_unit": "independent_example", "group_id": None,
                "model": metadata["model"], "provenance": metadata["provenance"],
                "protocol": metadata["protocol"], "task": task,
                "split": task_definitions[task]["split"], "task_definition": task_definitions[task],
                "example_id": eid, "content_hash": digest(sample["doc"]), "doc": sample["doc"],
                "harness_hashes": {k: sample[k] for k in ("doc_hash", "prompt_hash", "target_hash")},
                "arguments": args, "gold": {"index": gold, "text": choices[gold], "target": sample["target"]},
                "prediction": predictions,
                "candidates": [{"text": c, "continuation": a[1], "loglikelihood": s,
                                "characters": len(c), "character_normalized_score": n}
                               for c, a, s, n in zip(choices, args, scores, norm)],
                "metrics": metrics, "metric_specs": specs,
            })
        for key in records[-1]["metrics"]:
            observed = sum(r["metrics"][key] for r in records if r["task"] == task) / expected
            if not math.isclose(observed, raw["results"][task][key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{task}/{key}: sample mean differs from harness aggregate")
    return jsonable(records)


def write_bundle(out, raw, metadata, task_definitions, choices_by_task, expected_counts):
    """Write raw evidence first; a failed normalization leaves no success manifest."""
    out = Path(out)
    write_json(out / "raw_harness.json", raw)
    with (out / "samples.jsonl").open("x", encoding="utf-8") as f:
        for task, samples in raw["samples"].items():
            for sample in samples:
                f.write(canonical({"task": task, "sample": sample}) + "\n")
    records = normalize_samples(raw, metadata, task_definitions, choices_by_task, expected_counts)
    with (out / "records.jsonl").open("x", encoding="utf-8") as f:
        for record in records:
            f.write(canonical(record) + "\n")
    write_json(out / "manifest.json", {**metadata, "schema": SCHEMA, "expected_counts": expected_counts,
                                      "task_definitions": task_definitions,
                                      "files": {p: file_sha256(out / p) for p in
                                                ("raw_harness.json", "samples.jsonl", "records.jsonl")}})
    return records
