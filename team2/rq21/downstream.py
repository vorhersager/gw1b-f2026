"""The downstream stage: run the lm-eval tasks and save per-example scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import lm_eval

from gw1b.evaluate import load_run
from gw1b.lm_eval_adapter import GW1BLM
from gw1b.tokenizer import Tokenizer
from rq21.runs import save_json, select_runs

DEFAULT_TASKS = ("hellaswag", "arc_easy", "piqa")
BATCH_SIZE = 16


def extract_records(task: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert one task's harness samples into records of task, example id, gold answer and per-example metrics.

    Raises if an example id repeats or a metric value is missing or non-finite.
    """
    records = []
    seen = set()
    for sample in samples:
        example_id = int(sample["doc_id"])
        if example_id in seen:
            raise ValueError(f"{task}: example id {example_id} appears more than once")
        seen.add(example_id)
        metrics = {m: float(sample[m]) for m in sample["metrics"]}
        if not metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError(f"{task}: example {example_id} has missing or non-finite metrics: {metrics}")
        records.append({"task": task, "id": example_id, "gold": str(sample["target"]), "metrics": metrics})
    return records


def cmd_downstream(args: argparse.Namespace) -> int:
    """Evaluate each selected run on --tasks and write downstream.jsonl and downstream.json to <out>/<run name>/.

    Calls lm_eval.simple_evaluate through gw1b.lm_eval_adapter.GW1BLM, zero-shot with log_samples=True. Raises if
    per-example means disagree with the harness aggregates.
    --limit exists only for the smoke job. Runs with an existing result are skipped unless --force is given.
    """
    runs = select_runs(args)
    if not runs:
        return 1

    tasks = args.tasks.split(",")
    for name, run_dir in runs.items():
        folder = Path(args.out) / name
        if (folder / "downstream.json").exists() and not args.force:  # written last, so it means this run finished
            print(f"{name}: skipped, {folder / 'downstream.json'} exists")
            continue
        model, cfg, step = load_run(run_dir, shard_params=False)
        lm = GW1BLM(model, Tokenizer(cfg.data.tokenizer), batch_size=BATCH_SIZE)
        result = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=args.limit, batch_size=BATCH_SIZE,
                                         log_samples=True, bootstrap_iters=0)  # the analysis does its own bootstrap

        records = []
        scores = {}
        for task, samples in result["samples"].items():
            task_records = extract_records(task, samples)
            scores[task] = {}
            for metric in task_records[0]["metrics"]:
                mean = sum(r["metrics"][metric] for r in task_records) / len(task_records)
                reported = result["results"][task][f"{metric},{samples[0]['filter']}"]
                if abs(mean - reported) > 1e-6:
                    raise ValueError(f"{name} {task} {metric}: per-example mean {mean} but harness reported {reported}")
                scores[task][metric] = mean
            records += task_records

        folder.mkdir(parents=True, exist_ok=True)
        (folder / "downstream.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        save_json(str(folder / "downstream.json"), {"run": name, "batch_size": BATCH_SIZE, "limit": args.limit,
                                                    "n_examples": {t: sum(r["task"] == t for r in records) for t in scores},
                                                    "scores": scores}, step=step)
        print(f"{name}: {scores}")
    return 0
