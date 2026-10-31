"""Opt-in RQ4.1 provenance and pinned-harness task inspection."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
from pathlib import Path

from .rq41_records import SUPPORTED_TASKS, digest, file_sha256, jsonable


def prepare_tasks(names, limit, num_fewshot, batch_size, seed, revisions=None):
    if importlib.metadata.version("lm-eval") != "0.4.13":
        raise ValueError("RQ4.1 records require the inspected lm-eval==0.4.13 interface")
    if not names or len(set(names)) != len(names) or not set(names) <= SUPPORTED_TASKS:
        raise ValueError("RQ4.1 supports only hellaswag,arc_easy,piqa; groups are unsupported")
    if num_fewshot != 0 or limit is None or limit < 1 or batch_size < 1 or seed < 0:
        raise ValueError("RQ4.1 requires --num-fewshot 0, positive --limit/--batch-size and nonnegative --eval-seed")
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    if revisions is not None:
        if set(revisions) != set(names) or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v)
                                               for v in revisions.values()):
            raise ValueError("Task revisions must give an immutable 40-character Hub commit for every requested task")
        # In 0.4.13 an inline {task: name, ...} does not inherit registered YAML.
        # Resolve the complete installed template before adding a revision.
        from lm_eval.tasks._yaml_loader import load_yaml
        specs = []
        for name in names:
            config = dict(load_yaml(manager.task_index[name].yaml_path))
            config["dataset_kwargs"] = {**(config.get("dataset_kwargs") or {}), "revision": revisions[name]}
            specs.append(config)
    else:
        specs = names
    loaded = manager.load(specs)
    if loaded["groups"] or set(loaded["tasks"]) != set(names):
        raise ValueError("Task groups/aliases are unsupported")
    return manager, loaded["tasks"]


def inspect_tasks(objects, raw, limit):
    definitions, choices, counts = {}, {}, {}
    for name, task in objects.items():
        config = raw["configs"][name]
        if config["output_type"] != "multiple_choice" or task.multiple_input or task.multiple_target:
            raise ValueError("Only independent single-answer multiple-choice tasks supported")
        docs = task.eval_docs
        split = config["test_split"] if task.has_test_docs() else config["validation_split"]
        counts[name] = min(limit, len(docs))
        choices[name] = {str(s["doc_id"]): task.doc_to_choice(s["doc"]) for s in raw["samples"][name]}
        definitions[name] = {
            "config": config, "version": raw["versions"][name], "split": split,
            "expected_samples": counts[name], "selection": "first N harness doc_iterator IDs (no shuffle)",
            "dataset_fingerprint": getattr(docs, "_fingerprint", None),
            "dataset_version": str(docs.info.version) if getattr(docs, "info", None) else None,
            "requested_dataset_revision": (config.get("dataset_kwargs") or {}).get("revision"),
        }
    return jsonable(definitions), jsonable(choices), counts


def build_metadata(run_dir, cfg, model, tok, loaded_step, batch_size, seed, synthetic=False):
    import gw1b
    import jax
    import lm_eval
    from .checkpoint import read_meta

    root = Path(__file__).resolve().parents[1]
    def git(*args):
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    meta = read_meta(str(Path(run_dir) / "checkpoints"), loaded_step)
    if meta.get("step", loaded_step) != loaded_step:
        raise ValueError("Checkpoint metadata step differs from restored model step")
    training_config = meta.get("config") or cfg.to_dict()
    dependencies = {}
    for package in ("lm-eval", "jax", "jaxlib", "flax", "orbax-checkpoint", "sentencepiece", "numpy",
                    "datasets", "huggingface-hub", "torch", "transformers", "jinja2", "pyyaml", "optax"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    harness_root = Path(lm_eval.__file__).resolve().parent
    harness_sources = {p: file_sha256(harness_root / p) for p in
                       ("api/task.py", "evaluator.py", "tasks/hellaswag/hellaswag.yaml",
                        "tasks/hellaswag/utils.py", "tasks/arc/arc_easy.yaml", "tasks/piqa/piqa.yaml")}
    source_files = ("lm_eval_adapter.py", "rq41_records.py", "rq41_evaluation.py", "model.py",
                    "tokenizer.py", "evaluate.py", "checkpoint.py", "sharding.py", "config.py", "generate.py")
    sources = {p: file_sha256(root / "gw1b" / p) for p in source_files}
    protocol = {
        "prompt_format": "lm-eval-0.4.13-standard-continuation-v1", "num_fewshot": 0,
        "apply_chat_template": False, "system_instruction": None, "response_filter": "none",
        "evaluation_seeds": {"python": seed, "numpy": seed, "torch": seed, "fewshot": seed},
        "max_length": model.cfg.max_seq_len,
        "truncation": "adapter keeps last max_length+1 context/continuation tokens; scores retained continuation",
        "tokenizer": {"sha256": file_sha256(tok.path), "vocab_size": tok.vocab_size,
                      "bos_added": False, "empty_context_prefix_id": tok.eos_id},
        "precision": {"compute": model.cfg.dtype, "parameters": model.cfg.param_dtype,
                      "log_softmax": "float32", "attention": model.attn_impl,
                      "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision)},
        "batch_size": batch_size, "dependencies": dependencies,
        "scoring_source_sha256": digest(sources), "harness_sources": harness_sources,
    }
    return {
        "synthetic": synthetic,
        "model": {"run_dir": str(Path(run_dir).resolve()), "run_name": cfg.run.name,
                  "loaded_step": loaded_step, "training_tokens": meta.get("tokens_seen"),
                  "training_tokens_source": "checkpoint.meta.tokens_seen" if "tokens_seen" in meta else None,
                  "training_seeds": {"initialization": training_config["run"].get("seed"),
                                     "data_order": training_config["data"].get("seed")},
                  "training_seeds_source": "checkpoint.meta.config" if meta.get("config") else "run.config.yaml",
                  "training_config_sha256": digest(training_config), "n_params": cfg.model.n_params},
        "provenance": {"package_path": gw1b.__file__, "git_commit": git("rev-parse", "HEAD"),
                       "git_status": git("status", "--porcelain"), "source_sha256": sources,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "devices": [str(d) for d in jax.devices()], "tokenizer_path": tok.path,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "protocol": protocol,
    }
