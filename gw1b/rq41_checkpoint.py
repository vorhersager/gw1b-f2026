"""Read-only Orbax inventory and model-restore gate. Run Pegasus computation via Slurm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rq41_records import write_json


def verify(run_dir, step=None):
    import orbax.checkpoint as ocp
    from .checkpoint import read_meta
    from .evaluate import load_run

    directory = Path(run_dir).resolve()
    if not (directory / "config.yaml").is_file() or not (directory / "checkpoints").is_dir():
        raise FileNotFoundError(f"Missing config/checkpoints under {directory}")
    manager = ocp.CheckpointManager(str(directory / "checkpoints"),
                                   options=ocp.CheckpointManagerOptions(read_only=True))
    try:
        steps = sorted(manager.all_steps())
    finally:
        manager.close()
    if not steps or (step is not None and step not in steps):
        raise ValueError(f"Requested completed step {step} unavailable; Orbax steps: {steps}")
    selected = steps[-1] if step is None else step
    # Listing a numeric directory is insufficient: require a successful parameter restore.
    model, cfg, loaded = load_run(str(directory), selected, shard_params=False)
    meta = read_meta(str(directory / "checkpoints"), loaded)
    if meta.get("step", loaded) != loaded:
        raise ValueError("Restored parameter step and checkpoint metadata disagree")
    from .tokenizer import Tokenizer
    Tokenizer(cfg.data.tokenizer)  # fail here if the saved tokenizer is unavailable
    return {"run": str(directory), "completed_steps": steps, "loaded_step": loaded,
            "training_tokens": meta.get("tokens_seen"), "parameter_restore_verified": True,
            "compute_dtype": model.cfg.dtype}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--out", required=True, help="new verification JSON path")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error("verification output already exists; use a new path")
    report = verify(args.run, args.step)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
