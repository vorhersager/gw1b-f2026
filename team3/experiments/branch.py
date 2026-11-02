"""Team 3: start a cooldown branch from a checkpoint of a longer run (E6 data axis, the one-run ladder of RQ 3.2).

    gw1b python team3/experiments/branch.py <trunk run dir> --step 1526 --tokens 500000000 \\
        --name team3-e6-50m-adamw-lr4.8e-3-wd0.1-tok0.5B-s0
    gw1b train --gpus 4 --gpu-type v100 --time 4:00:00 --name gw1b-e6-branch --config <branch dir>/config.yaml

The branch is the trunk's config with a shorter run (--tokens, or --set run.total_steps=N) and a new name. The
script copies the trunk checkpoint at --step into the new run directory, together with the trunk's metrics up to
that step, so `gw1b train` resumes there and runs only the rest: the branch's own cooldown.

It refuses unless the branch would have trained exactly like the trunk up to --step: same model, data, batch and
optimizer, and the same learning rate at every step so far (both schedules are evaluated step by step). Then the
branch equals a standalone run of its length with the same seeds. For a WSD trunk that means branching where the
shorter run starts its cooldown, e.g. step 1526 for a 0.5B-token 50M run with decay_fraction 0.2: the trunk keeps
those checkpoints when run.ckpt_every and run.ckpt_keep_every divide them (team3/experiments/e6_tokens.sh).
Run the branch on the same number of GPUs as the trunk, so gradient accumulation (in the optimizer state) matches.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gw1b import checkpoint as ckpt_lib  # noqa: E402
from gw1b.config import TrainConfig, config_from_dict, load_config, save_config  # noqa: E402
from gw1b.train import lr_schedule  # noqa: E402

# Fields a branch may change: its name and length, the schedule after the branch point (checked step by step
# below), and bookkeeping that does not touch the weights.
FREE = {
    "run": {"name", "total_tokens", "total_steps", "ckpt_every", "ckpt_keep", "ckpt_keep_every", "eval_every",
            "eval_batches", "eval_seqs", "log_every", "tensorboard", "wandb", "profile", "gpu_hour_price_usd"},
    "optim": {"schedule", "decay_fraction", "min_lr_ratio"},
}


def check_branch(trunk: TrainConfig, branch: TrainConfig, step: int) -> list[str]:
    """Why `branch` would not have trained exactly like `trunk` for the first `step` steps (empty if it would).
    A checkpoint at `step` holds `step` updates, which used the schedule at counts 0 .. step-1."""
    problems = []
    t, b = trunk.to_dict(), branch.to_dict()
    for section in t:
        for key, value in t[section].items():
            if key not in FREE.get(section, ()) and b[section][key] != value:
                problems.append(f"{section}.{key}: trunk {value!r}, branch {b[section][key]!r}")
    if not 0 < step <= branch.resolved_total_steps():
        problems.append(f"step {step} is outside the branch's {branch.resolved_total_steps()} steps")
    else:
        steps = np.arange(step)
        lr_t = np.asarray(lr_schedule(trunk.optim, trunk.resolved_total_steps())(steps))
        lr_b = np.asarray(lr_schedule(branch.optim, branch.resolved_total_steps())(steps))
        bad = np.flatnonzero(~np.isclose(lr_t, lr_b, rtol=1e-6, atol=0))
        if bad.size:
            s = int(bad[0])
            problems.append(f"learning rates differ at update {s} (trunk {lr_t[s]:.3e}, branch {lr_b[s]:.3e}); "
                            f"branch at or before the branch's cooldown start")
    return problems


def cooldown_start(cfg: TrainConfig) -> int | None:
    """First step of the decay of a WSD run (None for other schedules)."""
    if cfg.optim.schedule != "wsd":
        return None
    total = cfg.resolved_total_steps()
    return total - max(1, int(total * cfg.optim.decay_fraction))


def branch_overrides(name: str, tokens: int | None, sets: list[str]) -> list[str]:
    """Overrides that turn the trunk's config into the branch's. A length given in steps drops the trunk's token
    budget, which would otherwise take precedence (TrainConfig.resolved_total_steps)."""
    out = [f"run.name={name}"] + ([f"run.total_tokens={tokens}"] if tokens else [])
    if any(x.startswith("run.total_steps=") for x in sets) and not tokens \
            and not any(x.startswith("run.total_tokens=") for x in sets):
        out.append("run.total_tokens=null")
    return out + sets


def history_problems(trunk_dir: str, branch: TrainConfig, step: int) -> list[str]:
    """Check the branch against the config stored in every kept trunk checkpoint up to `step`, not only the trunk's
    config.yaml (rewritten on each resume): a trunk resumed with other settings fails at the first checkpoint
    saved under them. Each checkpoint's config is the one its job trained with since the previous checkpoint."""
    ckpt_dir = os.path.join(trunk_dir, "checkpoints")
    problems = []
    for s in sorted(int(d) for d in os.listdir(ckpt_dir) if d.isdigit() and int(d) <= step):
        meta = ckpt_lib.read_meta(ckpt_dir, s)
        if "config" not in meta:
            problems.append(f"checkpoint {s} has no stored config")
            continue
        problems += [f"checkpoint {s}: {p}" for p in check_branch(config_from_dict(meta["config"]), branch, s)]
    return problems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trunk", help="run directory of the long run")
    ap.add_argument("--step", type=int, required=True, help="trunk checkpoint to branch from")
    ap.add_argument("--name", required=True, help="run name of the branch")
    ap.add_argument("--tokens", type=int, help="total tokens of the branch (same as --set run.total_tokens=N)")
    ap.add_argument("--set", action="append", default=[], metavar="section.field=value",
                    help="other changes for the branch, e.g. optim.decay_fraction=0.3")
    a = ap.parse_args(argv)

    branch = load_config(os.path.join(a.trunk, "config.yaml"), branch_overrides(a.name, a.tokens, a.set))

    src = os.path.join(a.trunk, "checkpoints", str(a.step))
    if not os.path.isdir(src):
        kept = sorted(int(d) for d in os.listdir(os.path.join(a.trunk, "checkpoints")) if d.isdigit()) \
            if os.path.isdir(os.path.join(a.trunk, "checkpoints")) else []
        raise SystemExit(f"no checkpoint at step {a.step} in {a.trunk} (kept: {kept or 'none'})")
    problems = history_problems(a.trunk, branch, a.step)
    if problems:
        raise SystemExit("branch would not match the trunk up to step %d:\n  " % a.step + "\n  ".join(problems))
    start = cooldown_start(branch)
    if start is not None and a.step != start:
        print(f"[branch] note: the branch's cooldown starts at step {start}, not at the branch point {a.step}")

    out = os.path.join(branch.run.out_dir, branch.run.name)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick another --name or remove it")
    shutil.copytree(src, os.path.join(out, "checkpoints", str(a.step)))
    save_config(branch, os.path.join(out, "config.yaml"))
    with open(os.path.join(a.trunk, "metrics.jsonl")) as f_in, open(os.path.join(out, "metrics.jsonl"), "w") as f_out:
        for line in f_in:   # the trunk's curve up to the branch point, so the branch's curve starts at step 0
            if line.strip() and json.loads(line).get("step", 0) <= a.step:
                f_out.write(line)
    with open(os.path.join(out, "branch.json"), "w") as f:
        json.dump({"trunk": os.path.abspath(a.trunk), "step": a.step, "total_steps": branch.resolved_total_steps()},
                  f, indent=2)
    print(f"[branch] {out}: step {a.step} of {os.path.basename(os.path.normpath(a.trunk))}, "
          f"{branch.resolved_total_steps() - a.step} steps to go (of {branch.resolved_total_steps()})")
    print(f"[branch] start it with:  gw1b train <same --gpus/--gpu-type as the trunk> --config {out}/config.yaml")


if __name__ == "__main__":
    main()
