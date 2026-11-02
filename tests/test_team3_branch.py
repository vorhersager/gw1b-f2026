"""Team 3 branching tool (team3/experiments/branch.py).  Run:  python -m pytest tests -q

A cooldown branch of a long WSD run must equal a standalone WSD run of the branch's length.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "team3", "experiments"))

from gw1b.config import load_config, save_config  # noqa: E402
from gw1b.data import write_synthetic_dataset  # noqa: E402
import branch  # noqa: E402

TINY = os.path.join(ROOT, "configs", "tiny_debug.yaml")
# WSD with a 20% cooldown: 10 steps cool down from step 8, 20 from 16, 30 from 24, so a 30-step trunk can branch
# into a 10- and a 20-step run (the same steps as E6's 10, 20 and 30 tokens per parameter).
WSD = ["optim.schedule=wsd", "optim.decay_fraction=0.2", "optim.warmup_steps=4"]


def test_check_branch_accepts_the_cooldown_start_only():
    trunk = load_config(TINY, WSD + ["run.total_steps=30"])
    short = load_config(TINY, WSD + ["run.total_steps=20", "run.name=b"])
    assert branch.cooldown_start(short) == 16 and branch.cooldown_start(trunk) == 24
    assert branch.check_branch(trunk, short, 16) == []
    assert branch.check_branch(trunk, short, 8) == []                 # earlier is fine too: still constant there
    assert branch.check_branch(trunk, short, 17) == []                # 17 updates used counts 0..16: all equal
    assert "learning rates differ at update 17" in branch.check_branch(trunk, short, 18)[0]
    whole = load_config(TINY, WSD + ["run.total_steps=30", "run.name=b"])
    assert "not inside" in branch.check_branch(trunk, whole, 30)[0]      # nothing left to train: no summary, no eval
    other = load_config(TINY, WSD + ["run.total_steps=20", "optim.lr=2e-3"])
    assert any(p.startswith("optim.lr") for p in branch.check_branch(trunk, other, 16))
    reseeded = load_config(TINY, WSD + ["run.total_steps=20", "data.seed=1"])
    assert any(p.startswith("data.seed") for p in branch.check_branch(trunk, reseeded, 16))


def test_trunk_must_keep_every_checkpoint():
    keeps = lambda every, keep: load_config(TINY, [f"run.ckpt_every={every}", f"run.ckpt_keep_every={keep}"])
    assert branch.pruning_problem(keeps(8, 8)) is None and branch.pruning_problem(keeps(16, 8)) is None
    assert branch.pruning_problem(keeps(8, 16)) and branch.pruning_problem(load_config(TINY))  # default: prunes


def test_length_in_steps_replaces_the_inherited_token_budget():
    assert "run.total_tokens=null" in branch.branch_overrides("b", None, ["run.total_steps=20"])
    assert "run.total_tokens=null" not in branch.branch_overrides("b", 4096, ["run.total_steps=20"])
    trunk = load_config(TINY, WSD + ["run.total_tokens=61440"])          # 30 steps of 16 x 128 tokens
    with tempfile.TemporaryDirectory() as d:
        save_config(trunk, os.path.join(d, "config.yaml"))
        short = load_config(os.path.join(d, "config.yaml"), branch.branch_overrides("b", None, ["run.total_steps=20"]))
    assert short.resolved_total_steps() == 20


def test_branch_equals_a_standalone_run(tmp_path):
    env = {**os.environ, "GW1B_SCRATCH": str(tmp_path), "GW1B_GROUP": str(tmp_path), "USER": "test"}
    write_synthetic_dataset(str(tmp_path / "data/synthetic/train"), n_tokens=60_000, shard_size=30_000)
    write_synthetic_dataset(str(tmp_path / "data/synthetic/val"), n_tokens=10_000, seed=1, prefix="val")
    runs = tmp_path / "users/test/runs"
    sets = WSD + ["run.log_every=2", "run.eval_every=10", "run.eval_batches=2", "run.ckpt_every=8",
                  "run.ckpt_keep_every=8"]

    def train(name, steps, config=TINY):
        args = [sys.executable, "-m", "gw1b.train", "--config", config]
        for s in (sets + [f"run.name={name}", f"run.total_steps={steps}"] if config == TINY else []):
            args += ["--set", s]
        out = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout

    train("trunk", 30)
    train("alone", 20)
    tool = [sys.executable, os.path.join(ROOT, "team3", "experiments", "branch.py"), str(runs / "trunk"),
            "--name", "branch", "--set", "run.total_steps=20"]
    bad = subprocess.run(tool + ["--step", "16", "--set", "optim.lr=2e-3"], env=env, cwd=ROOT, capture_output=True,
                         text=True)
    assert bad.returncode != 0 and "optim.lr" in bad.stderr and not (runs / "branch").exists()
    ok = subprocess.run(tool + ["--step", "16"], env=env, cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert "resumed from step 16" in train("branch", 20, config=str(runs / "branch" / "config.yaml"))

    summary = {r: json.load(open(runs / r / "summary.json")) for r in ("branch", "alone")}
    assert summary["branch"]["step"] == 20
    assert summary["branch"]["val_loss"] == pytest.approx(summary["alone"]["val_loss"], rel=1e-5)
    curve = lambda r: {m["step"]: m["train/loss"] for m in map(json.loads, open(runs / r / "metrics.jsonl"))
                       if "train/loss" in m}
    b, a = curve("branch"), curve("alone")
    assert set(a) <= set(b) and min(b) == 1 and max(b) == 20         # the trunk's curve up to 16, then the branch
    assert [b[s] for s in sorted(a)] == pytest.approx([a[s] for s in sorted(a)], rel=1e-5)
    assert json.load(open(runs / "branch" / "branch.json"))["step"] == 16

    # A trunk resumed with another learning rate: its config.yaml now says 2e-3, but the checkpoint at step 16 was
    # trained with 1e-3. Branching from it with the new config must fail.
    train("changed", 16)
    args = [sys.executable, "-m", "gw1b.train", "--config", TINY]
    for s in sets + ["run.name=changed", "run.total_steps=30", "optim.lr=2e-3"]:
        args += ["--set", s]
    assert subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True).returncode == 0
    tool[2] = str(runs / "changed")
    out = subprocess.run(tool[:3] + ["--name", "changed-branch", "--set", "run.total_steps=20", "--step", "16"],
                         env=env, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 and "checkpoint 16: optim.lr" in out.stderr and not (runs / "changed-branch").exists()
