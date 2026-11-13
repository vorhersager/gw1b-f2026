"""RQ 2.1 evaluation: the command line for the depth-vs-width study (see team2/team2_eval_plan.md).

Subcommands, in the order they are normally run:

    validate     check the run directories and print the shape table (CPU only)
    loss         score each final checkpoint on one fixed set of validation windows
    downstream   run the lm-eval tasks and save per-example scores
    latency      time KV-cached decoding on one A100

Run directories are never modified. Results go to <out>/<run name>/.
"""
from __future__ import annotations

import argparse

from rq21.downstream import DEFAULT_TASKS, cmd_downstream
from rq21.latency import cmd_latency
from rq21.loss import cmd_loss
from rq21.runs import validate


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the chosen subcommand and return its exit code."""
    # Options every subcommand needs: which runs to use.
    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument("--runs-root", required=True, help="folder containing the run directories")
    runs.add_argument("--prefix", default="rq_2_1_100m_", help="only use runs whose names start with this")

    # Options for the stages that produce results. These also include the options above.
    stage = argparse.ArgumentParser(add_help=False, parents=[runs])
    stage.add_argument("--run", action="append", help="only this run name (repeatable)")
    stage.add_argument("--out", required=True, help="folder for results")
    stage.add_argument("--force", action="store_true", help="redo runs that already have a result")

    parser = argparse.ArgumentParser(description="RQ 2.1 evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    # Each subcommand records the function that does its work in func, which is called below.
    # validate only reads the runs and prints, so it has no --out.
    p = commands.add_parser("validate", parents=[runs], help="check the runs and print the shape table")
    p.add_argument("--nominal-params", type=float, default=100e6, help="expected parameter count")
    p.set_defaults(func=validate)

    p = commands.add_parser("loss", parents=[stage], help="score checkpoints on fixed validation windows")
    p.set_defaults(func=cmd_loss)

    p = commands.add_parser("downstream", parents=[stage], help="run the lm-eval tasks")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated lm-eval task names")
    p.add_argument("--limit", type=int, default=None, help="examples per task, for the smoke job only")
    p.set_defaults(func=cmd_downstream)

    p = commands.add_parser("latency", parents=[stage], help="time cached decoding on one A100")
    p.set_defaults(func=cmd_latency)

    # Run the chosen subcommand. Its return value becomes the exit code: 0 means success.
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
