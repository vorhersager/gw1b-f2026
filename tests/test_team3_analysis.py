"""Team 3 analysis: scale_report.py on fake run directories in the trainer's format.  Run:  python -m pytest tests -q"""
import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "team3", "analysis"))

import numpy as np  # noqa: E402

from gw1b.config import load_config, save_config  # noqa: E402
import scale_report  # noqa: E402

SIZE_50M = os.path.join(ROOT, "configs", "team3", "e1_50m.yaml")       # 10 tokens/parameter


def fake_run(root, name, opt, lr, final, seed=0, schedule="cosine", step_time=1.0, finished=True, losses=None):
    """A run whose validation loss falls linearly from final + 1 to `final` over 4 evaluations of 250M tokens
    (or follows `losses`)."""
    d = os.path.join(root, name)
    os.makedirs(d)
    cfg = load_config(SIZE_50M, [f"optim.name={opt}", f"optim.lr={lr}", f"run.seed={seed}",
                                 f"optim.schedule={schedule}", f"run.name={name}"])
    save_config(cfg, os.path.join(d, "config.yaml"))
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for i in range(4):
            f.write(json.dumps({"step": i, "perf/step_time_s": step_time}) + "\n")
        for i, loss in enumerate(np.linspace(final + 1, final, 4) if losses is None else losses, start=1):
            f.write(json.dumps({"step": 1000 * i, "tokens_seen": 250_000_000 * i, "val/loss": loss}) + "\n")
    if finished:
        json.dump({"finished": True}, open(os.path.join(d, "summary.json"), "w"))


def test_tokens_to_reach_interpolates():
    curve = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
    assert scale_report.tokens_to_reach(curve, 3.5) == pytest.approx(2.5)
    assert scale_report.tokens_to_reach(curve, 5.5) == 1.0
    assert scale_report.tokens_to_reach(curve, 2.0) is None


def test_scale_report_best_settings_and_multiplier(tmp_path):
    runs = str(tmp_path / "runs")
    fake_run(runs, "team3-e1-adamw-lr3e-4-wd0.1-s0", "adamw", 3e-4, 4.0)
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.5)          # AdamW winner
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s1", "adamw", 6e-4, 3.7, seed=1)
    fake_run(runs, "team3-e1-muon-lr6e-4-wd0.1-s0", "muon", 6e-4, 3.0, step_time=1.1)  # Muon winner
    fake_run(runs, "team3-e1-muon-lr1.2e-3-wd0.1-s0", "muon", 1.2e-3, 2.0, finished=False)  # unfinished: ignored
    fake_run(runs, "team3-e1-muon-lr2.4e-3-wd0.1-s0", "muon", 2.4e-3, 3.2, step_time=1.1)
    fake_run(runs, "team3-e1bwsd-muon-lr6e-4-wd0.1-s0", "muon", 6e-4, 2.9, schedule="wsd")  # separate group
    fake_run(runs, "other-team-run", "adamw", 6e-4, 1.0)                              # not team3-: ignored
    fake_run(runs, "team3-e0-pilot-adamw-50m", "adamw", 6e-4, 1.0)                    # E0 pilot: excluded
    out = str(tmp_path / "report")
    scale_report.main([runs, "--out", out, "--no-plots"])  # plots: test_plots_are_written (needs matplotlib)

    assert all("e0-pilot" not in r["run"] for r in csv.DictReader(open(os.path.join(out, "runs.csv"))))
    best = {(r["group"], r["optimizer"]): r for r in csv.DictReader(open(os.path.join(out, "best.csv")))}
    cosine = next(g for g, _ in best if "cosine" in g)
    assert float(best[cosine, "adamw"]["lr"]) == 6e-4 and int(best[cosine, "adamw"]["n_seeds"]) == 2
    assert float(best[cosine, "adamw"]["mean"]) == pytest.approx(3.6)
    assert float(best[cosine, "muon"]["lr"]) == 6e-4                     # the unfinished 2.0 run does not count
    assert any("wsd" in g for g, _ in best)

    mult = {r["group"]: r for r in csv.DictReader(open(os.path.join(out, "multipliers.csv")))}[cosine]
    # Muon's curve 4.0 -> 3.0 over 250M..1B tokens reaches AdamW's mean 3.6 at 550M tokens: 55% of AdamW's 1B
    assert float(mult["token_share"]) == pytest.approx(0.55)
    assert float(mult["step_time_ratio"]) == pytest.approx(1.1)
    assert float(mult["time_share"]) == pytest.approx(0.605)
    assert os.path.exists(os.path.join(out, "runs.csv"))


def test_plots_are_written(tmp_path):
    pytest.importorskip("matplotlib")  # not in the class CI's package list; installed on Pegasus
    runs = str(tmp_path / "runs")
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.5)
    fake_run(runs, "team3-e1-muon-lr6e-4-wd0.1-s0", "muon", 6e-4, 3.0)
    out = str(tmp_path / "report")
    scale_report.main([runs, "--out", out])
    (group,) = {r["group"] for r in csv.DictReader(open(os.path.join(out, "runs.csv")))}
    for f in ("scale.png", f"curves_{group}.png", f"lr_{group}.png"):
        assert os.path.exists(os.path.join(out, f)), f


def test_diverged_run_never_wins(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    nan = float("nan")
    fake_run(runs, "team3-e1-adamw-lr1.2e-3-wd0.1-s0", "adamw", 1.2e-3, nan, losses=[5.0, 4.0, nan, nan])  # sorts first
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.5)
    out = str(tmp_path / "report")
    scale_report.main([runs, "--out", out, "--no-plots"])
    (row,) = csv.DictReader(open(os.path.join(out, "best.csv")))
    assert float(row["lr"]) == 6e-4 and float(row["mean"]) == pytest.approx(3.5)
    assert "diverged" in capsys.readouterr().out


def test_token_share_needs_every_muon_seed(tmp_path):
    runs = str(tmp_path / "runs")
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.5)
    fake_run(runs, "team3-e1-muon-lr6e-4-wd0.1-s0", "muon", 6e-4, 3.0)              # reaches 3.5 at 625M tokens
    fake_run(runs, "team3-e1-muon-lr6e-4-wd0.1-s1", "muon", 6e-4, 3.6, seed=1)      # never reaches 3.5
    out = str(tmp_path / "report")
    scale_report.main([runs, "--out", out, "--no-plots"])
    (m,) = csv.DictReader(open(os.path.join(out, "multipliers.csv")))
    assert (m["muon_seeds_reaching"], m["muon_seeds"]) == ("1", "2")
    assert m["token_share"] == "" and m["time_share"] == ""                      # not the one seed's 62.5%


def test_scale_trend_keeps_token_budgets_apart():
    row = lambda size, tpp, loss, schedule="cosine": {"params_m": size, "tok_per_param": tpp, "schedule": schedule,
                                                       "loss": loss}
    rows = [row(50, 10, 4.0), row(100, 10, 3.6), row(200, 10, 3.3),  # the ladder
            row(50, 20, 3.8),                                         # data-axis point at 50M: one size, no trend
            row(100, 10, 3.5, schedule="wsd")]                        # schedule check: not on the trend
    series = scale_report.scale_series(rows, lambda r: (r["loss"],))
    assert series == {10: [(50, 4.0), (100, 3.6), (200, 3.3)]}


def test_rerun_into_same_out_leaves_nothing_stale(tmp_path):
    runs = str(tmp_path / "runs")
    fake_run(runs, "team3-e1-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.5)
    fake_run(runs, "team3-e1-muon-lr6e-4-wd0.1-s0", "muon", 6e-4, 3.0)
    fake_run(runs, "team3-e1b-adamw-lr6e-4-wd0.1-s0", "adamw", 6e-4, 3.4, finished=False)  # still training
    out = str(tmp_path / "report")
    scale_report.main([runs, "--out", out, "--no-plots"])
    assert len(list(csv.DictReader(open(os.path.join(out, "multipliers.csv"))))) == 1
    for old_plot in ("scale.png", "curves_old.png", "lr_old.png"):  # as an earlier report with plots leaves them
        open(os.path.join(out, old_plot), "w").close()
    keep = os.path.join(out, "notes.txt")
    open(keep, "w").write("not ours")

    scale_report.main([runs, "--prefix", "team3-e1b-", "--out", out, "--no-plots"])  # no finished pair here
    for table in ("best.csv", "multipliers.csv"):
        assert list(csv.DictReader(open(os.path.join(out, table)))) == []  # header only, not the old rows
    assert [r["run"] for r in csv.DictReader(open(os.path.join(out, "runs.csv")))] == ["team3-e1b-adamw-lr6e-4-wd0.1-s0"]
    assert not [f for f in os.listdir(out) if f.endswith(".png")]  # the earlier report's plots are gone
    assert os.path.exists(keep)                                       # files the script did not write stay


def test_diverged_selected_seed_blocks_multiplier(tmp_path):
    runs = str(tmp_path / 'runs')
    fake_run(runs, 'team3-adamw', 'adamw', 6e-4, 3.5)
    fake_run(runs, 'team3-muon-s0', 'muon', 6e-4, 3.0)
    fake_run(runs, 'team3-muon-s1', 'muon', 6e-4, float('nan'), seed=1,
             losses=[4.0, 3.0, float('nan'), float('nan')])
    best = scale_report.best_settings([scale_report.load_run(os.path.join(runs, n)) for n in os.listdir(runs)])
    m = scale_report.multiplier(best)
    assert (m['muon_seeds_reaching'], m['muon_seeds']) == (1, 2)
    assert m['token_share'] is None and m['time_share'] is None


def test_nonfinite_interpolation_is_not_a_reached_target():
    assert scale_report.tokens_to_reach(np.array([[1., float('nan')], [2., 3.]]), 3.5) is None


def test_group_key_preserves_exact_budget():
    r = dict(params_m=50, tok_per_param=10, dtype='float32', schedule='cosine', tokens_planned=500_000_000)
    assert scale_report.group_key(r) != scale_report.group_key(r | {'tokens_planned': 510_000_000})


def test_empty_selection_clears_old_outputs(tmp_path):
    runs, out = str(tmp_path / 'runs'), str(tmp_path / 'report')
    fake_run(runs, 'team3-adamw', 'adamw', 6e-4, 3.5)
    scale_report.main([runs, '--out', out, '--no-plots'])
    with pytest.raises(SystemExit, match='no runs'):
        scale_report.main([runs, '--prefix', 'absent-', '--out', out, '--no-plots'])
    assert not any(os.path.exists(os.path.join(out, f)) for f in ('runs.csv', 'best.csv', 'multipliers.csv'))
