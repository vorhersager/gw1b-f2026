"""Synthetic CPU statistical/serialization tests; no model or dataset downloads."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gw1b.rq41_analysis import analyze, bootstrap_mean, read_records
from gw1b.rq41_records import canonical, digest, normalize_samples, reserve_output, write_bundle


@pytest.fixture
def inputs():
    fixture = json.loads((Path(__file__).parent / "fixtures/rq41/synthetic_mc.json").read_text())
    samples, choices = [], {}
    for ex in fixture["examples"]:
        doc = {"synthetic": True, "query": ex["query"], "choices": ex["choices"], "label": ex["gold"]}
        args = [[ex["query"], " " + c] for c in ex["choices"]]
        responses = [[s, False] for s in ex["scores"]]
        samples.append({"doc_id": ex["id"], "doc": doc, "target": ex["gold"], "arguments": args,
                        "resps": [[r] for r in responses], "filtered_resps": responses, "filter": "none",
                        "metrics": ["acc", "acc_norm"], "acc": ex["acc"], "acc_norm": ex["acc_norm"],
                        "doc_hash": digest(doc), "prompt_hash": digest(ex["query"]), "target_hash": digest(ex["gold"])})
        choices[ex["id"]] = ex["choices"]
    raw = {"samples": {"piqa": samples}, "results": {"piqa": fixture["expected_scores"]},
           "n-samples": {"piqa": {"original": 4, "effective": 4}}}
    metadata = {"synthetic": True, "model": {"run_dir": "synthetic-only", "loaded_step": 2,
                "training_tokens": 128, "training_seeds": {"initialization": 0, "data_order": 0}},
                "provenance": {"description": "synthetic fixture"},
                "protocol": {"num_fewshot": 0, "seed": 0, "tokenizer": "synthetic"}}
    definitions = {"piqa": {"config": {"output_type": "multiple_choice", "target_delimiter": " "},
                            "split": "validation", "version": "synthetic-v1", "expected_samples": 4}}
    return raw, metadata, definitions, {"piqa": choices}, {"piqa": 4}


def save_records(path, records):
    path.write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def records(inputs):
    return normalize_samples(*inputs)


def test_serialization_analysis_roundtrip(tmp_path, inputs):
    out = reserve_output(tmp_path / "bundle")
    write_bundle(out, *inputs)
    report = analyze(out / "records.jsonl")  # exercise default 10,000 replicates
    scores = {item["metric"]: item for item in report["tasks"]}
    assert scores["acc,none"]["score_a"]["estimate"] == 0.25
    assert scores["acc_norm,none"]["score_a"]["estimate"] == 0.75
    assert scores["acc,none"]["n"] == 4
    assert scores["acc,none"]["unit"] == "fraction"
    assert report["replicates"] == 10_000 and report["synthetic"] is True
    assert "unadjusted" in report["interval"]
    assert json.loads((out / "manifest.json").read_text())["expected_counts"] == {"piqa": 4}
    assert len((out / "samples.jsonl").read_text().splitlines()) == 4
    row = next(iter(read_records(out / "records.jsonl").values()))
    assert row["candidates"][0]["characters"] == 1  # é is two bytes, one Python character
    assert row["prediction"] == {"acc": 0, "acc_norm": 1}


def test_identical_and_shuffled_pairs_are_exactly_zero(tmp_path, records):
    a = save_records(tmp_path / "a.jsonl", records)
    b_rows = copy.deepcopy(records)
    for r in b_rows:
        r["model"]["loaded_step"] = 4
    b = save_records(tmp_path / "b.jsonl", b_rows)
    report = analyze(a, b, seed=19)
    for item in report["tasks"]:
        assert item["difference_b_minus_a"]["estimate"] == 0
        assert item["difference_b_minus_a"]["ci95"] == [0, 0]
    save_records(b, list(reversed(b_rows)))
    save_records(a, [records[i] for i in [2, 0, 3, 1]])
    assert analyze(a, b, seed=19) == report


def test_nonzero_paired_difference_and_seed_reproducibility(tmp_path, records):
    a = save_records(tmp_path / "a.jsonl", records)
    b_rows = copy.deepcopy(records)
    # One more raw-accuracy success, preserving the example and scoring protocol.
    b_rows[0]["prediction"]["acc"] = 1
    b_rows[0]["metrics"]["acc,none"] = 1.0
    b = save_records(tmp_path / "b.jsonl", b_rows)
    first = analyze(a, b, replicates=1234, seed=7)
    assert analyze(a, b, replicates=1234, seed=7) == first
    assert first["tasks"][0]["difference_b_minus_a"]["estimate"] == 0.25
    # A nonconstant paired vector catches implementations that accidentally bootstrap independent samples.
    # Bernoulli(1/4) sampled four times has percentile endpoints 0 and 3/4.
    assert first["tasks"][0]["difference_b_minus_a"]["ci95"] == [0.0, 0.75]
    save_records(b, list(reversed(b_rows)))
    assert analyze(a, b, replicates=1234, seed=7) == first


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate"), ("missing", "expected 4"), ("conflicting_content", "Content hash"),
    ("nonfinite", "Out of range"), ("missing_metric", "Missing metric"),
    ("mixed_seed", "Mixed runs"), ("grouped", "Unsupported task/grouped"),
    ("mmlu", "Unsupported task/grouped"), ("wrong_spec", "Unsupported metric"),
])
def test_invalid_inputs_rejected(tmp_path, records, mutation, match):
    bad = copy.deepcopy(records)
    if mutation == "duplicate": bad.append(copy.deepcopy(bad[0]))
    if mutation == "missing": bad.pop()
    if mutation == "conflicting_content": bad[0]["doc"]["query"] = "changed"
    if mutation == "nonfinite": bad[0]["metrics"]["acc,none"] = float("nan")
    if mutation == "missing_metric": del bad[0]["metrics"]["acc,none"]
    if mutation == "mixed_seed": bad[0]["model"]["training_seeds"]["initialization"] = 42
    if mutation == "grouped": bad[0]["group_id"] = "shared-passage"
    if mutation == "mmlu": bad[0]["task"] = "mmlu"
    if mutation == "wrong_spec":
        for row in bad: row["metric_specs"]["acc_norm,none"]["normalization"] = "tokens"
    # json.dumps deliberately permits NaN here to prove the reader rejects it.
    path = tmp_path / "bad.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(ValueError, match=match): read_records(path)


@pytest.mark.parametrize("field", ["doc", "task_definition", "protocol", "arguments", "gold", "training_seed", "id"])
def test_incompatible_pairs_rejected(tmp_path, records, field):
    a = save_records(tmp_path / "a.jsonl", records)
    bad = copy.deepcopy(records)
    if field == "doc":
        bad[0]["doc"]["query"] = "changed content"
        bad[0]["content_hash"] = digest(bad[0]["doc"])
    elif field == "task_definition":
        for row in bad: row[field]["version"] = "different"
    elif field == "protocol":
        for row in bad: row[field]["seed"] = 12
    elif field == "arguments": bad[0][field][0][0] = "changed prompt"
    elif field == "gold": bad[0][field]["text"] = "changed gold"
    elif field == "training_seed":
        for row in bad: row["model"]["training_seeds"]["initialization"] = 9
    else: bad[0]["example_id"] = "missing-counterpart"
    b = save_records(tmp_path / "b.jsonl", bad)
    with pytest.raises(ValueError, match="Incompatible|Different training seeds|Missing counterparts"):
        analyze(a, b)


def test_output_collision_and_incomplete_samples(tmp_path, inputs):
    out = reserve_output(tmp_path / "bundle")
    with pytest.raises(FileExistsError): reserve_output(out)
    raw, *rest = copy.deepcopy(inputs)
    raw["samples"]["piqa"].pop()
    with pytest.raises(ValueError, match="expected 4"):
        write_bundle(out, raw, *rest)
    assert (out / "raw_harness.json").exists()
    assert not (out / "manifest.json").exists()


def test_harness_metric_mismatch_is_not_silently_fixed(inputs):
    bad = copy.deepcopy(inputs)
    bad[0]["samples"]["piqa"][0]["acc"] = 1.0
    with pytest.raises(ValueError, match="disagrees with harness"):
        normalize_samples(*bad)


def test_cli_and_report_overwrite_guard(tmp_path, records):
    path = save_records(tmp_path / "records.jsonl", records)
    out = tmp_path / "report.json"
    command = [sys.executable, "-m", "gw1b.rq41_analysis", str(path), "--seed", "5", "--out", str(out)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == json.loads(out.read_text())
    assert subprocess.run(command, capture_output=True).returncode == 2


@pytest.mark.parametrize("values,replicates,seed", [([], 10, 0), ([float("inf")], 10, 0), ([1], 0, 0), ([1], 10, -1)])
def test_bad_bootstrap_arguments(values, replicates, seed):
    with pytest.raises(ValueError): bootstrap_mean(values, replicates, seed)


def test_pilot_selection_uses_timing_only():
    from gw1b.rq41_pilot import choose_limit
    assert choose_limit(10, 1000) == 500
    assert choose_limit(100, 1000) == 100
    assert choose_limit(100, 500) == 50
    with pytest.raises(TimeoutError): choose_limit(100, 100)


def test_pilot_checkout_identity_accepts_aliases_and_rejects_other_directories(tmp_path, monkeypatch):
    from gw1b.rq41_pilot import verify_checkout
    actual = tmp_path / "actual"
    (actual / "gw1b").mkdir(parents=True)
    module = actual / "gw1b/rq41_pilot.py"
    module.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    assert verify_checkout(alias, module) == actual
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="differs"): verify_checkout(other, module)
    with pytest.raises(RuntimeError, match="Pass --checkout"): verify_checkout(None, module)
    # Bind mounts may retain distinct resolved path strings despite same inode.
    checked = []
    def same_inode(a, b):
        checked.append((a, b))
        return True
    monkeypatch.setattr("gw1b.rq41_pilot.os.path.samefile", same_inode)
    assert verify_checkout(other, module) == actual
    assert checked == [(actual, other)]


def test_pilot_determinism_and_table(tmp_path, records):
    from gw1b.rq41_pilot import check_repeat, write_tables
    a = save_records(tmp_path / "a.jsonl", records)
    b = save_records(tmp_path / "b.jsonl", list(reversed(records)))
    assert check_repeat(a, b)["max_absolute_loglikelihood_difference"] == 0
    report = analyze(a, b, replicates=100, seed=0)
    write_tables(report, tmp_path)
    import csv
    with (tmp_path / "noise_table.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["score_a"]) == 25.0
    assert float(rows[0]["difference_b_minus_a"]) == 0
    assert "unadjusted" in (tmp_path / "noise_table.md").read_text()
    bad = copy.deepcopy(records)
    bad[0]["candidates"][0]["loglikelihood"] += 0.01
    save_records(b, bad)
    with pytest.raises(ValueError, match="exceeds"): check_repeat(a, b)
