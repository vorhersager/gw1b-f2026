"""Offline integration: real pinned harness + JAX adapter + Orbax, synthetic data/model.

Only dataset downloading is substituted. No Pegasus checkpoint/benchmark claim.
"""
import importlib.metadata
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def test_pinned_harness_jax_checkpoint_bundle(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("sentencepiece")
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("lm_eval")
    if importlib.metadata.version("lm-eval") != "0.4.13":
        pytest.skip("Requires inspected lm-eval==0.4.13")

    from gw1b import checkpoint
    from gw1b.config import DataConfig, ModelConfig, RunConfig, TrainConfig
    from gw1b.lm_eval_adapter import run
    from gw1b.rq41_analysis import analyze, read_records
    from gw1b.rq41_checkpoint import verify
    from gw1b.sharding import create_model, make_mesh
    from gw1b.tokenizer import Tokenizer, train_sentencepiece

    def synthetic_dataset(path, name=None, **kwargs):
        assert kwargs.get("revision") == "a" * 40
        docs = []
        for i in range(4):
            if path == "Rowan/hellaswag":
                doc = {"ctx_a": f"Synthetic scene {i}.", "ctx_b": "then", "activity_label": "Synthetic",
                       "endings": ["é", "long", "other", "last"], "label": str(i % 4)}
            elif path == "allenai/ai2_arc":
                doc = {"id": f"synthetic-{i}", "question": f"Synthetic science {i}?",
                       "choices": {"label": ["A", "B", "C"], "text": ["é", "long", "other"]},
                       "answerKey": ["A", "B", "C"][i % 3]}
            elif path == "baber/piqa":
                doc = {"goal": f"Synthetic goal {i}", "sol1": "é", "sol2": "long", "label": i % 2}
            else:
                raise AssertionError(f"Unexpected network dataset request: {path}")
            docs.append(doc)
        return datasets.DatasetDict({split: datasets.Dataset.from_list(docs)
                                     for split in ("train", "validation", "test")})

    monkeypatch.setattr(datasets, "load_dataset", synthetic_dataset)
    corpus = tmp_path / "synthetic-corpus.txt"
    corpus.write_text(("Synthetic scene goal science then é long other last. Question Answer:\n" * 100), encoding="utf-8")
    tok_path = train_sentencepiece(str(corpus), str(tmp_path / "synthetic-tokenizer"), vocab_size=320,
                                  num_threads=1, extra_args={"hard_vocab_limit": False, "minloglevel": 2})
    tok = Tokenizer(tok_path)
    run_dir = tmp_path / "synthetic-untrained-run"
    run_dir.mkdir()
    cfg = TrainConfig(model=ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                                        n_kv_heads=1, head_dim=8, d_ff=32, max_seq_len=64, dtype="float32"),
                      data=DataConfig(tokenizer=tok_path, seq_len=64),
                      run=RunConfig(name="synthetic-untrained-integration", batch_size=2, shard_params=False))
    (run_dir / "config.yaml").write_text(cfg.to_yaml())
    model = create_model(cfg, make_mesh())
    manager = checkpoint.make_manager(str(run_dir / "checkpoints"), save_every=1, async_save=False)
    try:
        checkpoint.save(manager, 0, model, None, {"step": 0, "tokens_seen": 0, "config": cfg.to_dict()}, force=True)
        manager.wait_until_finished()
    finally:
        manager.close()
    verified = verify(run_dir)
    assert verified["loaded_step"] == 0 and verified["parameter_restore_verified"]
    with pytest.raises(ValueError, match="unavailable"):
        verify(run_dir, 99)

    a = tmp_path / "synthetic-evaluation-a"
    summary = run(str(run_dir), ["hellaswag", "arc_easy", "piqa"], limit=3, batch_size=2,
                  num_fewshot=0, out_dir=str(a), rq41=True, eval_seed=7, synthetic=True,
                  task_revisions={name: "a" * 40 for name in ("hellaswag", "arc_easy", "piqa")})
    assert summary["step"] == 0
    rows = read_records(a / "records.jsonl")
    assert len(rows) == 9
    assert {row["split"] for row in rows.values()} == {"validation", "test"}
    assert all(row["synthetic"] and row["model"]["training_tokens"] == 0 for row in rows.values())
    assert all(row["task_definition"]["dataset_fingerprint"] for row in rows.values())
    assert all(row["task_definition"]["requested_dataset_revision"] == "a" * 40 for row in rows.values())
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["model"]["loaded_step"] == 0
    assert manifest["expected_counts"] == {"hellaswag": 3, "arc_easy": 3, "piqa": 3}
    assert Path(manifest["provenance"]["package_path"]).resolve().parent.name == "gw1b"

    # Repeat actual inference at the same batch/precision; this is a determinism check, not a new seed.
    b = tmp_path / "synthetic-evaluation-b"
    run(str(run_dir), ["hellaswag", "arc_easy", "piqa"], limit=3, batch_size=2,
        num_fewshot=0, out_dir=str(b), rq41=True, eval_seed=7, synthetic=True,
        task_revisions={name: "a" * 40 for name in ("hellaswag", "arc_easy", "piqa")})
    report = analyze(a / "records.jsonl", b / "records.jsonl")
    assert all(item["difference_b_minus_a"]["ci95"] == [0.0, 0.0] for item in report["tasks"])
    b_rows = read_records(b / "records.jsonl")
    assert [rows[k]["candidates"] for k in sorted(rows)] == [b_rows[k]["candidates"] for k in sorted(rows)]
    with pytest.raises(FileExistsError):
        run(str(run_dir), ["piqa"], limit=3, num_fewshot=0, out_dir=str(a), rq41=True)


def test_legacy_aggregate_flow_is_default(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    lm_eval = pytest.importorskip("lm_eval")
    from types import SimpleNamespace
    from gw1b import evaluate, lm_eval_adapter

    monkeypatch.setattr(evaluate, "load_run", lambda *args, **kwargs:
                        (object(), SimpleNamespace(data=SimpleNamespace(tokenizer="unused"),
                                                   model=SimpleNamespace(n_params=1)), 123))
    monkeypatch.setattr(lm_eval_adapter, "Tokenizer", lambda *args: object())
    monkeypatch.setattr(lm_eval_adapter, "GW1BLM", lambda *args, **kwargs: object())
    def fake_evaluate(**kwargs):
        assert kwargs["log_samples"] is False
        assert kwargs["tasks"] == ["other_task"]
        assert "random_seed" not in kwargs
        return {"results": {"other_task": {"acc,none": 0.5}}}
    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_evaluate)
    result = lm_eval_adapter.run("unused", ["other_task"], out_dir=str(tmp_path))
    assert result["step"] == 123
    assert (tmp_path / "step123_other_task.json").exists()
    assert not (tmp_path / "records.jsonl").exists()
