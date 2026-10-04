"""CPU smoke tests for the gw1b package.  Run:  python -m pytest tests -q   (about 3 minutes on a laptop)

They cover: model forward + KV cache, sharded (8 fake devices) FSDP training step, data loader
determinism, config loading, a 40-step training run with checkpoint + resume, generation,
and the HF export (numerical check only if transformers+torch are installed).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gw1b.config import ModelConfig, load_config  # noqa: E402
from gw1b.data import BatchLoader, TokenDataset, write_synthetic_dataset  # noqa: E402
from gw1b.model import GW1BModel, count_params, cross_entropy  # noqa: E402

TINY = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
            max_seq_len=64, dtype="float32")


def test_param_count_matches_formula():
    for kw in [{}, dict(activation="gelu"), dict(norm="layernorm"), dict(pos="learned"), dict(tie_embeddings=False),
               dict(n_kv_heads=4), dict(n_kv_heads=1)]:
        cfg = ModelConfig(**{**TINY, **kw})
        assert count_params(GW1BModel(cfg, rngs=nnx.Rngs(0))) == cfg.n_params


def test_design_doc_config_is_1p15b():
    cfg = load_config(os.path.join(ROOT, "configs", "gw1b_1p15b.yaml"))
    assert abs(cfg.model.n_params / 1e9 - 1.15) < 0.01


def test_config_found_from_any_directory(tmp_path, monkeypatch):
    """`gw1b train --config configs/50m.yaml` runs from $HOME: the path is resolved against the repo as well."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GW1B_HOME", raising=False)
    ref = load_config(os.path.join(ROOT, "configs", "50m.yaml"))
    for spec in ("configs/50m.yaml", "50m.yaml", "50m"):
        assert load_config(spec).model == ref.model, spec
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "50m.yaml").write_text("model: {n_layers: 3}\n")   # a student's own copy wins where it exists
    assert load_config("configs/50m.yaml").model.n_layers == 3
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("configs/nope.yaml")


def test_kv_cache_matches_full_forward():
    cfg = ModelConfig(**TINY)
    model = GW1BModel(cfg, rngs=nnx.Rngs(0))
    toks = jax.random.randint(jax.random.key(1), (2, 16), 0, cfg.vocab_size)
    full = model(toks)
    cache = model.init_cache(2, 32)
    lg, cache = model(toks[:, :10], cache=cache, pos=0)
    lg2, cache = model(toks[:, 10:], cache=cache, pos=10)
    assert jnp.allclose(jnp.concatenate([lg, lg2], 1), full, atol=1e-4)


def test_dtype_auto_and_v100_attention_fallback(monkeypatch):
    """`dtype: auto` is float32 on CPU; on a GPU without bf16 dots (V100) attention is computed in f32."""
    from gw1b import model as model_lib
    from gw1b.sharding import describe_dtype, resolve_config
    from gw1b.config import TrainConfig
    assert ModelConfig().dtype == "auto"
    cfg = ModelConfig(**{**TINY, "dtype": "auto"})
    assert model_lib.resolve_dtype(cfg).dtype == "float32"          # CPU: no bf16 hardware
    assert GW1BModel(cfg, rngs=nnx.Rngs(0)).cfg.dtype == "float32"  # the model stores the resolved config
    assert resolve_config(TrainConfig(model=cfg)).model.dtype == "float32"
    assert "auto -> float32" in describe_dtype(TrainConfig(model=cfg))
    with pytest.raises(AssertionError):
        load_config(None, ["model.dtype=fp8"])

    # bf16 model: the fallback (what a V100 gets) must give the same attention as the regular XLA path
    bf = ModelConfig(**{**TINY, "dtype": "bfloat16"})
    model = GW1BModel(bf, rngs=nnx.Rngs(0))
    toks = jax.random.randint(jax.random.key(2), (2, 16), 0, bf.vocab_size)
    regular = model(toks)
    lowered = nnx.jit(lambda m, t: m(t)).lower(model, toks).as_text()
    assert "lhs_precision_type = bf16" in lowered                   # the explicit BF16_BF16_F32 dot algorithm ...
    monkeypatch.setattr(model_lib, "xla_bf16_dot_supported", lambda: False)
    lowered = nnx.jit(lambda m, t: m(t)).lower(model, toks).as_text()
    assert "lhs_precision_type = bf16" not in lowered               # ... is gone on a V100 (it does not compile there)
    fallback = model(toks)
    cache = model.init_cache(2, 16)
    decoded, _ = model(toks, cache=cache, pos=0)
    assert regular.dtype == fallback.dtype == jnp.float32
    assert jnp.allclose(regular, fallback, atol=5e-2, rtol=5e-2)   # bf16 rounding differs slightly
    assert jnp.allclose(decoded, fallback, atol=5e-2, rtol=5e-2)


def test_chunked_loss_remat_and_micro_batch():
    """model.loss (chunked cross-entropy) and run.remat give the full computation's loss and gradients;
    choose_micro_batch fits the micro-batch to the GPU memory."""
    from gw1b.budget import activation_bytes_per_seq, choose_micro_batch
    from gw1b.config import TrainConfig, OptimConfig, RunConfig, DataConfig
    toks = jax.random.randint(jax.random.key(1), (3, 64), 0, 512)
    tg = jnp.roll(toks, -1, axis=1)
    model = GW1BModel(ModelConfig(**TINY), rngs=nnx.Rngs(0))
    lf, gf = nnx.value_and_grad(lambda m: cross_entropy(m(toks), tg))(model)
    lc, gc = nnx.value_and_grad(lambda m: m.loss(toks, tg, chunk=16))(model)
    assert abs(float(lf) - float(lc)) < 1e-5
    assert max(jax.tree.leaves(jax.tree.map(lambda a, b: float(jnp.max(jnp.abs(a - b))), gf, gc))) < 1e-6
    mask = (toks % 3 == 0)
    assert abs(float(model.loss(toks, tg, mask=mask, chunk=16)) - float(cross_entropy(model(toks), tg, mask))) < 1e-5
    rem = GW1BModel(ModelConfig(**TINY), rngs=nnx.Rngs(0), remat=True)
    lr, gr = nnx.value_and_grad(lambda m: m.loss(toks, tg, chunk=16))(rem)
    assert abs(float(lr) - float(lf)) < 1e-5
    assert max(jax.tree.leaves(jax.tree.map(lambda a, b: float(jnp.max(jnp.abs(a - b))), gf, gr))) < 1e-6

    class FakeV100:  # 16 GB, JAX reports the 75 % the allocator may use
        device_kind = "Tesla V100-SXM2-16GB"
        def memory_stats(self):
            return {"bytes_limit": int(0.75 * 16e9)}
    m50 = ModelConfig(vocab_size=32000, d_model=512, n_layers=12, n_heads=8, n_kv_heads=2, head_dim=64, d_ff=1408,
                      max_seq_len=2048, dtype="float32")
    cfg = TrainConfig(model=m50, data=DataConfig(seq_len=2048), optim=OptimConfig(), run=RunConfig(batch_size=128))
    micro, ga, why = choose_micro_batch(cfg, "xla", 1, 1, device=FakeV100())
    assert micro * ga == 128 and 1 <= micro <= 32, (micro, ga, why)
    assert activation_bytes_per_seq(m50, 2048, "float32", "xla", True, 512) < activation_bytes_per_seq(m50, 2048, "float32", "xla", False, 512)
    assert activation_bytes_per_seq(m50, 2048, "float32", "xla", False, 512) < activation_bytes_per_seq(m50, 2048, "float32", "xla", False, 0)
    cfg2 = TrainConfig(model=m50, data=DataConfig(seq_len=2048), optim=OptimConfig(grad_accum=4), run=RunConfig(batch_size=128))
    assert choose_micro_batch(cfg2, "xla", 2, 2, device=FakeV100())[:2] == (16, 4)   # explicit grad_accum, 2 devices
    assert load_config(None, ["optim.grad_accum=auto"]).optim.grad_accum is None
    assert load_config(None, ["model.dtype=auto"]).model.dtype == "auto"


def test_sharded_training_step_reduces_loss():
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
    import optax
    cfg = ModelConfig(**TINY)
    mesh = jax.make_mesh((jax.device_count(),), ("data",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        @nnx.jit
        def create():
            return GW1BModel(cfg, rngs=nnx.Rngs(0), n_shards=mesh.size)
        model = create()
        assert model.blocks[0].mlp.up_proj.kernel[...].sharding.spec != P(None, None) or mesh.size == 1
        opt = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

        @nnx.jit
        def step(model, opt, batch):
            def loss_fn(m):
                return cross_entropy(m(batch["inputs"]), batch["targets"])
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            return loss
        toks = jax.device_put(jax.random.randint(jax.random.key(0), (16, 17), 0, cfg.vocab_size),
                              NamedSharding(mesh, P("data", None)))
        batch = {"inputs": toks[:, :-1], "targets": toks[:, 1:]}
        losses = [float(step(model, opt, batch)) for _ in range(20)]
    assert losses[-1] < losses[0] - 0.5


def test_loader_is_deterministic_and_resumable(tmp_path):
    write_synthetic_dataset(str(tmp_path / "train"), n_tokens=50_000, shard_size=20_000)
    ds = TokenDataset(str(tmp_path / "train"), 32)
    l1 = BatchLoader(ds, 8, seed=3); b0, b1 = next(l1), next(l1); l1.close()
    l2 = BatchLoader(ds, 8, seed=3, start_step=1); assert np.array_equal(next(l2)["inputs"], b1["inputs"]); l2.close()
    parts = []
    for r in range(4):
        l = BatchLoader(ds, 8, seed=3, rank=r, world=4); parts.append(next(l)["inputs"]); l.close()
    assert np.array_equal(np.concatenate(parts), b0["inputs"])
    assert np.array_equal(b0["inputs"][:, 1:], b0["targets"][:, :-1])


def test_train_checkpoint_resume(tmp_path):
    env = {**os.environ, "GW1B_SCRATCH": str(tmp_path), "GW1B_GROUP": str(tmp_path), "USER": "test"}
    write_synthetic_dataset(str(tmp_path / "data/synthetic/train"), n_tokens=60_000, shard_size=30_000)
    write_synthetic_dataset(str(tmp_path / "data/synthetic/val"), n_tokens=10_000, seed=1, prefix="val")
    base = [sys.executable, "-m", "gw1b.train", "--config", os.path.join(ROOT, "configs", "tiny_debug.yaml"),
            "--set", "run.log_every=10", "--set", "run.ckpt_every=20", "--set", "run.eval_every=20", "--set", "run.eval_batches=2"]
    out1 = subprocess.run(base + ["--set", "run.total_steps=20"], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out1.returncode == 0, out1.stderr[-2000:]
    assert "checkpoint saved at step 20" in out1.stdout
    out2 = subprocess.run(base + ["--set", "run.total_steps=30"], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out2.returncode == 0, out2.stderr[-2000:]
    assert "resumed from step 20" in out2.stdout and "step 30/30" in out2.stdout
    assert os.path.exists(tmp_path / "users/test/runs/tiny_debug/summary.json")


def test_viz_export_and_server(tmp_path):
    """gw1b.viz: a checkpoint reduces to tiles + stats; the server lists runs and serves the export and the page."""
    import json
    import threading
    import urllib.request
    from gw1b.config import load_config
    from gw1b.train import train
    from gw1b.viz import export as vx, server as vs
    write_synthetic_dataset(str(tmp_path / "train"), n_tokens=20_000, shard_size=10_000)
    cfg = load_config(os.path.join(ROOT, "configs", "tiny_debug.yaml"),
                      [f"data.train_dir={tmp_path}/train", "data.val_dir=", f"run.out_dir={tmp_path}/runs", "run.name=viz",
                       "run.total_steps=4", "run.ckpt_every=2", "run.log_every=2", "run.eval_every=1000", "run.tensorboard=false"])
    train(cfg, resume=False)
    run_dir = str(tmp_path / "runs" / "viz")
    assert vx.checkpoint_steps(run_dir) == [2, 4]
    data = vx.export_run(run_dir, tile=8)
    assert data["step"] == 4 and data["compare_step"] == 2 and data["n_params"] == cfg.model.n_params
    ids = {t["id"] for t in data["tensors"]}
    assert "embed.embedding" in ids and "blocks.0.attn.q_proj.kernel" in ids and "final_norm.scale" in ids
    q = next(t for t in data["tensors"] if t["id"] == "blocks.0.attn.q_proj.kernel")
    assert q["layer"] == 0 and q["group"] == "attn" and q["label"] == "Q" and q["rows"] <= 8 and q["cols"] <= 8
    assert len(q["rms"]) == q["rows"] * q["cols"] == len(q["mean"]) == len(q["delta"]) and q["delta_stats"]["rel"] > 0
    assert all(v >= 0 for v in q["rms"]) and abs(q["stats"]["rms"] - cfg.model.init_std) < 0.01
    emb = next(t for t in data["tensors"] if t["id"] == "embed.embedding")
    assert emb["shape"] == [cfg.model.vocab_size, cfg.model.d_model] and emb["rows"] == 8
    vec = next(t for t in data["tensors"] if t["id"] == "final_norm.scale")
    assert vec["rows"] == 1 and vec["cols"] == min(cfg.model.d_model, 8 * 8)   # vectors: at most tile*tile cells
    assert data["metrics"].get("train/loss") and data["metrics"]["step"] <= 4
    json.loads(vx.dumps(data))                                   # valid JSON (no NaN)
    # the checkpoint above is FSDP-sharded over 8 fake devices; a 1-device process (the viz server on CPU) must
    # still read it (Orbax "Topology mismatch" otherwise)
    env = {**os.environ, "XLA_FLAGS": "", "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-m", "gw1b.viz.export", "--run", run_dir, "--tile", "4",
                          "--out", str(tmp_path / "one-device.json")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.load(open(tmp_path / "one-device.json"))["step"] == 4
    html = vx.write_html(data, str(tmp_path / "model.html"))
    assert os.path.getsize(html) > 500_000 and 'id="gw1b-data"' in open(html).read()

    httpd = vs.serve([str(tmp_path / "runs")], host="127.0.0.1", port=0, tile=8)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True); th.start()
    try:
        get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30).read()
        runs = json.loads(get("/api/runs"))["runs"]
        assert [r["name"] for r in runs] == ["viz"] and runs[0]["latest"] == 4
        exp = json.loads(get("/api/export?run=viz&step=4"))
        assert exp["step"] == 4 and exp["compare_step"] == 2
        assert os.path.exists(os.path.join(run_dir, "viz", "step-4-vs-2.json"))   # disk cache
        assert json.loads(get("/api/export?run=viz&step=2&compare=none"))["compare_step"] is None
        assert b"GW1B model visualizer" in get("/") and b"OrbitControls" in get("/vendor/OrbitControls.js")
        with pytest.raises(urllib.error.HTTPError):
            get("/api/export?run=nope")
    finally:
        httpd.shutdown(); httpd.server_close()


def test_generate_batched_equals_single(tmp_path):
    from gw1b.generate import generate
    from gw1b.tokenizer import Tokenizer, train_sentencepiece
    rng = np.random.default_rng(0)
    syll = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    words = ["".join(rng.choice(syll, rng.integers(1, 4))) for _ in range(800)]
    corpus = tmp_path / "corpus.txt"
    with open(corpus, "w") as f:
        for _ in range(3000):
            f.write(" ".join(rng.choice(words, rng.integers(3, 12))) + ".\n")
    tok = Tokenizer(train_sentencepiece(str(corpus), str(tmp_path / "tok"), vocab_size=400, input_sentence_size=3000))
    model = GW1BModel(ModelConfig(**{**TINY, "vocab_size": 400, "max_seq_len": 128}), rngs=nnx.Rngs(0))
    prompts = ["alpha beta", "students train pegasus jax model tokens"]
    batched = generate(model, tok, prompts, max_new_tokens=6)
    single = [generate(model, tok, p, max_new_tokens=6)[0] for p in prompts]
    assert batched == single


def test_hf_export(tmp_path):
    pytest.importorskip("safetensors")
    from gw1b.export_hf import export, to_hf_state_dict
    model = GW1BModel(ModelConfig(**TINY), rngs=nnx.Rngs(0))
    sd = to_hf_state_dict(model)
    assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (64, 64)
    assert sd["model.layers.0.self_attn.k_proj.weight"].shape == (32, 64)
    export(model, str(tmp_path / "hf"), dtype="float32")
    assert (tmp_path / "hf" / "model.safetensors").exists()
    try:
        import torch, transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers/torch not installed: skipping numerical verification")
    from gw1b.export_hf import verify
    assert verify(model, str(tmp_path / "hf")) < 1e-3
