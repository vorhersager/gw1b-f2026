"""Compute budget calculator — "treat GPU-hours (and dollars) as an experimental variable".

    python -m gw1b.budget --config configs/gw1b_1p15b.yaml --tokens 20e9 --gpu a100 --n-gpus 8 --mfu 0.35
    python -m gw1b.budget --params 1.15e9 --tokens 100e9 --gpu a100 --n-gpus 16

Training FLOPs use the standard estimate  C ≈ 6·N·D  (+ attention term 12·L·T·d per token).
"""
from __future__ import annotations

import argparse
import re

# dense (non-sparse) peak TFLOP/s for the matmul precision students will actually use
GPU_PEAK_TFLOPS = {
    # name: (bf16/fp16 tensor-core peak, fp32 peak, memory GB, notional $/GPU-hour on public clouds)
    "v100": (125.0, 15.7, 16, 1.0),        # Pegasus 2x/4x V100 nodes, --gpu-type v100 (no bf16 -> use fp16 or f32)
    "a100": (312.0, 19.5, 80, 2.0),        # Pegasus gpu050/gpu051: 8x A100-80GB PCIe, --gpu-type a100
    "l40s": (362.0, 91.6, 48, 1.5),        # Pegasus L40S nodes, --gpu-type l40s
    "rtx6000": (500.0, 120.0, 96, 2.5),    # RTX PRO 6000 Blackwell 96GB nodes (Oct 2026): 1 PFLOP/s FP16 sparse spec -> ~500 dense; measure!
    "h100": (989.0, 67.0, 96, 3.5),        # Pegasus Grace Hopper nodes (H100 96GB, superChip partition, 1 per node) - ARM host!
    "gh200": (989.0, 67.0, 96, 3.5),       # alias of h100: --gpu-type gh200
    "v6e":  (918.0, 918.0, 32, 2.7),       # TPU Trillium chip (design-doc reference)
}


def gpu_key(device_kind: str) -> str | None:
    k = device_kind.lower()
    for name in GPU_PEAK_TFLOPS:
        if name in k:
            return name
    if "gh200" in k or "hopper" in k:
        return "h100"
    return None


def peak_flops(device_kind: str, dtype: str = "bfloat16") -> float | None:
    key = gpu_key(device_kind)
    if key is None:
        return None
    tc, f32, _, _ = GPU_PEAK_TFLOPS[key]
    if dtype == "float32":
        return f32 * 1e12
    if key == "v100" and dtype == "bfloat16":
        return f32 * 1e12  # V100 has no bf16 tensor cores: bf16 matmuls run ~f32 speed
    return tc * 1e12


def gpu_memory_bytes(device_kind: str) -> float | None:
    """Total memory of a GPU from the table (fallback when JAX cannot report it)."""
    key = gpu_key(device_kind)
    return None if key is None else GPU_PEAK_TFLOPS[key][2] * 1e9


def activation_bytes_per_seq(m, seq_len: int, dtype: str, attn_impl: str, remat: bool, loss_chunk: int) -> float:
    """Rough activation memory one training sequence needs (bytes), for choosing the micro-batch.

    m: ModelConfig. Counts what the backward pass has to keep: the block activations (or only the block
    inputs with remat), one layer's attention scores on the XLA path (they are rematerialised, so one layer
    at a time), and the f32 logits of one loss chunk (logits + log-softmax + gradient).
    """
    db = 2 if dtype in ("bfloat16", "float16") else 4
    T, V, d, L, N, ff = seq_len, m.vocab_size, m.d_model, m.n_layers, m.n_heads, m.d_ff
    per_layer = T * (10 * d + 3 * ff) * db                      # norms, q/k/v/o, residuals, gate/up/silu
    blocks = L * T * d * db + per_layer if remat else L * per_layer
    scores = 0.0 if attn_impl == "cudnn" else N * T * T * (8 + db)   # f32 scores + grad + probs, one layer live
    chunk = loss_chunk if 0 < loss_chunk < T else T
    logits = 3.0 * chunk * V * 4
    return blocks + scores + logits


def choose_micro_batch(cfg, attn_impl: str, n_devices: int, n_shards: int, device=None) -> tuple[int, int, str]:
    """(sequences per device per micro-step, grad_accum, explanation) for cfg.run.batch_size on n_devices.

    optim.grad_accum set -> honoured. Otherwise the largest divisor of the per-device batch whose estimated
    activation memory fits what XLA may allocate on the device (JAX's memory_stats or the GPU table).
    """
    import jax
    m, batch, T = cfg.model, cfg.run.batch_size, cfg.data.seq_len
    assert batch % n_devices == 0, f"run.batch_size={batch} must be a multiple of the {n_devices} devices"
    per_dev = batch // n_devices
    if cfg.optim.grad_accum:
        ga = cfg.optim.grad_accum
        assert per_dev % ga == 0, f"batch_size/n_devices={per_dev} must be divisible by optim.grad_accum={ga}"
        return per_dev // ga, ga, f"optim.grad_accum={ga} from the config"
    dev = device or jax.local_devices()[0]
    limit = None
    try:
        limit = dev.memory_stats().get("bytes_limit")
    except Exception:
        limit = None
    if not limit:
        total = gpu_memory_bytes(dev.device_kind)
        limit = 0.75 * total if total else None
    if not limit:  # CPU or unknown accelerator: no memory model, one micro-step
        return per_dev, 1, f"no memory information for {dev.device_kind}: one micro-step"
    dtype = m.dtype if m.dtype != "auto" else "float32"
    n_params = m.n_params
    static = n_params * 24.0 / max(1, n_shards) + n_params * 4.0 + 1.0e9   # f32 master + grad + Adam, cast copy, XLA workspace
    per_seq = activation_bytes_per_seq(m, T, dtype, attn_impl, cfg.run.remat, cfg.run.loss_chunk)
    usable = max(0.0, (limit - static) * 0.85)
    fit = max(1, int(usable // per_seq))
    micro = max(d for d in range(1, per_dev + 1) if per_dev % d == 0 and d <= fit)
    why = (f"auto: {limit/1e9:.0f} GB usable on {dev.device_kind}, ~{static/1e9:.1f} GB for parameters/optimizer, "
           f"~{per_seq/1e9:.2f} GB per sequence -> up to {fit} per device")
    if per_seq > usable:
        if static > 0.6 * limit:
            why += (f" — even one sequence may not fit: parameters + optimizer state alone take ~{static/1e9:.1f} GB; "
                    f"use more GPUs (--gpus 4: FSDP shards them) or a bigger GPU (--gpu-type a100)")
        else:
            why += " — even one sequence may not fit: try --set run.remat=true, a shorter data.seq_len or more GPUs"
    return micro, per_dev // micro, why


def flops_per_token(n_params: int, n_layers: int, seq_len: int, d_model: int) -> float:
    """6N (fwd+bwd matmuls) + 12·L·T·d (attention scores, fwd+bwd)."""
    return 6.0 * n_params + 12.0 * n_layers * seq_len * d_model


def estimate(n_params: float, tokens: float, gpu: str, n_gpus: int, mfu: float = 0.35, dtype: str = "bfloat16",
             n_layers: int | None = None, seq_len: int | None = None, d_model: int | None = None,
             price_per_gpu_hour: float | None = None) -> dict:
    tc, f32, mem, price = GPU_PEAK_TFLOPS[gpu]
    peak = peak_flops(gpu, dtype)
    if n_layers and seq_len and d_model:
        fpt = flops_per_token(int(n_params), n_layers, seq_len, d_model)
    else:
        fpt = 6.0 * n_params
    total_flops = fpt * tokens
    throughput = peak * mfu * n_gpus  # FLOP/s
    seconds = total_flops / throughput
    gpu_hours = seconds * n_gpus / 3600
    price = price if price_per_gpu_hour is None else price_per_gpu_hour
    return {
        "n_params": n_params, "tokens": tokens, "gpu": gpu, "n_gpus": n_gpus, "mfu": mfu, "dtype": dtype,
        "flops_per_token": fpt, "total_flops": total_flops, "tokens_per_second": tokens / seconds,
        "wall_hours": seconds / 3600, "wall_days": seconds / 86400, "gpu_hours": gpu_hours,
        "usd_equivalent": gpu_hours * price, "chinchilla_tokens": 20 * n_params,
    }


def format_estimate(e: dict) -> str:
    return (f"{e['n_params']/1e9:.2f}B params x {e['tokens']/1e9:.1f}B tokens = {e['total_flops']:.2e} FLOPs\n"
            f"  {e['n_gpus']} x {e['gpu'].upper()} @ {e['mfu']:.0%} MFU ({e['dtype']}): "
            f"{e['tokens_per_second']/1e3:,.0f}k tokens/s -> {e['wall_hours']:.1f} h wall "
            f"({e['wall_days']:.2f} days), {e['gpu_hours']:.0f} GPU-hours "
            f"(~${e['usd_equivalent']:,.0f} at cloud prices)\n"
            f"  Chinchilla-optimal tokens for this size: {e['chinchilla_tokens']/1e9:.1f}B")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GW1B compute budget calculator")
    ap.add_argument("--config", help="YAML config to take model size from")
    ap.add_argument("--params", type=float, help="parameter count if no --config, e.g. 1.15e9")
    ap.add_argument("--tokens", type=float, required=True, help="training tokens, e.g. 20e9")
    ap.add_argument("--gpu", default="a100", choices=sorted(GPU_PEAK_TFLOPS))
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--mfu", type=float, default=0.35, help="assumed model FLOPs utilisation (0.25-0.45 typical)")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args(argv)
    kw = {}
    if a.config:
        from .config import load_config
        cfg = load_config(a.config)
        n = cfg.model.n_params
        kw = dict(n_layers=cfg.model.n_layers, seq_len=cfg.data.seq_len, d_model=cfg.model.d_model)
        print(f"config {a.config}: {n/1e6:,.1f}M params")
    else:
        n = a.params
        if not n:
            ap.error("give --config or --params")
    for gpus in sorted({a.n_gpus, 1, 4, 8, 16}):
        print(format_estimate(estimate(n, a.tokens, a.gpu, gpus, a.mfu, a.dtype, **kw)))
        print()


if __name__ == "__main__":
    main()
