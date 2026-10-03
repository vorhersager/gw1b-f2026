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
