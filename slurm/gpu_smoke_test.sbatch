#!/bin/bash
# 3-minute GPU smoke test — submitted by `gw1b doctor --gpu`.
#SBATCH --job-name=gw1b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
set -euo pipefail
source "$GW1B_HOME/gw1b.env"
echo "node: $(hostname -s)  partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true
"$GW1B_HOME/bin/gw1b-exec" python - <<'PY'
import time, jax, jax.numpy as jnp
from gw1b.model import device_bf16_support
print("jax", jax.__version__, "backend:", jax.default_backend())
devs = jax.devices(); print("devices:", devs)
assert devs[0].platform == "gpu", "JAX did not find a GPU — see docs/TROUBLESHOOTING.md (driver version?)"
bf16, why = device_bf16_support()
dtype = jnp.bfloat16 if bf16 else jnp.float32   # V100 (compute capability 7.0) has no bf16 tensor cores
print(f"compute dtype for this GPU: {'bfloat16' if bf16 else 'float32'} — {why}")
x = jnp.ones((8192, 8192), dtype)
f = jax.jit(lambda a: (a @ a).sum()); f(x).block_until_ready()
t = time.time(); [f(x).block_until_ready() for _ in range(10)]; dt = (time.time() - t) / 10
print(f"{jnp.dtype(dtype).name} 8192^2 matmul: {2*8192**3/dt/1e12:.1f} TFLOP/s on {devs[0].device_kind}")
PY
echo "--- tiny end-to-end training run (model.dtype=auto: bf16 where the GPU has it, else float32) ---"
export GW1B_SCRATCH_TEST="$GW1B_SCRATCH/users/$USER/smoke"
"$GW1B_HOME/bin/gw1b-exec" python - <<'PY'
import os
from gw1b.data import write_synthetic_dataset
S = os.environ["GW1B_SCRATCH_TEST"]
write_synthetic_dataset(f"{S}/data/synthetic/train", n_tokens=200_000, shard_size=100_000)
write_synthetic_dataset(f"{S}/data/synthetic/val", n_tokens=20_000, seed=1, prefix="val")
PY
"$GW1B_HOME/bin/gw1b-exec" python -m gw1b.train --config "$GW1B_HOME/configs/tiny_debug.yaml" \
    --set data.train_dir="$GW1B_SCRATCH_TEST/data/synthetic/train" --set data.val_dir="$GW1B_SCRATCH_TEST/data/synthetic/val" \
    --set run.out_dir="$GW1B_SCRATCH_TEST/runs" --set run.name="smoke-$SLURM_JOB_ID" --set run.total_steps=60 --set model.dtype=auto --no-resume
echo "SMOKE TEST PASSED"
