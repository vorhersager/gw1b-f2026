"""Device mesh, distributed initialization and sharded model creation.

One mesh axis ("data") is used for everything:
    * batches are split along it (data parallelism), and
    * with run.shard_params=True every parameter / optimizer moment is split along it as well (FSDP),
      so a 1.15B model fits comfortably even on 16 GB V100s.
XLA inserts the all-gathers / reduce-scatters automatically (GSPMD).

Multi-node jobs launch one process per GPU (`srun --ntasks-per-node=<gpus>`); JAX detects Slurm and
wires the processes together in `init_distributed()`.
"""
from __future__ import annotations

import os

import jax
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from .config import TrainConfig
from .model import MESH_AXIS, GW1BModel, resolve_attn_impl


def init_distributed() -> bool:
    """Initialise multi-process JAX when launched with `srun` and more than one task. Returns True if so."""
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    forced = os.environ.get("GW1B_DISTRIBUTED", "") == "1"
    if (ntasks > 1 and "SLURM_STEP_NODELIST" in os.environ) or forced:
        jax.distributed.initialize()  # coordinator/process ids come from SLURM_* variables
        return True
    return False


def make_mesh(n_devices: int | None = None) -> Mesh:
    n = n_devices or jax.device_count()
    return jax.make_mesh((n,), (MESH_AXIS,), axis_types=(AxisType.Auto,))


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P(MESH_AXIS, None))


def create_model(cfg: TrainConfig, mesh: Mesh, seed: int | None = None) -> GW1BModel:
    """Create the model directly sharded across the mesh (params never materialise on one device)."""
    n_shards = mesh.size if cfg.run.shard_params else 1
    attn_impl = resolve_attn_impl(cfg.model)
    seed = cfg.run.seed if seed is None else seed

    with jax.set_mesh(mesh):
        @nnx.jit
        def _create():
            model = GW1BModel(cfg.model, rngs=nnx.Rngs(seed), n_shards=n_shards, attn_impl=attn_impl)
            if n_shards == 1 and mesh.size > 1:  # pure data parallel: replicate parameters explicitly
                state = nnx.state(model)
                state = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, P()), state)
                nnx.update(model, state)
            return model

        return _create()


def put_batch(batch: dict[str, np.ndarray], mesh: Mesh) -> dict[str, jax.Array]:
    """Move a (process-local) numpy batch onto the mesh, sharded along the batch axis."""
    sharding = batch_sharding(mesh)
    if jax.process_count() == 1:
        return {k: jax.device_put(v, sharding) for k, v in batch.items()}
    return {k: jax.make_array_from_process_local_data(sharding, v) for k, v in batch.items()}


def describe_devices() -> str:
    devs = jax.devices()
    kinds = sorted({d.device_kind for d in devs})
    return (f"{len(devs)} device(s) [{', '.join(kinds)}] on {jax.process_count()} process(es); "
            f"this process: {jax.process_index()} with {jax.local_device_count()} local device(s)")
