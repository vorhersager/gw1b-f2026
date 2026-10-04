"""Configuration for GW1B experiments.

A run is described by one YAML file with four sections (model / data / optim / run).
Every field has a default, so a YAML file only needs to list what it changes.
Command-line overrides use dotted paths:  --set model.n_layers=12 --set optim.lr=6e-4
"""
from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    # --- size ---------------------------------------------------------------
    vocab_size: int = 32000
    d_model: int = 1792
    n_layers: int = 32
    n_heads: int = 28
    n_kv_heads: int = 7          # == n_heads -> multi-head attention; 1 -> multi-query; else GQA
    head_dim: int = 64
    d_ff: int = 4864
    max_seq_len: int = 2048
    # --- architecture knobs (Team 2 ablations) -------------------------------
    activation: str = "swiglu"   # "swiglu" | "gelu" | "relu"
    norm: str = "rmsnorm"        # "rmsnorm" | "layernorm"
    pos: str = "rope"            # "rope" | "learned"
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # --- numerics ------------------------------------------------------------
    dtype: str = "auto"          # compute dtype: "auto" = bfloat16 on GPUs that have it (A100/L40S/H100/Blackwell),
                                 # float32 on V100 and CPU (no bf16 tensor cores) | "bfloat16" | "float16" | "float32"
    param_dtype: str = "float32" # master weights
    init_std: float = 0.02
    attn_implementation: str = "auto"  # "auto" | "xla" | "cudnn"  (cudnn = flash attention, Ampere+ only)

    @property
    def n_params(self) -> int:
        """Exact parameter count (used for the compute budget)."""
        d, L, V = self.d_model, self.n_layers, self.vocab_size
        q = d * self.n_heads * self.head_dim
        kv = 2 * d * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * d
        attn = q + kv + o
        if self.activation == "swiglu":
            mlp = 3 * d * self.d_ff
        else:
            mlp = 2 * d * self.d_ff
        norms = 2 * d if self.norm == "rmsnorm" else 4 * d
        per_layer = attn + mlp + norms
        emb = V * d
        head = 0 if self.tie_embeddings else V * d
        final_norm = d if self.norm == "rmsnorm" else 2 * d
        pos = self.max_seq_len * d if self.pos == "learned" else 0
        return L * per_layer + emb + head + final_norm + pos

    @property
    def n_params_non_embedding(self) -> int:
        return self.n_params - self.vocab_size * self.d_model


@dataclass(frozen=True)
class DataConfig:
    train_dir: str = "${GW1B_SCRATCH}/data/fineweb-edu-10B/train"
    val_dir: str = "${GW1B_SCRATCH}/data/fineweb-edu-10B/val"
    tokenizer: str = "${GW1B_GROUP}/tokenizer/gw1b-32k.model"
    seq_len: int = 2048
    shuffle: bool = True
    seed: int = 0
    prefetch: int = 4
    num_workers: int = 2  # CPU threads that assemble batches


@dataclass(frozen=True)
class OptimConfig:
    name: str = "adamw"          # "adamw" | "lion" | "sgd" (Team 3 can add more in train.py)
    lr: float = 3e-4
    min_lr_ratio: float = 0.1    # final LR = lr * min_lr_ratio (cosine) ; ignored by "constant"
    schedule: str = "cosine"     # "cosine" | "wsd" | "constant" | "linear"
    warmup_steps: int = 1000
    decay_fraction: float = 0.1  # WSD only: fraction of total steps spent decaying at the end
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int | None = None  # micro-steps per optimizer step; None/auto = the largest micro-batch that fits the GPU
    decay_norm_and_bias: bool = False  # apply weight decay to 1-D params? (Llama recipes: no)


@dataclass(frozen=True)
class RunConfig:
    name: str = "debug"
    out_dir: str = "${GW1B_SCRATCH}/users/${USER}/runs"
    batch_size: int = 256        # sequences per optimizer step (global, across all GPUs)
    total_steps: int = 10000
    total_tokens: int | None = None  # if set, overrides total_steps = total_tokens / (batch_size*seq_len)
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 20
    ckpt_every: int = 1000
    ckpt_keep: int = 3
    ckpt_keep_every: int | None = None  # additionally keep every N-th step forever (Team 5 checkpoint studies)
    seed: int = 0
    shard_params: bool = True    # FSDP: shard parameters + optimizer state across GPUs
    remat: bool = False          # gradient checkpointing per block: ~4x less activation memory, ~30 % more compute
    loss_chunk: int = 512        # positions per chunk of the memory-efficient cross-entropy (0 = full [B,T,V] logits)
    profile: bool = False        # write a JAX profiler trace for the first steps
    wandb: bool = False          # log to Weights & Biases (needs WANDB_API_KEY; use WANDB_MODE=offline on compute nodes)
    tensorboard: bool = True
    gpu_hour_price_usd: float | None = None  # notional $/GPU-hour for the "$" every run reports; None = by GPU type
                                            # (budget.GPU_PEAK_TFLOPS: V100 $1, A100 $2, L40S $1.5, RTX 6000 $2.5, H100 $3.5)


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def tokens_per_step(self) -> int:
        return self.run.batch_size * self.data.seq_len

    def resolved_total_steps(self) -> int:
        if self.run.total_tokens:
            return max(1, int(self.run.total_tokens // self.tokens_per_step))
        return self.run.total_steps

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


# ---------------------------------------------------------------------------
# loading / merging
# ---------------------------------------------------------------------------
_SECTIONS = {"model": ModelConfig, "data": DataConfig, "optim": OptimConfig, "run": RunConfig}


def _base_type(t: Any) -> Any:
    """`int | None` -> int ; `int` -> int."""
    if isinstance(t, types.UnionType):
        args = [a for a in t.__args__ if a is not type(None)]
        return args[0] if len(args) == 1 else t
    return t


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a value (from YAML or the command line) to the type of the dataclass field.

    Strings are coerced because PyYAML reads `3e-4` as a string and the CLI always gives strings.
    """
    t = _base_type(target_type)
    optional = isinstance(target_type, types.UnionType) and type(None) in target_type.__args__
    if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "")):
        return None
    if optional and isinstance(value, str) and value.strip().lower() == "auto":
        return None  # e.g. optim.grad_accum: auto
    if t is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if t is int:
        if isinstance(value, str):
            return int(float(value))
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return int(value)
    if t is float:
        return float(value)
    if t is str and isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))  # expand ${GW1B_SCRATCH}, ~ ...
    return value


def _build(section_cls, values: dict[str, Any]):
    known = typing.get_type_hints(section_cls)
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {section_cls.__name__} field(s): {sorted(unknown)}. "
                         f"Valid fields: {sorted(known)}")
    kwargs = {}
    for f in dataclasses.fields(section_cls):  # defaults go through _coerce too, so ${ENV} in defaults expands
        raw = values.get(f.name, f.default if f.default is not dataclasses.MISSING else f.default_factory())
        kwargs[f.name] = _coerce(raw, known[f.name])
    return section_cls(**kwargs)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_config_path(path: str) -> str:
    """Find a config file given as `configs/50m.yaml`, `50m.yaml` or just `50m`, from any working directory.

    Order: the path as given (absolute or relative to the current directory), then the same path under the
    repository root (`$GW1B_HOME`, where the configs/ folder lives), then configs/<basename> there. So
    `gw1b train --config configs/50m.yaml` works from $HOME as well as from a checkout, and a student's own
    copy in ./configs/ wins when they run from its parent directory.
    """
    if not path:
        return path
    names = [path] if path.endswith((".yaml", ".yml")) else [path, path + ".yaml"]
    roots = [r for r in (os.environ.get("GW1B_HOME"), REPO_ROOT) if r]
    candidates: list[str] = []
    for n in names:
        candidates.append(n)
    for r in roots:
        for n in names:
            candidates.append(os.path.join(r, n))
            candidates.append(os.path.join(r, "configs", os.path.basename(n)))
    seen: set[str] = set()
    for c in candidates:
        c = os.path.expanduser(os.path.expandvars(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    looked = ", ".join(dict.fromkeys(candidates))
    raise FileNotFoundError(f"config {path!r} not found (looked for: {looked}). "
                            f"Available: {', '.join(sorted(os.listdir(os.path.join(REPO_ROOT, 'configs'))))}")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config (optionally chained with `extends: other.yaml`) and apply overrides.

    `path` may be `configs/50m.yaml`, `50m.yaml` or `50m` (see resolve_config_path).
    """
    raw: dict[str, Any] = {}
    if path:
        raw = _load_yaml_with_extends(resolve_config_path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.field=value, got {item!r}")
        key, value = item.split("=", 1)
        section, _, name = key.partition(".")
        if section not in _SECTIONS or not name:
            raise ValueError(f"Override key must be one of {list(_SECTIONS)}.<field>, got {key!r}")
        raw.setdefault(section, {})[name] = value
    sections = {}
    for name, cls in _SECTIONS.items():
        sections[name] = _build(cls, raw.get(name, {}) or {})
    cfg = TrainConfig(**sections)
    _validate(cfg)
    return cfg


def _load_yaml_with_extends(path: str) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        base = _load_yaml_with_extends(parent_path)
        for section, values in raw.items():
            base.setdefault(section, {}).update(values or {})
        return base
    return raw


def _validate(cfg: TrainConfig) -> None:
    m = cfg.model
    assert m.n_heads % m.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
    assert m.activation in ("swiglu", "gelu", "relu")
    assert m.norm in ("rmsnorm", "layernorm")
    assert m.pos in ("rope", "learned")
    assert m.head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert m.dtype in ("auto", "bfloat16", "float16", "float32"), f"model.dtype={m.dtype!r}: auto|bfloat16|float16|float32"
    assert m.param_dtype in ("float32", "bfloat16", "float16"), f"model.param_dtype={m.param_dtype!r}"
    assert m.attn_implementation in ("auto", "xla", "cudnn"), f"model.attn_implementation={m.attn_implementation!r}"
    assert cfg.data.seq_len <= m.max_seq_len, "data.seq_len must be <= model.max_seq_len"
    if cfg.optim.grad_accum:
        assert cfg.run.batch_size % cfg.optim.grad_accum == 0, "batch_size must be divisible by grad_accum"
    assert cfg.run.loss_chunk >= 0
    assert cfg.optim.schedule in ("cosine", "wsd", "constant", "linear")


def config_summary(cfg: TrainConfig) -> str:
    m = cfg.model
    lines = [
        f"model: {m.n_params/1e6:,.1f}M params ({m.n_params_non_embedding/1e6:,.1f}M non-embedding)  "
        f"L={m.n_layers} d={m.d_model} heads={m.n_heads}/{m.n_kv_heads}kv ff={m.d_ff} vocab={m.vocab_size} "
        f"ctx={cfg.data.seq_len} {m.activation}/{m.norm}/{m.pos} tie={m.tie_embeddings} dtype={m.dtype}",
        f"optim: {cfg.optim.name} lr={cfg.optim.lr} {cfg.optim.schedule} warmup={cfg.optim.warmup_steps} "
        f"wd={cfg.optim.weight_decay} clip={cfg.optim.grad_clip} accum={cfg.optim.grad_accum or 'auto'}"
        f"{' remat' if cfg.run.remat else ''}",
        f"run:   batch={cfg.run.batch_size} seqs x {cfg.data.seq_len} = {cfg.tokens_per_step:,} tokens/step, "
        f"{cfg.resolved_total_steps():,} steps = {cfg.resolved_total_steps()*cfg.tokens_per_step/1e9:.2f}B tokens",
    ]
    return "\n".join(lines)


def save_config(cfg: TrainConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(cfg.to_yaml())


def config_from_dict(d: dict[str, Any]) -> TrainConfig:
    return TrainConfig(**{name: _build(cls, d.get(name, {}) or {}) for name, cls in _SECTIONS.items()})


def config_from_json(path: str) -> TrainConfig:
    with open(path) as f:
        return config_from_dict(json.load(f))
