# Docker twin of gw1b.def — for building the image on a laptop/CI and converting to .sif.
FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false GW1B_ENV_VERSION=2026.09 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true NCCL_DEBUG=WARN
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates openssh-client procps htop tmux vim-tiny less zstd libgomp1 libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-eval.txt /opt/gw1b/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /opt/gw1b/requirements.txt \
    && python -m pip install --no-cache-dir -r /opt/gw1b/requirements-eval.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import jax, flax, optax, orbax.checkpoint, sentencepiece, datasets, lm_eval, torch" \
    && python -m pip freeze > /opt/gw1b/pip-freeze.txt \
    && mkdir -p /lustre /SEAS /CCAS /GWSB /SMHS /SPH /ESIA /CPS /GSEHD /LAW /c1 /groups /scratch /data
CMD ["python"]
