#!/usr/bin/env bash
# Fallback when Apptainer is not available on Pegasus: a shared virtual environment built from the
# same lock file with `uv` (no root needed, ~5 GB). Everyone then runs the identical package set.
#
#   env/build_venv.sh                 # -> $GW1B_VENV (from gw1b.env), default $GW1B_GROUP/venv
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../gw1b.env"
target="${1:-$GW1B_VENV}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${GW1B_SCRATCH:-/tmp}/tmp/uv-cache-$USER}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$GW1B_GROUP/uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
uv venv --python 3.12 "$target"
uv pip install --python "$target/bin/python" -r "$here/requirements.txt"
uv pip install --python "$target/bin/python" -r "$here/requirements-eval.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$target/bin/python" -c "import jax, flax, optax, orbax.checkpoint, sentencepiece, datasets, lm_eval, torch; print('ok', jax.__version__)"
"$target/bin/python" -m pip freeze > "$target/pip-freeze.txt" 2>/dev/null || "$target/bin/python" -m uv pip freeze > "$target/pip-freeze.txt" || true
chmod -R a+rX "$target" 2>/dev/null || true
echo "venv ready: $target  (set GW1B_RUNTIME=venv in gw1b.env or leave 'auto')"
