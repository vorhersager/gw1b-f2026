#!/usr/bin/env bash
# Regenerate the fully pinned lock file for the core environment (run anywhere with `uv`).
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile requirements.in -o requirements.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-annotate --no-header
echo "wrote requirements.txt"
