#!/usr/bin/env bash
# Build the GW1B container image (instructor / TA, once per semester).
#
#   On Pegasus (if `apptainer` or `singularity` is available and --fakeroot is enabled):
#       env/build_sif.sh                      # -> $GW1B_GROUP/sif/gw1b-<version>.sif
#   On any Linux machine with Apptainer installed:
#       env/build_sif.sh /path/to/out.sif
#   Without Apptainer (laptop with Docker): build a Docker image and convert it
#       docker build -t gw1b -f env/Dockerfile env && \
#       docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD":/output --privileged \
#           quay.io/singularity/docker2singularity:v4.1.0 --name gw1b.sif gw1b
#   then copy the .sif to $GW1B_GROUP/sif/ with scp/Globus.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../gw1b.env"
version="${GW1B_ENV_VERSION:-2026.09}"
out="${1:-$GW1B_GROUP/sif/gw1b-$version.sif}"
mkdir -p "$(dirname "$out")"

ctr="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$ctr" ]]; then
    # try the module system
    if ! command -v module >/dev/null 2>&1; then
        [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
        [[ -n "${LMOD_PKG:-}" && -f "$LMOD_PKG/init/bash" ]] && source "$LMOD_PKG/init/bash" || true
    fi
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    ctr="$(command -v apptainer || command -v singularity || true)"
fi
if [[ -z "$ctr" ]]; then
    echo "No apptainer/singularity found. Build on another machine (see comments) or use env/build_venv.sh." >&2
    exit 1
fi
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${GW1B_SCRATCH:-/tmp}/tmp/apptainer-$USER}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${GW1B_SCRATCH:-/tmp}/tmp/apptainer-cache-$USER}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "Building $out with $ctr (tmp: $APPTAINER_TMPDIR) ..."
cd "$here"
if "$ctr" build --fakeroot "$out" gw1b.def 2>/dev/null; then :
else
    echo "--fakeroot build failed or unsupported; retrying without it (needs root)..."
    "$ctr" build "$out" gw1b.def
fi
ln -sfn "$(basename "$out")" "$(dirname "$out")/gw1b.sif"
echo "Done: $out  (symlink $(dirname "$out")/gw1b.sif)"
echo "Smoke test (on a GPU node):  apptainer exec --nv $out python -c 'import jax; print(jax.devices())'"
