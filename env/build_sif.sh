#!/usr/bin/env bash
# Build (or pull) the GW1B container image (instructor / TA, once per semester).
#
#   On Pegasus — no fakeroot there, so PULL the image GitHub Actions built from env/Dockerfile:
#       env/build_sif.sh --from-image ghcr.io/vorhersager/gw1b-f2026:2026.09   # -> $GW1B_GROUP/sif/gw1b-<version>.sif
#       env/build_sif.sh --arch arm64 --from-image ghcr.io/vorhersager/gw1b-f2026:2026.09-arm64   # GH200 nodes
#     (`apptainer build x.sif docker://...` needs no privileges; it converts the OCI layers to a SIF.
#      Publish the image first: GitHub -> Actions -> "build-image" -> Run workflow, or push a tag env-2026.09.)
#   On a machine where `apptainer build --fakeroot` works:
#       env/build_sif.sh [/path/to/out.sif]                                     # builds from env/gw1b.def
#   Without Apptainer (laptop with Docker): build a Docker image and convert it
#       docker build -t gw1b -f env/Dockerfile env && \
#       docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD":/output --privileged \
#           quay.io/singularity/docker2singularity:v4.1.0 --name gw1b.sif gw1b
#   then copy the .sif to $GW1B_GROUP/sif/ with scp/Globus.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../gw1b.env"
version="${GW1B_ENV_VERSION:-2026.09}"
image=""; arch=""
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --from-image) image="${2:?image name}"; shift 2 ;;
        --arch)       arch="${2:?amd64|arm64}"; shift 2 ;;       # arm64 = image for the GH200 (superChip) nodes
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done
suffix=""; [[ "$arch" == arm64 ]] && suffix="-arm64"
out="${1:-$GW1B_GROUP/sif/gw1b-$version$suffix.sif}"
mkdir -p "$(dirname "$out")"

ctr="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$ctr" ]]; then
    # try the module system
    if ! command -v module >/dev/null 2>&1; then
        [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
        [[ -n "${LMOD_PKG:-}" && -f "$LMOD_PKG/init/bash" ]] && source "$LMOD_PKG/init/bash" || true
        [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash || true
    fi
    module load "${GW1B_APPTAINER_MODULE:-apptainer}" 2>/dev/null || module load singularity 2>/dev/null || true
    ctr="$(command -v apptainer || command -v singularity || true)"
fi
if [[ -z "$ctr" ]]; then
    echo "No apptainer/singularity found. Build on another machine (see comments) or use env/build_venv.sh." >&2
    exit 1
fi
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${GW1B_SCRATCH:-/tmp}/tmp/apptainer-tmp-$USER}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${GW1B_SCRATCH:-/tmp}/tmp/apptainer-cache-$USER}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$here"
if [[ -n "$image" ]]; then
    echo "Pulling docker://$image -> $out with $ctr (cache: $APPTAINER_CACHEDIR; needs ~3x the image size free) ..."
    if [[ "$arch" == arm64 ]]; then "$ctr" build --arch arm64 "$out" "docker://$image"; else "$ctr" build "$out" "docker://$image"; fi
else
    echo "Building $out from gw1b.def with $ctr (tmp: $APPTAINER_TMPDIR) ..."
    if "$ctr" build --fakeroot "$out" gw1b.def 2>/dev/null; then :
    else
        echo "--fakeroot build failed or unsupported (no /etc/subuid entry?); retrying without it (needs root)..."
        "$ctr" build "$out" gw1b.def
    fi
fi
ln -sfn "$(basename "$out")" "$(dirname "$out")/gw1b$suffix.sif"
echo "Done: $out  (symlink $(dirname "$out")/gw1b$suffix.sif)"
echo "Smoke test (on a GPU node):  apptainer exec --nv $out python -c 'import jax; print(jax.devices())'"
