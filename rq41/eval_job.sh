#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script through Slurm}"
run="$1"
out="$2"
step="${3:-}"
[[ "$run" = /* && "$out" = /* ]] || { echo 'Use absolute run/output paths' >&2; exit 2; }
[[ ! -e "$out" && ! -e "$out.verification.json" ]] || { echo 'Output exists; choose a new output path' >&2; exit 1; }
python - <<'PY'
import os, pathlib, subprocess, gw1b
expected = pathlib.Path(os.environ['GW1B_HOME']).resolve()
actual = pathlib.Path(gw1b.__file__).resolve()
assert actual.parent == expected / 'gw1b', (actual, expected)
print('IMPORTED_PACKAGE:', actual, flush=True)
subprocess.run(['git', '-C', str(expected), 'rev-parse', 'HEAD'], check=True)
subprocess.run(['git', '-C', str(expected), 'status', '--short'], check=True)
PY
mkdir -p "$(dirname "$out")"
verify_args=(--run "$run" --out "$out.verification.json")
[[ -z "$step" ]] || verify_args+=(--step "$step")
python -m gw1b.rq41_checkpoint "${verify_args[@]}"
verified_step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["loaded_step"])' "$out.verification.json")"
# Pass the exact verified step; never race a later 'latest' checkpoint.
python -m gw1b.lm_eval_adapter --run "$run" --step "$verified_step" \
    --tasks hellaswag,arc_easy,piqa --num-fewshot 0 --limit 50 --batch-size 2 \
    --rq41 --eval-seed 0 --out "$out"
python - "$out/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest['expected_counts'] == {'hellaswag': 50, 'arc_easy': 50, 'piqa': 50}, manifest['expected_counts']
assert manifest['synthetic'] is False
print('Verified 150 real-task sample records; smoke scores are infrastructure checks only.')
PY
python -m gw1b.rq41_analysis "$out/records.jsonl" --seed 0 --out "$out/analysis.json"
