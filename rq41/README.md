# RQ4.1 first milestone

## October 29 real-model pilot

The user authorized canceling tiny training job 138631889 on October 28; Slurm
confirmed cancellation with zero runtime. Use existing research weights instead:
`rq41/pilot_manifest.json` freezes class-run steps 1000/2000, exact input metadata
hashes and a timing-only sample-count rule. `rq41/pilot_revisions.json` pins actual
Hub commits. `PROTOCOL_PILOT.md` is the exploratory protocol; `CLASS_UPDATE_DRAFT.md`
contains unsent announcement/seed-budget/question-form coordination text.

The driver verifies both parameter restores, evaluates 50 examples/task twice,
checks deterministic predictions and score differences <=1e-5, then freezes N from
[500,100,50] using timing before main results. It evaluates both steps and writes
`comparison.json`, `noise_table.md`, `noise_table.csv`, `cost.json` and `complete.json`.
It fails closed and retains logs/partial output if any check fails. Raw data/output
are private and Git-ignored. This is a one-seed 50M pilot, not the three-seed 100M study.

```bash
# Pegasus, private checkout, after committing/pushing and pulling the pilot code:
cd /scratch/gw1b-class/users/g37693783/rq41-dev
sbatch --output=/scratch/gw1b-class/users/g37693783/jobs/rq41-real-%j.out \
  rq41/pilot.sbatch "$PWD" \
  /scratch/gw1b-class/users/g37693783/rq41_outputs/NEW_UNIQUE_PILOT_DIRECTORY
```

One V100, two CPUs, 16 GB, 30-minute maximum; never execute the driver on a login node.
When using `gw1b run -m gw1b.rq41_pilot` directly, pass `--checkout "$PWD"`.
The checkout guard compares filesystem identity, so `/scratch` and `/gpfs` mount
aliases are accepted while a genuinely different imported checkout is rejected.
Job 138648114 failed before evaluation at the original strict path guard; retry
with a new output directory after pulling the guard fix. No benchmark results
were produced by that failed six-second job.

Replacement job **138650471 completed (0:0)** on October 28 at execution commit
`fb664b0`: both restores passed; 150 validation examples repeated with exactly
identical scores/predictions; 500 examples/task at each checkpoint produced 3,000
main records and paired intervals. Slurm elapsed was **309 seconds** on one V100
with two CPUs allocated (0.08583 V100-hours, about $0.086 at the class's notional
$1/hour). Total CPU time was 300.944 seconds; peak host RSS about 2.85 GiB.
The previous failed attempt adds six allocated GPU seconds to the overall cost.
The current focused CPU suite passes **31 tests**.

Verified outputs are backed up locally in ignored
`rq41_outputs/real-pilot-20261028-fb664b0/`: `REPORT.md`, `noise_table.md`,
`comparison.json`, `allocation.json`, raw bundles and logs. All 12 manifest file
digests were verified after transfer. Reviewed aggregate results are included in
[draft PR #4](https://github.com/vorhersager/gw1b-f2026/pull/4). Keep the raw
benchmark documents/records out of Git. This result remains an exploratory
single-seed 50M pilot; the three-seed 100M milestone is still outstanding.
The adapter's new optional `--task-revisions FILE` accepts task-to-40-character-commit
JSON and passes those actual revisions into dataset loading. Other users' defaults
are unchanged. Historical initial-smoke handoff notes below remain for context.

This implements the logging/statistics part of proposal Gates 1–2. The workspace's
`RQ4_1_Revised_Proposal.pdf` was verified as a PDF and read with pypdf
(seven extracted pages). The primary study is three 100M training seeds;
the unfinished 50M job is an infrastructure check, excluded from research claims.
No cluster evaluation or checkpoint availability is asserted here.

Existing flow: `gw1b run` submits `slurm/run.sbatch`, which invokes
`bin/gw1b-exec python ...`. `evaluate.load_run` reconstructs the JAX model from
`config.yaml` and restores parameters with Orbax. `lm_eval_adapter.GW1BLM`
uses harness `TemplateLM` tokenization and batched teacher-forced likelihoods.
The existing aggregate-only evaluation remains the default. `--rq41` opts into
zero-shot records for **hellaswag, arc_easy, piqa** with **lm-eval 0.4.13**.

## Mac: local verification

Run from the discovered development repository:

```bash
cd '/Users/omarimamverdiyev/Desktop/ML PROJECT 1/gw1b-f2026'
python3 -m venv .venv                 # only needed when the private venv is absent
.venv/bin/python -m pip install -r rq41/requirements-cpu.txt
.venv/bin/python -m pytest tests/test_rq41.py -q

# Optional real harness/JAX/Orbax integration, still entirely synthetic and offline:
.venv/bin/python -m pip install -r rq41/requirements-integration.txt
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_rq41_integration.py -q
bash -n rq41/check_job.sh rq41/eval_job.sh rq41/smoke.sbatch
```

`tests/fixtures/rq41/synthetic_mc.json` contains invented questions and scores.
The statistical tests cover known means (0.25 raw, 0.75 normalized), exact zero
paired effects, shuffled IDs, reproducible bootstrap draws, nonzero paired
effects, invalid inputs, count checks, serialization and CLI output collisions.
The integration test substitutes only dataset downloading: it uses the actual
three pinned YAML task templates/preprocessing, a temporary SentencePiece model,
an untrained tiny JAX model, real Orbax save/restore and repeated real inference.
It checks all nine saved examples and exact repeated candidate scores.
It is **not** a downloaded-benchmark or Pegasus-model integration test.
Missing optional dependencies cause explicit pytest skips. Existing model/training
regressions can be run with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests -q`;
their own additional dependencies may be needed.

Verified locally on 28 October 2026 with the pins below: **46 passed, 1 skipped**
for the full repository suite. The skip is the existing HF export numerical check
(torch/transformers absent); both new integration tests passed. Shell syntax,
Python compilation, `pip check` and `git diff --check` also passed. The dependency
files pin relevant top-level CPU versions, not the complete class CUDA environment.

## Pegasus: inspect the existing training job

Read-only Slurm checks on 28 October 2026 confirmed job **138631889** is
**PENDING**, reason **Priority**, with zero runtime and no estimated start time.
The recorded submission matches all requested smoke-training settings, including
one V100 and the 30-minute limit. Its log and checkpoint directory have not been
created yet; this is consistent with the job not having started.

SSH now works through the user's authenticated control connection at
`~/.ssh/pegasus-rq41` (unlocked key plus Pegasus two-factor authentication).
The Mac commands below reuse that connection; it must still be active.
The latest job status must be checked again before checkpoint-dependent work.

First copy this checkout to a **private** Pegasus directory; do not replace the
shared installation. The following commands create a new private copy, retaining
the local Git commit and uncommitted development changes:

The branch is published on `OmarImamverdiyev/gw1b-f2026`. Use that fork as
`origin` and keep the class repository as `upstream`. For a new private Pegasus
checkout, use the following; the existing `rq41-dev` checkout is already configured:

```bash
# Pegasus; run once after the branch exists on GitHub.
git clone --branch team4/rq41-eval-foundation \
  https://github.com/OmarImamverdiyev/gw1b-f2026.git \
  /scratch/gw1b-class/users/g37693783/rq41-dev
cd /scratch/gw1b-class/users/g37693783/rq41-dev
git remote add upstream https://github.com/vorhersager/gw1b-f2026.git
git pull --ff-only
git status --short
git rev-parse HEAD
```

Future updates use `git pull --ff-only` in that same private checkout. Compare
`git rev-parse HEAD` on the Mac and Pegasus before evaluation. A nonempty working
tree must be reviewed before pulling; do not discard local cluster edits.

The direct-copy alternative below also works before pushing to GitHub:

```bash
# Mac; the timestamp names a fresh remote directory. Keep its printed value.
cd '/Users/omarimamverdiyev/Desktop/ML PROJECT 1/gw1b-f2026'
RQ41_REMOTE="/scratch/gw1b-class/users/g37693783/rq41-dev-$(date +%Y%m%d-%H%M%S)"
echo "$RQ41_REMOTE"
ssh -S "$HOME/.ssh/pegasus-rq41" -o BatchMode=yes G37693783@pegasus.arc.gwu.edu "mkdir '$RQ41_REMOTE'"
rsync -av -e "ssh -S \"$HOME/.ssh/pegasus-rq41\" -o BatchMode=yes" \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='rq41_outputs/' --exclude='runs/' --exclude='checkpoints/' \
  --exclude='datasets/' --exclude='data/' --exclude='lm_eval/' \
  --exclude='gw1b.local.env' --exclude='.env' --exclude='*.key' --exclude='*.pem' \
  ./ "G37693783@pegasus.arc.gwu.edu:$RQ41_REMOTE/"
ssh -S "$HOME/.ssh/pegasus-rq41" -o BatchMode=yes G37693783@pegasus.arc.gwu.edu
```

```bash
# Pegasus login shell: replace the value with the private directory printed above.
RQ41_CHECKOUT=/scratch/gw1b-class/users/g37693783/rq41-dev-YYYYMMDD-HHMMSS
cd "$RQ41_CHECKOUT"
bash rq41/check_job.sh
```

The inspection script runs `squeue -j 138631889`, `sacct`, `scontrol`, tails matching
logs under `/scratch/gw1b-class/users/g37693783/jobs`, and lists checkpoint
directories. Pending/running jobs and absent checkpoints are normal. Do not
duplicate, cancel or resubmit training. If `scontrol` reports a different
`StdOut`, inspect that file with `tail -n 80 /actual/StdOut/path`.
An absent `squeue` row is not proof of success: review `sacct` and logs.

The requested training budget was 20,480 tokens, 4 sequences/step and length 256
(20 optimizer steps, accumulation 4); completed checkpoints may therefore be
near steps 10/20, but **never infer availability from these targets**.
Numeric directories or a “checkpoint saved” message do not prove that asynchronous
Orbax writing completed. `rq41_checkpoint` requires successful parameter restore,
matching metadata and an available tokenizer before evaluation begins.

## Pegasus: manual evaluation submission after reviewing status/logs

The commands below are a handoff, **not executed by this implementation**.
Use Slurm for all Pegasus computation. Review job/log evidence first; the job
also fails closed if no restorable checkpoint exists. It never submits training.

```bash
# Pegasus; retain RQ41_CHECKOUT from above.
source /SEAS/groups/gw1b-class/gw1b-f2026/activate.sh
cd "$RQ41_CHECKOUT"
RQ41_RUN=/scratch/gw1b-class/users/g37693783/runs/team4-rq41-smoke-50m
RQ41_OUT="/scratch/gw1b-class/users/g37693783/rq41_outputs/smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p /scratch/gw1b-class/users/g37693783/jobs
echo "$RQ41_OUT"
sbatch --output=/scratch/gw1b-class/users/g37693783/jobs/rq41-eval-%j.out \
  rq41/smoke.sbatch "$RQ41_CHECKOUT" "$RQ41_RUN" "$RQ41_OUT"
# Optional fourth argument selects a specific verified step; otherwise latest completed.
```

The new evaluation job requests one V100 for 30 minutes. It sets `GW1B_HOME` to
the private checkout, changes to it, and invokes its `bin/gw1b-exec`.
**Setting only PYTHONPATH is insufficient**: that launcher prepends GW1B_HOME.
The inner job asserts `gw1b.__file__` is in the private checkout and prints
the package path, Git HEAD and working-tree status. No shared packages/code
are changed and no packages are installed by the job. If the shared runtime
lacks the pinned harness, stop and arrange a private environment separately.

Inside Slurm the script first inventories Orbax completed steps and restores a
model into a verification process, writing `$RQ41_OUT.verification.json`. It then
passes that **exact** step to a fresh adapter process:

```bash
# Already inside the Slurm runtime (eval_job.sh runs this for you):
python -m gw1b.lm_eval_adapter --run "$RQ41_RUN" --step ACTUAL_VERIFIED_STEP \
  --tasks hellaswag,arc_easy,piqa --num-fewshot 0 --limit 50 --batch-size 2 \
  --rq41 --eval-seed 0 --out "$RQ41_OUT"
```

Each task must yield `min(limit, eligible split size)` records, agreeing with
harness effective counts and aggregate means. The smoke script additionally
requires **exactly 50 per task (150 total)**. Failed checks exit nonzero.
Every output directory/file must be new; partial failed bundles must be inspected
and retries given new paths. A successful `manifest.json` marks completed
serialization/count/aggregate validation. The first real evaluation still needs
dataset download/cache access, enough GPU memory and a valid training checkpoint.

## Output contract

| File | Contents |
| --- | --- |
| `raw_harness.json` | Full returned harness result, including configs, samples, counts, versions and environment |
| `samples.jsonl` | `{task, sample}` wrappers around original harness per-example dictionaries |
| `records.jsonl` | Self-contained `gw1b.rq41.v1` analysis rows described below |
| `manifest.json` | Common metadata, expected counts, task definitions and SHA-256 digests of the three data files |
| `summary.json` | Existing-style aggregate scores and evaluation duration |
| `analysis.json` | Individual question-bootstrap scores produced by the smoke script |

JSON conversion preserves numeric values; numpy arrays/scalars and tuples become
JSON arrays/scalars. Callable objects retained in nested harness configs are encoded
as `{"__callable__": "module.qualname", "source": ...}` (source is null if unavailable),
never process-specific memory addresses. Missing measured fields stay null or are
absent; missing required metric values fail validation rather than becoming zero.

Each record contains:

- `model`: run directory/name, **actual restored step**, checkpoint tokens when
  present, initialization/data-order seeds and their source, training config hash.
- `task`, `split`, string `example_id` from harness `doc_id`, full processed `doc`,
  SHA-256 `content_hash` of canonical UTF-8 JSON. Identity is `(task, split, ID)`
  within the saved task definition/fingerprint, not a global dataset identifier.
- `task_definition`: full resolved config, task version, split, expected count,
  deterministic first-N selection, processed dataset fingerprint/version and
  requested revision when supplied. Standard task YAMLs do **not** pin a Hub
  revision: that field is null, not an invented commit. Freeze revisions before
  research evaluation; fingerprints plus per-example hashes guard this smoke comparison.
- `arguments`: complete rendered context/continuation requests; `harness_hashes`
  preserve the harness's own document/prompt/target hashes separately.
- `gold`, predictions for raw and normalized scoring, candidate text/continuation,
  summed natural-log likelihood, character denominator and normalized score.
- `metrics`/`metric_specs`: available per-example indicators with the response
  filter in the key (e.g. `acc,none`), units and improvement direction.
- `protocol`: prompt format, no chat/system prompt, zero few-shot, all four
  harness evaluation seeds, tokenizer full-file hash/vocabulary/BOS policy,
  context length/truncation, actual model precision/attention, batch size,
  dependency versions and scoring/harness source hashes.
- `provenance`: imported package path, Git commit/status, source hashes (also
  identify uncommitted code), Python/platform/devices, tokenizer path and Slurm job.
  `synthetic`, `sampling_unit`, `group_id` label data origin and supported units.

For all three inspected `ConfigurableTask` implementations, `acc` is the binary
correctness of the first argmax of **summed continuation log-likelihood**.
`acc_norm` is correctness after dividing each candidate score by
**Python `len(task.doc_to_choice(doc)[i])`**, i.e. Unicode characters of the
preprocessed answer text, **excluding the target delimiter**. It is neither a
token nor UTF-8-byte denominator. Ties use the first candidate. The serializer
uses the actual task choices/filtered responses, checks every observed indicator
and verifies its mean against the harness aggregate. Other filters, mutual-info,
multiple-input/target formulations and metrics are rejected in this milestone.

## CPU analysis and retrieving results

```bash
# Pegasus login: inspect small text outputs after the evaluation job finishes.
cat "$RQ41_OUT/summary.json"
cat "$RQ41_OUT/manifest.json"

# Mac: substitute the actual printed output directory; copy outside scratch for backup.
cd '/Users/omarimamverdiyev/Desktop/ML PROJECT 1/gw1b-f2026'
mkdir -p rq41_outputs
rsync -av -e "ssh -S \"$HOME/.ssh/pegasus-rq41\" -o BatchMode=yes" \
  G37693783@pegasus.arc.gwu.edu:/scratch/gw1b-class/users/g37693783/rq41_outputs/ rq41_outputs/
.venv/bin/python -m gw1b.rq41_analysis rq41_outputs/ACTUAL_A/records.jsonl --seed 0
# For two saved checkpoint evaluations A (earlier), B (later):
.venv/bin/python -m gw1b.rq41_analysis rq41_outputs/ACTUAL_A/records.jsonl \
  --compare rq41_outputs/ACTUAL_B/records.jsonl --replicates 10000 --seed 0 \
  --out rq41_outputs/comparison.json
```

Analysis uses 10,000 percentile bootstrap replicates by default (configurable),
sampling independent questions with replacement and computing their mean.
Pairs are sorted by identity, then the **same** sampled indices select both
checkpoints; equivalently we bootstrap per-question `B_i - A_i`. Reports contain
task/split/metric counts, fraction units, 95% intervals and direction: positive
B−A is improvement (multiply fractions by 100 for percentage points).
NumPy version/seed are recorded; identical versions/seeds reproduce draws.

Inputs reject duplicate IDs, missing counterparts or metric values, non-finite
values, mixed checkpoints/runs/seeds within a file, inconsistent definitions,
altered document hashes, different rendered requests/gold answers, tokenizer,
precision/batch/dependency/scoring settings, synthetic/real mixing and different
training seeds. Run each training seed separately; these intervals describe
**question-sampling uncertainty conditional on those checkpoints**, not
training-seed variation. Do not concatenate seed-by-question rows. Later report
per-seed effects plus seed-level mean/SD/range separately.

Intervals are **individual, unadjusted**, not simultaneous. No permutation tests,
Holm correction, corrected significance or equivalence claims are implemented.
Only the three independent answer-selection tasks are accepted; grouped or
shared-passage data require cluster resampling and are rejected explicitly.
BPB document resampling/recomputed ratios and MMLU subject-stratified macro/micro
analysis remain future work, as do LAMBADA, custom prompts/normalizations, full
sweeps, contamination scans and the class-written question set.

Next: verify the real training checkpoint and complete the 150-example Pegasus
smoke; check numerical repeatability on the V100; inventory matched 100M seed
checkpoints; then freeze revisions, identities, primary metrics, contrasts and
families before research evaluation. Preserve outputs outside scratch. Git ignores
generated evaluation outputs, datasets, checkpoints and common credential files;
only the small explicitly synthetic fixture belongs in source control.
