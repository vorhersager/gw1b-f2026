# RQ4.1 exploratory protocol — frozen 28 October 2026

Purpose: produce the first real-checkpoint uncertainty table by October 29.
This is an exploratory **50M, single-training-seed** study. It does not fulfill
the required three-seed 100M ablation and does not establish training-seed variation.

Inputs are frozen in `pilot_manifest.json` and `pilot_revisions.json` before
benchmark scores are observed. Use the existing class run `team2-gqa-vs-mha`,
steps 1000 and 2000 (262,144,000 and 524,288,000 training tokens), initialization
and data-order seed 0, the saved class tokenizer, float32, and context length 2048.
Checkpoint/config/tokenizer metadata hashes are checked before any model loads.
Successful Orbax parameter restoration is mandatory. Original class files are read-only.

Tasks: HellaSwag validation, ARC-Easy test, PIQA validation. Dataset Hub commits
are fixed in `pilot_revisions.json`. Use the pinned lm-eval 0.4.13 standard
continuation templates, zero-shot, no chat/system prompt, batch size 2, and all
four evaluation seeds 0. Keep preprocessing, option order, separators, and truncation
unchanged; the bundle records full rendered requests and processed-content hashes.

First evaluate 50 examples/task at step 2000, then repeat at identical settings.
Example sets, predictions and metrics must match exactly; absolute candidate
log-likelihood differences must be at most 1e-5. Choose the main sample count
from [500,100,50] using runtime only: reserve 120 seconds and require
`2 * max(validation_seconds, repeat_seconds) * N/50 * 1.5` to fit the remaining
27-minute driver budget. Save that decision before either main evaluation.
Evaluate the same first N eligible examples at both checkpoints. No score-based
sample selection or favorable-variant selection is permitted.

Report both available accuracy indicators: raw `acc` and `acc_norm` using summed
candidate log-likelihood divided by Python character length of task answer text,
excluding the target delimiter. Aggregate each as a mean. Pair by task/split/ID;
require identical documents, rendered requests, task definitions and scoring settings.

Resample independent question pairs with replacement: 10,000 replicates, bootstrap
seed 0, percentile endpoints 2.5/97.5. Display accuracy in percent and later-minus-earlier
changes in percentage points. Positive differences indicate improvement. Intervals
are individual, unadjusted, conditional on this training trajectory. No corrected
significance, equivalence, independent seed replication or benchmark-usefulness claim.

Resource ceiling: one V100, two CPUs, 16 GB host RAM, 30 minutes (at most 0.5 V100-hours;
notional maximum $0.50 at the class calculator's $1/V100-hour, not a Pegasus invoice).
All checkpoint/inference/analysis computation runs inside Slurm. CPU library thread
counts are restricted. Record job/commit/package path, dependency versions, elapsed
time, actual sample counts, precision, and actual allocated-GPU wall-time cost.

Outputs: raw harness evidence, validated records, repeatability report, selected-count
record, paired JSON analysis, Markdown/CSV noise tables and cost JSON. Keep these in
the user's ignored `rq41_outputs` directories; back them up to the Mac. Share only
reviewed aggregate results and the protocol, not checkpoints or benchmark documents.

Deferred: three independently trained 100M seeds (owner/budget unconfirmed), full
split evaluation, LAMBADA, MMLU subject strata, BPB documents, clustered resampling,
permutation/Holm procedures, prompt variants, contamination scans and class questions.
