# Class update draft — not sent

Team 4 / RQ4.1 update for October 29:

Our evaluation logging and paired-bootstrap foundation is implemented and available
for review in draft PR https://github.com/vorhersager/gw1b-f2026/pull/4.
The exploratory protocol is `rq41/PROTOCOL_PILOT.md`; checkpoint and dataset-revision
manifests are included in the branch. It uses standard zero-shot HellaSwag, ARC-Easy
and PIQA, identical examples across checkpoints, raw and character-normalized
accuracy, and 10,000-replicate individual, unadjusted paired bootstrap intervals.

We completed a real 50M, single-seed comparison of existing class checkpoints at
262,144,000 and 524,288,000 training tokens. Slurm job 138650471 finished successfully
(exit 0:0) in 5m09s on one V100 with two CPUs allocated. Both restores passed;
the 150-example validation repeated with exactly identical scores/predictions.
The predeclared timing rule selected 500 questions per task, producing 3,000 main
model-question evaluations and the preliminary paired noise table reported in PR #4.
All 12 bundle file checksums were verified after backing up the outputs to the Mac.
The earlier evaluation failed at a checkout-path guard; that fix passed 31 local tests.
The previously pending tiny training job was canceled without running.
This pilot does not complete the three-seed 100M milestone. Intervals remain
individual and unadjusted; results do not establish training-seed robustness.

## Seed ownership/budget request — draft

Han / Team 3: are three matched 100M training seeds already available? Please provide
run/checkpoint paths, actual token counts, initialization/data-order seeds and the
common training horizon. If unavailable, please confirm who owns the training and
which team's GPU budget covers it. We should avoid duplicate training and should
not call the three-seed milestone complete until these runs and results exist.

## Question collection coordination — draft

Yusra: please confirm the class-question form link, whether collection is open, and
who owns review/storage. If there is no form yet, the proposed fields are: question,
answer choices (when applicable), gold answer, explanation/reference, subject,
authorship date and optional contributor identifier. Collect original, verifiable
questions with no private information; keep development and final pools separate.
The research target is 300–500 reviewed questions. Authorship after the corpus cutoff
does not by itself establish that the underlying content is uncontaminated.

## Milestone status wording

- Logging/statistics code and local tests: completed; draft PR open.
- Pegasus guide: supplied and reviewed.
- Preliminary real-model noise table: completed and reported in PR #4; exploratory 50M, one seed.
- Protocol announcement/question form: drafts prepared; distribution/open status unverified.
- Three-seed 100M study: owner, budget and checkpoint availability still need confirmation.
