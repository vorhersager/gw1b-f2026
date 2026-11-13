# Team 2: RQ 2.1 evaluation scripts

These scripts measure each trained 100M model three ways: held-out loss, scores on downstream tasks, and decode latency. Together they let us compare depth against width when inference cost counts. The reasoning behind the design is in [team2_eval_plan.md](team2_eval_plan.md).

## How to use the scripts

### Before you start

- Run everything on the cluster, from the scripts folder. The script imports `lm_eval`, which is in the cluster environment but not in a normal laptop setup.
- If `gw1b` is not found, run `source activate.sh` from the repo root first.
- Everyone writes to the same results folder, so agree who runs each stage. If two people run `loss` or `downstream` on the same run at the same time, one person's results will overwrite the other's. `--force` also replaces existing results.

| What | Where |
|---|---|
| Scripts | `/scratch/gw1b-class/teams/team2/eval/scripts` |
| Results | `/scratch/gw1b-class/teams/team2/eval/eval_output` |
| Smoke test results | `/scratch/gw1b-class/teams/team2/eval/smoke_output` |
| Trained runs | `/scratch/gw1b-class/teams/team2/runs` |

The scripts can live anywhere, as long as `rq21_eval.py` and the `rq21/` folder stay side by side.

The examples below use these values. Set them again in each new shell, because a new shell does not keep them:

```bash
cd /scratch/gw1b-class/teams/team2/eval/scripts
RUNS=/scratch/gw1b-class/teams/team2/runs
OUT=/scratch/gw1b-class/teams/team2/eval/eval_output
```

### Step 1: Check the runs

This needs no GPU, so run it on the login node:

```bash
gw1b python rq21_eval.py validate --runs-root $RUNS
```

It should end with `10 runs, 0 problem(s)`. Do not start the full evaluation until it does.

If a run hasn't finished, the output looks like this:

```text
10 runs, 1 problem(s)
  - rq_2_1_100m_r3_t60: no summary.json
```

### Step 2: Smoke test

Try every stage on the two most extreme shapes before using more GPU time. `r3` has 56 layers and `r5` has 3 layers. Keep these results in their own folder: the full evaluation skips any run that already has a result, so a smoke result in `eval_output` would be kept in place of the full one.

Submit one batch job per stage from the login node, in the scripts folder.

```bash
SMOKE=/scratch/gw1b-class/teams/team2/eval/smoke_output
TWO="--run rq_2_1_100m_r3_t20 --run rq_2_1_100m_r5_t20"

gw1b run --gpus 1 --gpu-type a100 --time 1:00:00 --name rq21-loss       rq21_eval.py loss       --runs-root $RUNS --out $SMOKE $TWO
gw1b run --gpus 1 --gpu-type a100 --time 1:00:00 --name rq21-downstream rq21_eval.py downstream --runs-root $RUNS --out $SMOKE $TWO --limit 50
gw1b run --gpus 1 --gpu-type a100 --time 1:00:00 --name rq21-latency    rq21_eval.py latency    --runs-root $RUNS --out $SMOKE $TWO
```

| Part | Meaning |
|---|---|
| `gw1b run` | Submits a batch job to the queue |
| `--gpus 1 --gpu-type a100` | Requests one A100 |
| `--time 1:00:00` | Stops the job if it passes 1 hour |
| `--name` | Names the job, which also sets its log file name |
| `rq21_eval.py loss` | Runs the script, with the stage to run after it |
| `--runs-root $RUNS` | Where the trained runs are |
| `--out $SMOKE` | Where to save results |
| `$TWO` | Only uses `r3_t20` and `r5_t20` |
| `--limit 50` | Cuts downstream to 50 examples per task |

Jobs have finished once `gw1b status` no longer lists them. Then:

1. Check each job's log. A job that worked ends with `finished with exit code 0`.

   ```bash
   cd $GW1B_SCRATCH/users/$USER/jobs/
   tail -n 20 rq21-loss-*.out
   tail -n 20 rq21-downstream-*.out
   tail -n 20 rq21-latency-*.out
   ```

2. Check that both run folders have the same four files: `loss.json`, `downstream.json`, `downstream.jsonl` and `latency_<job id>.json`.

   ```bash
   ls /scratch/gw1b-class/teams/team2/eval/smoke_output/*
   ```

3. Check that the numbers make sense:

   - The loss is plausible and close to `val_loss` in the run's `summary.json`.
   - Downstream scores are above chance, which is about 25% on `hellaswag`. Fifty examples is noisy, so treat this as a rough check.
   - Latency accepts the A100, and compiled-only is faster than as-shipped.

4. Note how long each job took, and use that to set the time limits for the full evaluation.

### Step 3: Full evaluation

Submit these from the scripts folder. The time limits are based on estimates, so adjust them using what the smoke test showed.

```bash
gw1b run --gpus 1 --gpu-type a100 --time 2:00:00  rq21_eval.py loss       --runs-root $RUNS --out $OUT
gw1b run --gpus 1 --gpu-type a100 --time 14:00:00 rq21_eval.py downstream --runs-root $RUNS --out $OUT

# Twice on purpose: comparing the two jobs shows how noisy the timings are.
gw1b run --gpus 1 --gpu-type a100 --time 2:00:00  rq21_eval.py latency    --runs-root $RUNS --out $OUT
gw1b run --gpus 1 --gpu-type a100 --time 2:00:00  rq21_eval.py latency    --runs-root $RUNS --out $OUT
```

Use `gw1b status` to follow the jobs. Job output goes to `$GW1B_SCRATCH/users/$USER/jobs/`.

### Options

These options are for `rq21_eval.py`.

| Option | Commands | What it does |
|---|---|---|
| `--runs-root` | All | Folder containing the run folders. Required. |
| `--prefix` | All | Only use runs whose names start with this. Default: `rq_2_1_100m_`. |
| `--nominal-params` | `validate` | Expected parameter count. Default: 100 million. |
| `--out` | `loss`, `downstream`, `latency` | Folder for results. Required. |
| `--run` | `loss`, `downstream`, `latency` | Only this run. Repeat it to select several. |
| `--force` | `loss`, `downstream`, `latency` | Redo runs that already have a result. |
| `--tasks` | `downstream` | Comma-separated `lm_eval` tasks. Default: `hellaswag,arc_easy,piqa`. |
| `--limit` | `downstream` | Examples per task. For smoke tests only. |

If a job is interrupted, submit it again with the same command. Runs that already have a result are skipped. A latency job is different: it saves only at the very end, under a new job ID, so an interrupted one starts again from the beginning.

### Output

The scripts never change the run folders. Results go to `<out>/<run name>/`:

| File | Contents |
|---|---|
| `loss.json` | Mean loss, perplexity, the loss of each batch, and how many windows were scored |
| `downstream.json` | Mean score per task and metric, and the number of examples |
| `downstream.jsonl` | One line per example: task, example ID, correct answer and metrics |
| `latency_<job id>.json` | Time of every decode step, in ms, by version, cache length and repeat |

Each `.json` file also records the device and the checkpoint step. `downstream.jsonl` holds only the per-example records.

## Files and functions

### `rq21_eval.py`

The command line, and the only script you run directly.

| Function | What it does |
|---|---|
| `main` | Reads the command-line options, runs the chosen command and returns its exit code. 0 means success. |

### `rq21/runs.py`

Finds and checks run folders, and saves results. The other stages all use it.

| Function | What it does |
|---|---|
| `find_runs` | Lists the run folders under `--runs-root` that match `--prefix` and contain `config.yaml`. |
| `select_runs` | Applies `--run` to that list. If nothing matches or a name is unknown, it prints why and returns nothing. |
| `read_run` | Loads one run's config and summary, and calculates its parameter count, embedding share, aspect ratio and tokens/param. |
| `validate` | Runs the `validate` command: prints the table of runs, lists any problems, and returns 1 if there are any. |
| `_comparable_settings` | Helper for `validate`. Lists a run's settings, leaving out those allowed to differ between runs. |
| `save_json` | Writes a results file with the device and checkpoint step added. Raises an error rather than saving NaN. |

`validate` reports a problem if:

- A shape and budget combination is missing or repeated, or there are not 5 shapes.
- Tokens/param is more than 5% from 20 or 60.
- The parameter count is more than 2% from `--nominal-params`.
- A run has no `summary.json`, or is not marked as finished.
- A validation loss is missing or not finite.
- Any setting other than shape and token budget differs between runs, ignoring settings that cannot change the trained model.

It does not check that checkpoints load. The `loss` stage is the first to test that.

### `rq21/loss.py`

The `loss` command. It scores every run on the same validation windows, so losses can be compared across shapes. The validation loss saved during training may not be comparable.

| Function | What it does |
|---|---|
| `select_windows` | Picks the windows to score: all of them, or up to 8,192 evenly spaced, trimmed to full batches. |
| `score_windows` | Returns the mean loss of each batch. The analysis needs per-batch values for its bootstrap. Raises an error on a non-finite loss. |
| `cmd_loss` | Loads each selected run's final checkpoint, scores it, and writes `loss.json`. |

### `rq21/downstream.py`

The `downstream` command. It scores each model on multiple-choice benchmarks using `lm_eval`, with no worked examples given to the model.

| Function | What it does |
|---|---|
| `extract_records` | Converts one task's `lm_eval` output into per-example records. Raises an error on a repeated example or a missing or non-finite score. |
| `cmd_downstream` | Evaluates each selected run, checks that our per-example means match `lm_eval`'s totals, and writes `downstream.jsonl` and `downstream.json`. |

### `rq21/latency.py`

The `latency` command. It times one decode step at a time, using greedy decoding at batch size 1. It stops with an error unless the job has exactly one A100, and it must run inside a Slurm job.

Each run is timed in two versions:

- **As shipped** uses `gw1b.generate._decode_step`, the code the project actually runs.
- **Compiled only** does the same calculation through a plain `jax.jit`. `nnx.jit` adds overhead on each call that grows with depth, and this version leaves that out. Comparing the two shows whether depth is slow because of the model or because of that overhead.

| Function | What it does |
|---|---|
| `make_compiled_step` | Builds the compiled-only version of the decode step. |
| `time_decode` | Fills the cache with a fixed 64-token prompt, runs 32 warm-up steps and discards them, then returns the times of the next 256 steps in ms. |
| `cmd_latency` | Times both versions at cache lengths 512 and 2048, 5 times each, in a shuffled order that changes every repeat. Writes `latency_<job id>.json` for each run. |

## Common errors

| Error | What it means |
|---|---|
| `can't open file '.../rq21_eval.py'` | You are in the wrong folder. `cd` to the scripts folder first. |
| `No module named 'lm_eval'` | You are outside the cluster environment. Use `gw1b python` or a `gw1b shell` session. |
| `No module named 'rq21'` | The `rq21/` folder is missing from the scripts folder or was not copied fully. |
| `No runs starting with ...` or `No matching runs` | `--runs-root`, `--prefix` or a `--run` name is wrong. |
| `latency needs exactly one A100` | Request `--gpus 1 --gpu-type a100`. |
| `KeyError: 'SLURM_JOB_ID'` | `latency` was run outside a Slurm job, for example on the login node. |
