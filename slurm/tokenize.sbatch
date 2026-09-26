#!/bin/bash
# Tokenize FineWeb-Edu into training shards on a CPU node (run after `prepare_data download` and the
# tokenizer training). Usage:
#   sbatch --partition=$GW1B_PART_CPU slurm/tokenize.sbatch --sample 10BT \
#       --tokenizer $GW1B_GROUP/tokenizer/gw1b-32k.model --out $GW1B_SCRATCH/data/fineweb-edu-10B
#SBATCH --job-name=gw1b-tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=12:00:00
set -euo pipefail
source "$GW1B_HOME/gw1b.env"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1     # compute nodes: read the parquet files already in $HF_HOME
"$GW1B_HOME/bin/gw1b-exec" python -m gw1b.prepare_data tokenize --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
