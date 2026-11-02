#!/bin/bash
# Example Slurm job for ONE shard of stage 5 (adapt partition/account/modules).
# Submit stages 0-4 once (STAGES="0 1 2 3 4"), then an array for stage 5:
#   sbatch --array=0-3 --export=ALL,NUM_SHARDS=4 slurm_example.sbatch
# then stages 6-8 once (STAGES="6 7 8").
#SBATCH --job-name=act70b
#SBATCH --gres=gpu:4                 # 4 x 80 GB for fp16/bf16 70B
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=act70b_%A_%a.out

set -euo pipefail
cd "${REPO_ROOT:?set REPO_ROOT to the repository root}"
# source your environment here, e.g.:  source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf}"
export STAGES="${STAGES:-5}"
export SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
bash examples/act_three/scale70b/run_all.sh
