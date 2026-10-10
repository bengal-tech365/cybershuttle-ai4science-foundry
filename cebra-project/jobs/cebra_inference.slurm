#!/bin/bash
#SBATCH --job-name=cebra-hippocampus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cebra-%j.out
#SBATCH --error=logs/cebra-%j.err

# TODO: Confirm --gres=gpu:1 for the course allocation.
# TODO: Add the confirmed account, partition and/or QoS directives above.
# PENDING: 4 CPUs, 16G RAM and 30 minutes are provisional, not measured requirements.
# TODO: Add any required module load commands before environment activation.

set -euo pipefail
trap 'status=$?; echo "Job exit status: ${status}; finished: $(date -u +%FT%TZ)"; exit "$status"' EXIT

: "${SLURM_JOB_ID:?Submit with sbatch or run inside a SLURM allocation.}"
: "${SLURM_SUBMIT_DIR:?Submit from the project root.}"
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f run_inference.py || ! -d logs ]]; then
    echo "Submit from the project root after creating logs/." >&2
    exit 1
fi

: "${CEBRA_ENV:?Set CEBRA_ENV to the path of your CEBRA virtual environment before submitting.}"
source "$CEBRA_ENV/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date -u +%FT%TZ)"
python --version
python -m pip check

# Each job keeps its own checkpoint, embeddings and execution evidence.
run_dir="results/job-${SLURM_JOB_ID}"
python -u run_inference.py --seed 42 --data-dir data --output-dir "$run_dir"
python -u validation/validate_results.py --results-dir "$run_dir"
