#!/bin/bash
#----------------------------------------------------------------------------
# hdc_ngram_low_n.slurm -- extends hdc_train.py's N-gram sweep (N=3,4,5) down
# to N=1,2, to see whether N=3's advantage over N=4,5 continues or reverses
# below N=3. Runs scripts/run_ngram_low_n_experiment.py on TACC's development
# partition (128-core CPU nodes, 2h wall cap) -- same CPU-bound
# multiprocessing.Pool sweep pattern as hdc_train.py, just a wider N range.
#
# Submit (from the repo root):   sbatch slurm/hdc_ngram_low_n.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_ngram_low_n.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_ngram_low_n
#SBATCH -o logs/hdc_ngram_low_n.%j.out
#SBATCH -e logs/hdc_ngram_low_n.%j.err
#SBATCH -p development               # queue: no other jobs waiting here currently
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00                  # development partition's wall cap
#SBATCH -A OTH21017
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Node has $(nproc) cores"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_ngram_low_n.slurm` from there

# --- Environment ------------------------------------------------------------
source /scratch/09424/jiwoooop/snn-env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Run ----------------------------------------------------------------
WORKERS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
OUTDIR="results/results_ngram_low_n_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "Running N=1..5 sweep with $WORKERS workers -> $OUTDIR/"
python3 scripts/run_ngram_low_n_experiment.py --workers "$WORKERS" --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results + plots in: $SLURM_SUBMIT_DIR/$OUTDIR/"
