#!/bin/bash
#----------------------------------------------------------------------------
# hdc_dimension_memory.slurm -- measures memory usage vs. hypervector
# dimension D on TACC: theoretical item-memory/class-HV byte counts plus each
# D's actual process peak RSS (resource.getrusage, one fresh subprocess per D
# so measurements don't inherit a previous D's high-water mark).
#
# CPU-only, single-threaded per D (one subprocess at a time, sequential --
# peak-RSS attribution would be muddied by concurrent runs sharing a node's
# memory), so this runs on TACC's development partition (2h wall cap).
#
# Submit (from the repo root):   sbatch slurm/hdc_dimension_memory.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_dimension_memory.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_dim_mem
#SBATCH -o logs/hdc_dimension_memory.%j.out
#SBATCH -e logs/hdc_dimension_memory.%j.err
#SBATCH -p development               # queue: no other jobs waiting here currently
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00                  # development partition's wall cap
#SBATCH -A OTH21017
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_dimension_memory.slurm` from there

# --- Environment ------------------------------------------------------------
source /scratch/09424/jiwoooop/snn-env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Run ----------------------------------------------------------------
OUTDIR="results/results_dimension_memory_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "Running dimension-vs-memory sweep -> $OUTDIR/"
python3 scripts/run_dimension_memory_experiment.py --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results + plot in: $SLURM_SUBMIT_DIR/$OUTDIR/"
