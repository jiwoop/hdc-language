#!/bin/bash

#SBATCH -J snn_multibeta
#SBATCH -o logs/snn_multibeta.%j.out
#SBATCH -e logs/snn_multibeta.%j.err
#SBATCH -p gpu-a100-small            # queue
#SBATCH -N 1                         # nodes
#SBATCH -n 1                         # tasks
#SBATCH -t 04:00:00                  # wall time hh:mm:ss
#SBATCH -A OTH21017                  # Allocation ID
#SBATCH --mail-type=ALL              # email on BEGIN/END/FAIL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_gesture_snn_temporal_multibeta.slurm` from there

# --- Environment ------------------------------------------------------------
# Same prebuilt venv on $SCRATCH as hdc_gesture_snn.slurm (torch + snntorch).
source /scratch/09424/jiwoooop/snn-env/bin/activate

# --- Run ----------------------------------------------------------------
OUTDIR="results/results_snn_multibeta_${SLURM_JOB_ID}"

echo "Running multi-beta SNN comparison -> $OUTDIR/"
python3 scripts/run_gesture_snn_experiments.py --device cuda --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results in: $SLURM_SUBMIT_DIR/$OUTDIR/"
