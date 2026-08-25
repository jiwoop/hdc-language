#!/bin/bash
#----------------------------------------------------------------------------
# hdc_gesture_label_swap.slurm -- checks whether the SNN gesture classifier's
# lopsided confusion matrices (results_gesture_snn_temporal_multibeta_3341837/
# confusion_matrix_baseline.png predicting mostly "3", confusion_matrix_
# multibeta.png predicting mostly "1") are a code/slot bias or a property of
# the gesture data itself, by swapping labels 3<->5 (baseline) and 1<->6
# (multibeta) in both train and test splits, retraining from scratch, and
# comparing confusion matrices before/after.
#
# GPU-bound (matmul-per-timestep training loop, same as hdc_gesture_snn.slurm/
# snn_multibeta.slurm), routed to gpu-a100-dev (2h wall cap, no other jobs
# waiting there currently) rather than gpu-a100-small.
#
# Runs run_gesture_label_swap_experiment.py, which trains+evaluates 4 runs
# (2 configs x {original, swapped} labels):
#   1. baseline_original:  SNNGestureClassifierSynaptic, rate decode, labels as-is
#   2. baseline_swapped:   same model, labels 3<->5 swapped in train+test
#   3. multibeta_original: SNNGestureClassifierMultiBeta, rate decode, labels as-is
#   4. multibeta_swapped:  same model, labels 1<->6 swapped in train+test
#
# Submit (from the repo root):   sbatch slurm/hdc_gesture_label_swap.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_gesture_label_swap.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_gesture_swap
#SBATCH -o logs/hdc_gesture_label_swap.%j.out
#SBATCH -e logs/hdc_gesture_label_swap.%j.err
#SBATCH -p gpu-a100-dev               # queue: no other jobs waiting here currently
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00                   # gpu-a100-dev's wall cap
#SBATCH -A OTH21017
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_gesture_label_swap.slurm` from there

# --- Environment ------------------------------------------------------------
# Same prebuilt venv on $SCRATCH as hdc_gesture_snn.slurm / snn_multibeta.slurm
# (torch + snntorch).
source /scratch/09424/jiwoooop/snn-env/bin/activate

# --- Run ----------------------------------------------------------------
OUTDIR="results/results_gesture_label_swap_${SLURM_JOB_ID}"

echo "Running label-swap confusion-matrix check -> $OUTDIR/"
python3 scripts/run_gesture_label_swap_experiment.py --device cuda --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results in: $SLURM_SUBMIT_DIR/$OUTDIR/"
