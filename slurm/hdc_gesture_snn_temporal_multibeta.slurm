#!/bin/bash
#----------------------------------------------------------------------------
# hdc_gesture_snn_temporal_multibeta.slurm -- tests two fixes for the SNN
# gesture classifier's lack of temporal-order sensitivity (see
# plot_snn_error_analysis.py's confusion-matrix/margin analysis: a flat
# feedforward SNN with rate decoding has only a single fixed-rate membrane
# decay as "memory," and rate decoding -- summing spikes over all 315 steps --
# discards spike order entirely), on TACC's gpu-a100-small partition.
#
# Runs run_gesture_snn_temporal_multibeta.py, which trains+evaluates 4 configs
# (all delta-modulation encoded, using the best-known Synaptic hyperparameters
# from run_gesture_snn_hparam_sweep.py's results.json):
#   1. baseline:            SNNGestureClassifierSynaptic, rate decode (known best, ~46-48%)
#   2. temporal:             SNNGestureClassifierSynaptic, temporal (first-spike-time) decode
#   3. multibeta:            SNNGestureClassifierMultiBeta (fast+slow beta hidden layers), rate decode
#   4. multibeta_temporal:   SNNGestureClassifierMultiBeta, temporal decode
#
# Produces an accuracy-comparison bar chart plus a per-config confusion matrix
# (see src/snn_encoder.py's evaluate_snn(decode=...) and
# SNNGestureClassifierMultiBeta for the underlying implementation).
#
# Same GPU rationale as hdc_gesture_snn.slurm (matmul-bound training loop).
#
# Submit (from the repo root):   sbatch slurm/hdc_gesture_snn_temporal_multibeta.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_gesture_snn_temporal_multibeta.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_gesture_snn_tmb
#SBATCH -o logs/hdc_gesture_snn_temporal_multibeta.%j.out
#SBATCH -e logs/hdc_gesture_snn_temporal_multibeta.%j.err
#SBATCH -p gpu-a100-small            # queue (see hdc_gesture_snn.slurm for why this partition)
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
OUTDIR="results_gesture_snn_temporal_multibeta_${SLURM_JOB_ID}"

echo "Running temporal-decode + multi-beta comparison -> $OUTDIR/"
python3 scripts/run_gesture_snn_temporal_multibeta.py --device cuda --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results in: $SLURM_SUBMIT_DIR/$OUTDIR/"
