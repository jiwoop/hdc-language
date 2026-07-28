#!/bin/bash
#----------------------------------------------------------------------------
# hdc_gesture_snn_comparison.slurm -- SNN encoding/decoding/neuron-model
# comparison for UWaveGestureLibrary gesture classification, on TACC's
# gpu-a100-small partition.
#
# Runs the full pipeline in one job, sequentially, reusing the same GPU
# allocation (see hdc_gesture_snn.slurm's rationale for why this workload
# needs a GPU, unlike the CPU-only HDC sweeps):
#   1. run_gesture_snn_hparam_sweep.py   -- beta (Leaky) / alpha+beta (Synaptic)
#      sweep, using Delta modulation + Rate decoding (the encoding shared by
#      the two neuron-order methods compared in step 2).
#   2. run_gesture_snn_method_comparison.py -- trains+evaluates the 3 methods
#      (Rate+Rate+Leaky, Delta+Rate+Leaky, Delta+Rate+Synaptic) using step 1's
#      winning hyperparameters.
#   3. visualize_gesture_snn_spikes.py   -- spike-train visualizations for the
#      first training sample (input rasters + single-neuron response plots).
#      Cheap (single-neuron forward passes), so runs on CPU even inside this
#      GPU allocation -- no need for a separate job.
#   4. compare_gesture_methods.py        -- final comparison plot against the
#      existing committed HDC sweep (results/results_gesture_3321861).
#
# No wall-clock comparison against the CPU-only HDC methods is made anywhere
# in this project -- GPU vs CPU time isn't a fair comparison. Only accuracy
# (hardware-independent) is compared.
#
# Submit (from the repo root):   sbatch slurm/hdc_gesture_snn_comparison.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_gesture_snn_comparison.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_gesture_snn_comparison
#SBATCH -o logs/hdc_gesture_snn_comparison.%j.out
#SBATCH -e logs/hdc_gesture_snn_comparison.%j.err
#SBATCH -p gpu-a100-small            # queue (see hdc_gesture_snn.slurm for why this partition)
#SBATCH -N 1                         # nodes
#SBATCH -n 1                         # tasks
#SBATCH -t 04:00:00                  # wall time hh:mm:ss
#SBATCH -A OTH21017                  # Allocation ID
#SBATCH --mail-type=ALL              # email on BEGIN/END/FAIL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_gesture_snn_comparison.slurm` from there

# --- Environment ------------------------------------------------------------
# Same prebuilt venv on $SCRATCH as hdc_gesture_snn.slurm (torch + snntorch;
# see that script's comment for why $HOME's --user site-packages won't do).
source /scratch/09424/jiwoooop/snn-env/bin/activate

# --- Run ----------------------------------------------------------------
OUTDIR="results_gesture_snn_comparison_${SLURM_JOB_ID}"

echo "1/4: hyperparameter sweep -> $OUTDIR/hparam_sweep/"
python3 scripts/run_gesture_snn_hparam_sweep.py --device cuda --outdir "$OUTDIR/hparam_sweep"

echo "2/4: method comparison -> $OUTDIR/methods/"
python3 scripts/run_gesture_snn_method_comparison.py --device cuda \
    --hparam-dir "$OUTDIR/hparam_sweep" --outdir "$OUTDIR/methods"

echo "3/4: spike-train visualization -> $OUTDIR/spikes/"
python3 scripts/visualize_gesture_snn_spikes.py \
    --hparam-dir "$OUTDIR/hparam_sweep" --outdir "$OUTDIR/spikes"

echo "4/4: comparison vs. HDC methods -> $OUTDIR/comparison/"
python3 scripts/compare_gesture_methods.py \
    --hdc-dir results/results_gesture_3321861 \
    --snn-methods-dir "$OUTDIR/methods" \
    --outdir "$OUTDIR/comparison"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results in: $SLURM_SUBMIT_DIR/$OUTDIR/"
