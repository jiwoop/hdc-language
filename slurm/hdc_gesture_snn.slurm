#!/bin/bash
#----------------------------------------------------------------------------
# hdc_gesture_snn.slurm -- run the UWaveGestureLibrary SNN baseline (snnTorch)
# on TACC's gpu-h100 partition.
#
# Target: gpu-h100 (1 GPU). Unlike hdc_gesture.slurm (CPU-only -- the two HDC
# encoders' bottlenecks are a per-timestep Python loop and tslearn's GAK, neither
# dense-matmul-bound), the SNN training loop (fc1/fc2 applied every timestep,
# every epoch) is genuinely matmul-bound and is the first workload in this repo
# shaped for a GPU. Kept as a separate script/queue rather than repurposing
# hdc_gesture.slurm, so that script's CPU-only rationale stays accurate.
#
# No wall-clock comparison against the CPU-only HDC methods is made anywhere in
# this project -- GPU vs CPU time isn't a fair comparison. Only accuracy
# (hardware-independent) is compared across all three methods.
#
# Submit (from the repo root):   sbatch slurm/hdc_gesture_snn.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_gesture_snn.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_gesture_snn           # job name
#SBATCH -o logs/hdc_gesture_snn.%j.out  # stdout  (%j = job id)
#SBATCH -e logs/hdc_gesture_snn.%j.err  # stderr
#SBATCH -p gpu-h100                  # queue (LS6 GPU partition)
#SBATCH -N 1                         # nodes
#SBATCH -n 1                         # tasks
#SBATCH -t 04:00:00                  # wall time hh:mm:ss
#SBATCH -A OTH21017                  # Allocation ID

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_gesture_snn.slurm` from there

# --- Environment ------------------------------------------------------------
# Third-party deps: numpy, matplotlib (as in the other two sweeps) PLUS
# torch + snntorch for the SNN model/training. torch is a new class of dependency
# for this repo (previously deliberately avoided -- see nystrom_encoder.py's
# docstring -- since the HDC methods never needed it); it's required here since
# the SNN itself needs autograd/backprop-through-time.
module load python3
python3 -c "import numpy, matplotlib, torch, snntorch" || {
    echo "numpy/matplotlib/torch/snntorch not found under system python3 -- installing to --user site-packages"
    pip3 install --user numpy matplotlib torch snntorch
}

# --- Run ----------------------------------------------------------------
OUTDIR="results_gesture_snn_${SLURM_JOB_ID}"

echo "Running gesture SNN sweep -> $OUTDIR/"
python3 scripts/run_gesture_snn_experiments.py --device cuda --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results + plot in: $SLURM_SUBMIT_DIR/$OUTDIR/"
