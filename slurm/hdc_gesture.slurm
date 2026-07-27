#!/bin/bash
#----------------------------------------------------------------------------
# hdc_gesture.slurm -- run the UWaveGestureLibrary permutation-vs-Nystrom+GAK
# sweep on TACC Lonestar6.
#
# Target: the "normal" queue (128-core AMD EPYC Milan CPU nodes), same rationale
# as hdc_nystrom.slurm -- the bottlenecks here are a pure-Python per-timestep loop
# (permutation_encoder.encode_example_to_hv) and tslearn's GAK kernel evaluation
# (called once per example-landmark pair), neither of which is dense-matmul-bound.
# There is no GPU code in this project; a GPU would idle on both bottlenecks.
#
# GAK is the expensive part (~1.5ms/pair on this data). run_gesture_experiments.py
# computes each landmark-count's GAK kernel matrices ONCE and reuses them across
# the whole dimension sweep, so the dimension sweep itself is cheap once its one
# kernel build (at a fixed landmark count) completes; only the separate
# num-landmark sweep pays the GAK cost repeatedly, once per landmark count tested.
#
# Submit (from the repo root):   sbatch slurm/hdc_gesture.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_gesture.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_gesture              # job name
#SBATCH -o logs/hdc_gesture.%j.out  # stdout  (%j = job id)
#SBATCH -e logs/hdc_gesture.%j.err  # stderr
#SBATCH -p gpu-h100                   # queue (LS6 CPU partition)
#SBATCH -N 1                        # nodes
#SBATCH -n 1                        # tasks (single multiprocessing driver)
#SBATCH -t 06:00:00                 # wall time hh:mm:ss (generous; GAK dominates cost)
#SBATCH --mail-type=begin,end,fail
##SBATCH --mail-user=you@example.com   # <-- uncomment + set to get email notifications
##SBATCH -A YOUR_ALLOCATION            # <-- uncomment + set if your account needs -A

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Node has $(nproc) cores"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_gesture.slurm` from there

# --- Environment ------------------------------------------------------------
# Third-party deps: numpy, matplotlib (as in hdc_nystrom.slurm) PLUS tslearn for
# GAK (tslearn.metrics.cdist_gak/sigma_gak). tslearn pulls in scipy/scikit-learn/
# numba transitively -- all pure-Python-installable under TACC's system Python,
# no compiler toolchain expected to be needed, but this is new territory for this
# repo (previously numpy+matplotlib only) so watch the install step's output on
# first run.
module load python3
python3 -c "import numpy, matplotlib, tslearn" || {
    echo "numpy/matplotlib/tslearn not found under system python3 -- installing to --user site-packages"
    pip3 install --user numpy matplotlib tslearn
}

# --- Threading: 1 BLAS thread per worker ------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Run --------------------------------------------------------------------
WORKERS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
OUTDIR="results_gesture_${SLURM_JOB_ID}"

echo "Running gesture sweep with $WORKERS workers -> $OUTDIR/"
python3 scripts/run_gesture_experiments.py --workers "$WORKERS" --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results + plots in: $SLURM_SUBMIT_DIR/$OUTDIR/"
