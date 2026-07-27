#!/bin/bash
#----------------------------------------------------------------------------
# hdc_nystrom.slurm -- run the full hdc_nystrom.ipynb sweep on TACC Lonestar6.
#
# Target: the "normal" queue (128-core AMD EPYC Milan CPU nodes). The workload is
# CPU-bound pure-Python k-mer extraction fanned out over independent (D,k,landmark,...)
# points, so a single 128-core node running one process per core is the right shape --
# NOT a GPU node (there is no GPU code in this project; a GPU would idle on the bottleneck).
#
# One node is enough: the whole sweep (round 1 + round 2's added landmark-count/bit-budget
# sweeps, ~75 points total) still runs with all 128 cores busy on a single node, dominated by
# the single slowest point rather than the sum. Scaling past one node would need MPI/launcher
# plumbing this pure-multiprocessing driver doesn't use.
#
# Submit (from the repo root):   sbatch slurm/hdc_nystrom.slurm
# Monitor:  squeue -u $USER      /      tail -f logs/hdc_nystrom.<jobid>.out
#----------------------------------------------------------------------------
#SBATCH -J hdc_nystrom              # job name
#SBATCH -o logs/hdc_nystrom.%j.out  # stdout  (%j = job id)
#SBATCH -e logs/hdc_nystrom.%j.err  # stderr
#SBATCH -p normal              # queue (LS6 CPU partition)
#SBATCH -N 1                        # nodes
#SBATCH -n 1                        # tasks (single multiprocessing driver)
#SBATCH -t 02:00:00                 # wall time hh:mm:ss (bumped: fine landmark sweep + stratified
                                    # comparison now run 10 seeded repeats each for averaging)
#SBATCH --mail-type=begin,end,fail
##SBATCH --mail-user=jiwooo.park04@gmail.com  # <-- uncomment + set to get email notifications
##SBATCH -A YOUR_ALLOCATION            # <-- uncomment + set if your account needs -A

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Node has $(nproc) cores"

cd "$SLURM_SUBMIT_DIR"  # repo root -- must submit via `sbatch slurm/hdc_nystrom.slurm` from there

# --- Environment ------------------------------------------------------------
# No conda needed: the only third-party deps are numpy + matplotlib, both installable
# under TACC's system python3 module. Auto-installs to --user site-packages the first
# time (persists across jobs, so this is a no-op on every run after the first).
module load python3
python3 -c "import numpy, matplotlib" || {
    echo "numpy/matplotlib not found under system python3 -- installing to --user site-packages"
    pip3 install --user numpy matplotlib
}

# --- Threading: 1 BLAS thread per worker ------------------------------------
# The driver spawns ~128 Python worker processes, each doing its own matmuls. If BLAS
# also spawned 128 threads per worker we'd oversubscribe the node 128x and thrash. Pin
# every math backend to a single thread so the parallelism lives entirely at the
# process-pool level (one worker per core).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Run --------------------------------------------------------------------
# --workers defaults to SLURM_CPUS_ON_NODE inside the script; pass it explicitly to be safe.
WORKERS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
OUTDIR="results_${SLURM_JOB_ID}"

echo "Running full sweep with $WORKERS workers -> $OUTDIR/"
python3 scripts/run_experiments.py --workers "$WORKERS" --outdir "$OUTDIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results + plots in: $SLURM_SUBMIT_DIR/$OUTDIR/"
