#!/bin/bash

#SBATCH -J sparch_lif_shd
#SBATCH -o logs/sparch_lif_shd.%j.out
#SBATCH -e logs/sparch_lif_shd.%j.err
#SBATCH -p normal                    # CPU queue -- no GPU code in this training loop
#SBATCH -N 1                         # nodes
#SBATCH -n 1                         # tasks
#SBATCH -t 04:00:00                  # wall time hh:mm:ss; ~27s/epoch locally, 50 epochs is well under this
#SBATCH -A OTH21017                  # Allocation ID -- adjust if this differs on your account
#SBATCH --mail-type=ALL              # email on BEGIN/END/FAIL
#SBATCH --mail-user=jiwooo.park04@gmail.com

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# Submit from the hdc-language repo root: `sbatch slurm/train_lif_shd.slurm`.
# This trains sparch (vendored as a git submodule at sparch/), not
# hdc-language itself, so cd into the submodule before running anything.
cd "$SLURM_SUBMIT_DIR/sparch"

# --- Environment ------------------------------------------------------------
# Dedicated venv on $SCRATCH pinned to sparch's own requirements.txt
# (torch==1.12.0, numpy==1.22.3, ...) -- distinct from hdc-language's
# snn-env, which targets a different (snntorch) stack/torch version.
VENV_DIR="/scratch/09424/jiwoooop/sparch-env"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR (first run only)"
    module load python3
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# --- Dataset ------------------------------------------------------------
# SHD (~330MB total) is too large for git -- shd_train.h5 alone (256MB)
# exceeds GitHub's 100MB hard file-size limit -- so it isn't vendored in
# the repo/submodule. Download + decompress on first run only; subsequent
# jobs on the same $SCRATCH reuse the already-extracted files.
DATA_DIR="shd_dataset"
mkdir -p "$DATA_DIR"

download_shd() {
    local name="$1"
    if [ -f "$DATA_DIR/${name}.h5" ]; then
        return
    fi
    echo "Downloading ${name}.h5.gz (first run only)"
    curl -fsSL "https://zenkelab.org/datasets/${name}.h5.gz" | gunzip > "$DATA_DIR/${name}.h5.tmp"
    mv "$DATA_DIR/${name}.h5.tmp" "$DATA_DIR/${name}.h5"
}

download_shd shd_train
download_shd shd_test

# --- Run ----------------------------------------------------------------
EXP_DIR="exp/lif_shd_run_${SLURM_JOB_ID}"

echo "Training LIF/SHD for 50 epochs -> $EXP_DIR/"
python3 run_exp.py \
    --model_type LIF \
    --dataset_name shd \
    --data_folder shd_dataset \
    --new_exp_folder "$EXP_DIR" \
    --nb_epochs 50

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Checkpoint: $SLURM_SUBMIT_DIR/sparch/$EXP_DIR/checkpoints/best_model.pth"
