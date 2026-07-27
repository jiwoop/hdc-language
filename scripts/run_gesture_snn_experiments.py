"""Batch driver for the UWaveGestureLibrary SNN baseline (snnTorch), for TACC's
gpu-h100 partition.

A third point of comparison alongside run_gesture_experiments.py's two HDC methods
(permutation baseline, Nystrom+GAK) -- same train/test split (uwave_data.py), same
accuracy metric, but a genuinely different model class (a trained feedforward SNN,
not a bundled hypervector). See snn_encoder.py's module docstring for the model/
encoding details.

Why GPU (unlike hdc_gesture.slurm's CPU-only rationale): the SNN training loop is
dense-matmul-bound (fc1/fc2 applied every timestep, every epoch) -- the first
workload in this repo shaped for a GPU. The two HDC encoders stay CPU-only and
unchanged; nothing here touches them.

Why NOT a combined wall-clock comparison: GPU vs CPU time isn't a fair comparison,
so this driver (and hdc_gesture_snn.slurm) never compares timing against the HDC
methods -- only accuracy, which is hardware-independent.

Why no multiprocessing.Pool (unlike the HDC drivers): GPU work doesn't parallelize
across a process pool the way independent CPU sweep points do -- there is one GPU,
so sweep points run sequentially, reusing the loaded train/test data across points.

Usage:
    python run_gesture_snn_experiments.py [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs a reduced hidden_size grid with fewer epochs, for a smoke test before
committing a full run.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import torch

import snn_encoder as se
import uwave_data as data

# ---------------------------------------------------------------------------
# Experiment grid.
# ---------------------------------------------------------------------------
# Primary sweep axis for the SNN is hidden_size -- the model-capacity knob
# analogous to the HDC methods' vector dimension D, though not the same quantity
# (a hidden layer width vs. a bundled hypervector width), so results are reported
# on their own plot/axis rather than merged onto accuracy_vs_dimension_gesture.png.
HIDDEN_SIZES = [16, 32, 64, 128]
NUM_EPOCHS = 30
BATCH_SIZE = 32


def run_point(hidden_size, train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, hidden_size=hidden_size, num_epochs=num_epochs,
                          batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, batch_size=BATCH_SIZE,
                                           device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"method": "snn", "hidden_size": hidden_size, "num_epochs": num_epochs,
            "accuracy": accuracy, "per_class": per_class, "train_s": train_s, "eval_s": eval_s}


def plot_hidden_size(outdir, by_hidden):
    xs = sorted(by_hidden)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [by_hidden[x] * 100 for x in xs], marker='o', label="SNN (snnTorch, rate-coded)")
    plt.xlabel("Hidden layer size")
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary SNN accuracy vs. hidden layer size")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_hidden_size_snn.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_gesture_snn")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid + fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    hidden_sizes = [16, 64] if args.quick else HIDDEN_SIZES
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    results = []
    wall0 = time.time()
    for hidden_size in hidden_sizes:
        print(f"\n[hidden_size={hidden_size}] training...", flush=True)
        res = run_point(hidden_size, train_by_class, test_by_class, device, num_epochs)
        results.append(res)
        print(f"[hidden_size={hidden_size}] -> acc={res['accuracy']:.4f} "
              f"(train {res['train_s']:.1f}s, eval {res['eval_s']:.1f}s)", flush=True)

    wall_s = time.time() - wall0
    print(f"\nAll points done in {wall_s:.1f}s wall.", flush=True)

    by_hidden = {r["hidden_size"]: r["accuracy"] for r in results}
    per_class_by_hidden = {r["hidden_size"]: r["per_class"] for r in results}

    plot_hidden_size(args.outdir, by_hidden)

    summary = {
        "config": {"hidden_sizes": hidden_sizes, "num_epochs": num_epochs,
                    "batch_size": BATCH_SIZE, "device": device, "quick": args.quick,
                    "wall_seconds": wall_s},
        "accuracy_by_hidden_size": by_hidden,
        "per_class_by_hidden_size": per_class_by_hidden,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plot + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for hidden_size in sorted(by_hidden):
        print(f"  hidden_size={hidden_size:4d}  accuracy={by_hidden[hidden_size]:.4f}")


if __name__ == "__main__":
    main()
