"""Error-visualization for a trained SNN gesture classifier on the UWave test set.

Trains one SNN config (default: delta-modulation + rate decoding + snn.Synaptic,
with the best hyperparameters found by run_gesture_snn_hparam_sweep.py:
alpha=0.9, beta=0.5, threshold=0.25) and visualizes *how* it gets things wrong,
since the 8 gesture labels are categorical (no numeric "distance" between a
predicted and true class the way there would be for regression):

  1. confusion_matrix_snn.png -- 8x8 heatmap of true label (rows) vs. predicted
     label (cols), the primary "which classes get confused with which" view.
  2. error_margins_snn.png -- histogram of the rate-decoded margin (true class's
     spike count minus the top other class's spike count) for correct vs.
     incorrect predictions, showing whether errors are close calls or confident
     misfires.
  3. accuracy_per_class_snn.png -- bar chart of per-class accuracy.

Usage:
    python plot_snn_error_analysis.py [--method rate_leaky|delta_leaky|delta_synaptic]
                                       [--outdir DIR] [--quick] [--device cuda|cpu]
"""
import argparse
import json
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch

import snn_encoder as se
import uwave_data as data

NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

# Best hyperparameters from run_gesture_snn_hparam_sweep.py
# (results_gesture_snn_comparison_3325060/hparam_sweep/results.json "best" block).
LEAKY_BETA, LEAKY_THRESHOLD = 0.9, 0.25
SYNAPTIC_ALPHA, SYNAPTIC_BETA, SYNAPTIC_THRESHOLD = 0.9, 0.5, 0.25

LABELS = {
    "rate_leaky": "Rate + Rate decoding + Leaky",
    "delta_leaky": "Delta + Rate decoding + Leaky",
    "delta_synaptic": "Delta + Rate decoding + Synaptic",
}


def build_method(name):
    delta_encode_fn = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)
    methods = {
        "rate_leaky": (se.encode_rate_spike_trains, se.SNNGestureClassifierLeaky,
                        {"hidden_size": HIDDEN_SIZE, "beta": LEAKY_BETA, "threshold": LEAKY_THRESHOLD}),
        "delta_leaky": (delta_encode_fn, se.SNNGestureClassifierLeaky,
                         {"hidden_size": HIDDEN_SIZE, "beta": LEAKY_BETA, "threshold": LEAKY_THRESHOLD}),
        "delta_synaptic": (delta_encode_fn, se.SNNGestureClassifierSynaptic,
                            {"hidden_size": HIDDEN_SIZE, "alpha": SYNAPTIC_ALPHA,
                             "beta": SYNAPTIC_BETA, "threshold": SYNAPTIC_THRESHOLD}),
    }
    return methods[name]


def plot_confusion_matrix(outdir, confusion, method_name):
    classes = data.CLASSES
    matrix = np.array([[confusion[t][p] for p in classes] for t in classes])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.suptitle("UWaveGestureLibrary: confusion matrix", y=0.98)
    plt.title(LABELS[method_name], fontsize=9)

    vmax = matrix.max()
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = matrix[i, j]
            color = "white" if count > vmax / 2 else "black"
            plt.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix_snn.png"), dpi=150)
    plt.close()


def plot_error_margins(outdir, margins, method_name):
    correct_margins = [m for is_correct, m in margins if is_correct]
    wrong_margins = [m for is_correct, m in margins if not is_correct]

    plt.figure(figsize=(7, 5))
    bins = np.linspace(min(m for _, m in margins), max(m for _, m in margins), 30)
    plt.hist(correct_margins, bins=bins, alpha=0.6, label="correct", color="#55A868")
    plt.hist(wrong_margins, bins=bins, alpha=0.6, label="incorrect", color="#C44E52")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Margin (true class spike count - top other class spike count)")
    plt.ylabel("Number of test examples")
    plt.suptitle("UWaveGestureLibrary: rate-decoded margins", y=0.98)
    plt.title(LABELS[method_name], fontsize=9)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_margins_snn.png"), dpi=150)
    plt.close()


def plot_per_class_accuracy(outdir, per_class, method_name):
    classes = data.CLASSES
    accs = [per_class[c] * 100 for c in classes]

    plt.figure(figsize=(7, 5))
    plt.bar(classes, accs, color="#4C72B0")
    plt.xlabel("True class")
    plt.ylabel("Inference accuracy [%]")
    plt.suptitle("UWaveGestureLibrary: per-class accuracy", y=0.98)
    plt.title(LABELS[method_name], fontsize=9)
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_per_class_snn.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="delta_synaptic",
                         choices=["rate_leaky", "delta_leaky", "delta_synaptic"])
    parser.add_argument("--outdir", default="results_gesture_snn_error")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                         help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}, method: {args.method}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    encode_fn, model_cls, model_kwargs = build_method(args.method)

    t0 = time.time()
    model = se.train_snn(train_by_class, encode_fn, model_cls=model_cls, model_kwargs=model_kwargs,
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0
    print(f"Trained in {train_s:.1f}s.", flush=True)

    t0 = time.time()
    accuracy, per_class, confusion, margins = se.evaluate_snn(
        model, test_by_class, encode_fn, batch_size=BATCH_SIZE, device=device,
        return_per_class=True, return_confusion=True)
    eval_s = time.time() - t0
    print(f"Evaluated in {eval_s:.1f}s -> accuracy={accuracy:.4f}", flush=True)

    plot_confusion_matrix(args.outdir, confusion, args.method)
    plot_error_margins(args.outdir, margins, args.method)
    plot_per_class_accuracy(args.outdir, per_class, args.method)

    summary = {
        "config": {"method": args.method, "hidden_size": HIDDEN_SIZE, "num_epochs": num_epochs,
                    "batch_size": BATCH_SIZE, "leaky_beta": LEAKY_BETA, "leaky_threshold": LEAKY_THRESHOLD,
                    "synaptic_alpha": SYNAPTIC_ALPHA, "synaptic_beta": SYNAPTIC_BETA,
                    "synaptic_threshold": SYNAPTIC_THRESHOLD, "device": device, "quick": args.quick,
                    "train_seconds": train_s, "eval_seconds": eval_s},
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "confusion": confusion,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plots + results.json to {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
