"""Tests two fixes for the SNN gesture classifier's lack of temporal-order sensitivity
(see confusion-matrix/margin analysis in plot_snn_error_analysis.py -- a flat
feedforward SNN with rate decoding has only a single fixed-rate membrane decay as
"memory," and rate decoding (summing spikes over all 315 steps) discards spike order
entirely):

  1. Latency/temporal decoding -- decode=argmin(first spike time) instead of
     argmax(spike count), trained with SF.ce_temporal_loss() instead of
     SF.ce_rate_loss(). Makes *when* a class neuron first fires the signal, not just
     how many times.
  2. Multiple LIF layers with different beta -- SNNGestureClassifierMultiBeta
     (fc1->lif1[fast beta]->fc2->lif2[slow beta]->fc3->lif3) gives the hidden layers
     two different membrane-decay timescales instead of one shared beta.

Compares 4 configs, all using delta-modulation encoding (the best-performing input
coding from run_gesture_snn_method_comparison.py) and the best-known Synaptic/Leaky
hyperparameters from run_gesture_snn_hparam_sweep.py's results.json ("best" block):
  - baseline:            SNNGestureClassifierSynaptic, rate decode   (known best config)
  - temporal:             SNNGestureClassifierSynaptic, temporal decode
  - multibeta:            SNNGestureClassifierMultiBeta, rate decode
  - multibeta_temporal:   SNNGestureClassifierMultiBeta, temporal decode

Usage:
    python run_gesture_snn_temporal_multibeta.py [--outdir DIR] [--quick] [--device cuda|cpu]
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

import snntorch.functional as SF

import snn_encoder as se
import uwave_data as data

NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

# Best hyperparameters from run_gesture_snn_hparam_sweep.py
# (results_gesture_snn_comparison_3325060/hparam_sweep/results.json "best" block).
SYNAPTIC_ALPHA, SYNAPTIC_BETA, SYNAPTIC_THRESHOLD = 0.9, 0.5, 0.25
MULTIBETA_FAST, MULTIBETA_SLOW, MULTIBETA_OUT, MULTIBETA_THRESHOLD = 0.5, 0.95, 0.5, 0.25

LABELS = {
    "baseline": "Synaptic, rate decode (known best)",
    "temporal": "Synaptic, temporal decode",
    "multibeta": "MultiBeta, rate decode",
    "multibeta_temporal": "MultiBeta, temporal decode",
}
COLORS = {"baseline": "#4C72B0", "temporal": "#DD8452",
          "multibeta": "#55A868", "multibeta_temporal": "#8172B2"}


def build_configs():
    delta_encode_fn = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)
    synaptic_kwargs = {"hidden_size": HIDDEN_SIZE, "alpha": SYNAPTIC_ALPHA,
                        "beta": SYNAPTIC_BETA, "threshold": SYNAPTIC_THRESHOLD}
    multibeta_kwargs = {"hidden_size": HIDDEN_SIZE, "beta_fast": MULTIBETA_FAST,
                         "beta_slow": MULTIBETA_SLOW, "beta_out": MULTIBETA_OUT,
                         "threshold": MULTIBETA_THRESHOLD}
    return {
        "baseline": (delta_encode_fn, se.SNNGestureClassifierSynaptic, synaptic_kwargs,
                     SF.ce_rate_loss(), "rate"),
        "temporal": (delta_encode_fn, se.SNNGestureClassifierSynaptic, synaptic_kwargs,
                     SF.ce_temporal_loss(), "temporal"),
        "multibeta": (delta_encode_fn, se.SNNGestureClassifierMultiBeta, multibeta_kwargs,
                      SF.ce_rate_loss(), "rate"),
        "multibeta_temporal": (delta_encode_fn, se.SNNGestureClassifierMultiBeta, multibeta_kwargs,
                                SF.ce_temporal_loss(), "temporal"),
    }


def run_config(name, encode_fn, model_cls, model_kwargs, loss_fn, decode,
                train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, encode_fn, model_cls=model_cls, model_kwargs=model_kwargs,
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device, loss_fn=loss_fn)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class, confusion, margins = se.evaluate_snn(
        model, test_by_class, encode_fn, batch_size=BATCH_SIZE, device=device,
        return_per_class=True, return_confusion=True, decode=decode)
    eval_s = time.time() - t0

    return {"config": name, "accuracy": accuracy, "per_class": per_class,
            "confusion": confusion, "margins": margins, "train_s": train_s, "eval_s": eval_s}


def plot_accuracy_comparison(outdir, results):
    names = [r["config"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, accs, color=colors)
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary: temporal decoding + multi-beta comparison", fontsize=10)
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison_temporal_multibeta.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(outdir, confusion, name):
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
    plt.title(LABELS[name], fontsize=9)

    vmax = matrix.max()
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = matrix[i, j]
            color = "white" if count > vmax / 2 else "black"
            plt.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_gesture_snn_temporal_multibeta")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                         help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    configs = build_configs()
    results = []
    wall0 = time.time()
    for name, (encode_fn, model_cls, model_kwargs, loss_fn, decode) in configs.items():
        print(f"\n[{name}] training ({LABELS[name]})...", flush=True)
        res = run_config(name, encode_fn, model_cls, model_kwargs, loss_fn, decode,
                          train_by_class, test_by_class, device, num_epochs)
        results.append(res)
        print(f"[{name}] -> acc={res['accuracy']:.4f} "
              f"(train {res['train_s']:.1f}s, eval {res['eval_s']:.1f}s)", flush=True)
        plot_confusion_matrix(args.outdir, res["confusion"], name)

    wall_s = time.time() - wall0
    print(f"\nAll configs done in {wall_s:.1f}s wall.", flush=True)

    plot_accuracy_comparison(args.outdir, results)

    summary = {
        "config": {"hidden_size": HIDDEN_SIZE, "num_epochs": num_epochs, "batch_size": BATCH_SIZE,
                    "synaptic_alpha": SYNAPTIC_ALPHA, "synaptic_beta": SYNAPTIC_BETA,
                    "synaptic_threshold": SYNAPTIC_THRESHOLD,
                    "multibeta_fast": MULTIBETA_FAST, "multibeta_slow": MULTIBETA_SLOW,
                    "multibeta_out": MULTIBETA_OUT, "multibeta_threshold": MULTIBETA_THRESHOLD,
                    "device": device, "quick": args.quick, "wall_seconds": wall_s},
        "accuracy_by_config": {r["config"]: r["accuracy"] for r in results},
        "per_class_by_config": {r["config"]: r["per_class"] for r in results},
        "confusion_by_config": {r["config"]: r["confusion"] for r in results},
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plots + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {LABELS[r['config']]:<35} accuracy={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
