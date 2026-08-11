"""Batch driver for the UWaveGestureLibrary SNN baseline (snnTorch).

Usage:
    python run_gesture_snn.py [--outdir DIR] [--quick] [--device cuda|cpu]

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
import numpy as np
import torch

import snn_encoder as se
import snntorch.functional as SF
import uwave_data as data

# ---------------------------------------------------------------------------
# Experiment grid.
# ---------------------------------------------------------------------------
# hidden_size is the *first* hidden layer's width; each model class halves it per
# subsequent hidden layer internally (see snn_encoder._hidden_sizes_from_scalar).
HIDDEN_SIZES = [512, 256, 128, 64]
NUM_EPOCHS = 30
BATCH_SIZE = 32

# Decode schemes to compare: "rate" sums output spikes over time (order-blind),
# "temporal" scores by first-spike time (order-sensitive) -- see snn_encoder.py's
# evaluate_snn docstring. Each decode is paired with its matching loss below.
DECODES = ["rate", "temporal"]
LOSS_FNS = {"rate": SF.ce_rate_loss, "temporal": SF.ce_temporal_loss}


def _beta_kwargs(model_cls, num_layers, lo=0.9, hi=0.7):
    """Space beta linearly from `lo` (first hidden layer) to `hi` (output layer)
    across num_layers = num_hidden + 1 synaptic layers, keyed to match model_cls's
    beta_1..beta_{num_hidden}, beta_out constructor kwargs."""
    betas = np.linspace(lo, hi, num_layers).tolist()
    kwargs = {f"beta_{i+1}": b for i, b in enumerate(betas[:-1])}
    kwargs["beta_out"] = betas[-1]
    return kwargs


MODEL_CONFIGS = {
    "MultiBeta3": (se.SNNGestureClassifierMultiBeta3, _beta_kwargs(se.SNNGestureClassifierMultiBeta3, 3)),
    "MultiBeta4": (se.SNNGestureClassifierMultiBeta4, _beta_kwargs(se.SNNGestureClassifierMultiBeta4, 4)),
    "MultiBeta5": (se.SNNGestureClassifierMultiBeta5, _beta_kwargs(se.SNNGestureClassifierMultiBeta5, 5)),
}


def run_point(hidden_size, train_by_class, test_by_class, model_cls, beta_kwargs, decode, device, num_epochs):
    loss_fn = LOSS_FNS[decode]()

    t0 = time.time()
    model = se.train_snn(train_by_class, se.encode_rate_spike_trains,
                          model_cls,
                          model_kwargs={"hidden_size": hidden_size, **beta_kwargs}, num_epochs=num_epochs,
                          batch_size=BATCH_SIZE, device=device, loss_fn=loss_fn)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, se.encode_rate_spike_trains,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True,
                                           decode=decode)
    eval_s = time.time() - t0

    return {"method": "snn", "hidden_size": hidden_size, "decode": decode, "num_epochs": num_epochs,
            "accuracy": accuracy, "per_class": per_class, "train_s": train_s, "eval_s": eval_s}


def plot_hidden_size(outdir, by_model_hidden):
    plt.figure(figsize=(8, 5))
    for series_name, by_hidden in by_model_hidden.items():
        xs = sorted(by_hidden)
        plt.plot(xs, [by_hidden[x] * 100 for x in xs], marker='o',
                  label=f"SNN {series_name} (snnTorch)")
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

    hidden_sizes = [512, 64] if args.quick else HIDDEN_SIZES
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    results = []
    wall0 = time.time()
    for model_name, (model_cls, beta_kwargs) in MODEL_CONFIGS.items():
        for decode in DECODES:
            for hidden_size in hidden_sizes:
                print(f"\n[{model_name} decode={decode} hidden_size={hidden_size}] training...", flush=True)
                res = run_point(hidden_size, train_by_class, test_by_class, model_cls, beta_kwargs, decode,
                                 device, num_epochs)
                res["model"] = model_name
                results.append(res)
                print(f"[{model_name} decode={decode} hidden_size={hidden_size}] -> acc={res['accuracy']:.4f} "
                      f"(train {res['train_s']:.1f}s, eval {res['eval_s']:.1f}s)", flush=True)

    wall_s = time.time() - wall0
    print(f"\nAll points done in {wall_s:.1f}s wall.", flush=True)

    series_names = [f"{model_name}_{decode}" for model_name in MODEL_CONFIGS for decode in DECODES]
    by_series_hidden = {name: {} for name in series_names}
    per_class_by_series_hidden = {name: {} for name in series_names}
    for r in results:
        series = f"{r['model']}_{r['decode']}"
        by_series_hidden[series][r["hidden_size"]] = r["accuracy"]
        per_class_by_series_hidden[series][r["hidden_size"]] = r["per_class"]

    plot_hidden_size(args.outdir, by_series_hidden)

    summary = {
        "config": {"models": list(MODEL_CONFIGS), "decodes": DECODES, "hidden_sizes": hidden_sizes,
                    "num_epochs": num_epochs, "batch_size": BATCH_SIZE, "device": device, "quick": args.quick,
                    "wall_seconds": wall_s},
        "accuracy_by_series_hidden_size": by_series_hidden,
        "per_class_by_series_hidden_size": per_class_by_series_hidden,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plot + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for series in series_names:
        for hidden_size in sorted(by_series_hidden[series]):
            print(f"  {series} hidden_size={hidden_size:4d}  accuracy={by_series_hidden[series][hidden_size]:.4f}")


if __name__ == "__main__":
    main()
