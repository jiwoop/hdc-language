"""Threshold/beta/alpha hyperparameter sweep for the two LIF neuron models
(snn.Leaky, snn.Synaptic) used by the SNN gesture-classification methods
compared in run_gesture_snn_method_comparison.py.

Both neuron models are swept using the same encoding -- delta modulation + rate
decoding (encode_delta_spike_trains + evaluate_snn's argmax-of-summed-spikes) --
since that's the one encoding shared between the two methods actually being
compared by neuron order (Delta+Rate+Leaky vs. Delta+Rate+Synaptic). The winning
beta+threshold (Leaky) and alpha+beta+threshold (Synaptic) are then reused for
every method that shares that neuron type in run_gesture_snn_method_comparison.py,
including the Rate+Rate+Leaky method.

spike_threshold is swept alongside beta/alpha (not just beta/alpha alone) because
snn_encoder.DEFAULT_SPIKE_THRESHOLD=0.9 was found to chronically under-fire the
output layer with these small, randomly-initialized fc1/fc2 weights: ~76% of all
(timestep, example) pairs produce zero output spikes across all 8 classes. Since
SF.ce_rate_loss applies cross-entropy independently at every timestep, an
all-silent timestep contributes an identical, label-independent ln(8) loss term
no matter what -- diluting the training signal to ~24% of timesteps and making
loss/accuracy crawl improve only glacially over many epochs. Sweeping a lower
threshold alongside beta/alpha directly targets this.

Usage:
    python run_gesture_snn_hparam_sweep.py [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs a reduced threshold/beta grid with fewer epochs, for a smoke test
before committing a full run.
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
import torch

import snn_encoder as se
import uwave_data as data

# ---------------------------------------------------------------------------
# Experiment grid.
# ---------------------------------------------------------------------------
THRESHOLDS = [0.9, 0.5, 0.25]
LEAKY_BETAS = [0.9, 0.7, 0.5]
SYNAPTIC_ALPHA = 0.9
SYNAPTIC_BETAS = [0.9, 0.7, 0.5]
NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

ENCODE_FN = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)


def run_leaky_point(beta, threshold, train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, ENCODE_FN, model_cls=se.SNNGestureClassifierLeaky,
                          model_kwargs={"hidden_size": HIDDEN_SIZE, "beta": beta, "threshold": threshold},
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, ENCODE_FN,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"neuron": "leaky", "beta": beta, "threshold": threshold, "accuracy": accuracy,
            "per_class": per_class, "train_s": train_s, "eval_s": eval_s}


def run_synaptic_point(alpha, beta, threshold, train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, ENCODE_FN, model_cls=se.SNNGestureClassifierSynaptic,
                          model_kwargs={"hidden_size": HIDDEN_SIZE, "alpha": alpha, "beta": beta,
                                        "threshold": threshold},
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, ENCODE_FN,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"neuron": "synaptic", "alpha": alpha, "beta": beta, "threshold": threshold,
            "accuracy": accuracy, "per_class": per_class, "train_s": train_s, "eval_s": eval_s}


def plot_threshold_beta_sweep(outdir, leaky_results, synaptic_results, thresholds):
    fig, (ax_leaky, ax_synaptic) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    markers = ['o', 's', '^', 'd']

    for i, threshold in enumerate(thresholds):
        pts = [r for r in leaky_results if r["threshold"] == threshold]
        pts.sort(key=lambda r: r["beta"])
        ax_leaky.plot([r["beta"] for r in pts], [r["accuracy"] * 100 for r in pts],
                       marker=markers[i % len(markers)], label=f"threshold={threshold}")

        pts = [r for r in synaptic_results if r["threshold"] == threshold]
        pts.sort(key=lambda r: r["beta"])
        ax_synaptic.plot([r["beta"] for r in pts], [r["accuracy"] * 100 for r in pts],
                          marker=markers[i % len(markers)], label=f"threshold={threshold}")

    ax_leaky.set_title("snn.Leaky")
    ax_leaky.set_xlabel("beta")
    ax_leaky.set_ylabel("Inference accuracy [%]")
    ax_leaky.legend(); ax_leaky.grid(True, alpha=0.3)

    ax_synaptic.set_title(f"snn.Synaptic (alpha={SYNAPTIC_ALPHA})")
    ax_synaptic.set_xlabel("beta")
    ax_synaptic.legend(); ax_synaptic.grid(True, alpha=0.3)

    fig.suptitle("UWaveGestureLibrary SNN accuracy vs. beta, by spike threshold (Delta + Rate decoding)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_beta_snn_hparam.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_gesture_snn_hparam")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced threshold/beta grid + fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    thresholds = [0.9, 0.25] if args.quick else THRESHOLDS
    leaky_betas = [0.9, 0.5] if args.quick else LEAKY_BETAS
    synaptic_betas = [0.9, 0.5] if args.quick else SYNAPTIC_BETAS
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    wall0 = time.time()

    leaky_results = []
    for threshold in thresholds:
        for beta in leaky_betas:
            print(f"\n[leaky threshold={threshold} beta={beta}] training...", flush=True)
            res = run_leaky_point(beta, threshold, train_by_class, test_by_class, device, num_epochs)
            leaky_results.append(res)
            print(f"[leaky threshold={threshold} beta={beta}] -> acc={res['accuracy']:.4f}", flush=True)

    synaptic_results = []
    for threshold in thresholds:
        for beta in synaptic_betas:
            print(f"\n[synaptic alpha={SYNAPTIC_ALPHA} beta={beta} threshold={threshold}] training...",
                  flush=True)
            res = run_synaptic_point(SYNAPTIC_ALPHA, beta, threshold, train_by_class, test_by_class,
                                      device, num_epochs)
            synaptic_results.append(res)
            print(f"[synaptic alpha={SYNAPTIC_ALPHA} beta={beta} threshold={threshold}] "
                  f"-> acc={res['accuracy']:.4f}", flush=True)

    wall_s = time.time() - wall0
    print(f"\nAll points done in {wall_s:.1f}s wall.", flush=True)

    plot_threshold_beta_sweep(args.outdir, leaky_results, synaptic_results, thresholds)

    accuracy_by_config = {
        f"leaky_threshold{r['threshold']}_beta{r['beta']}": r["accuracy"] for r in leaky_results
    }
    accuracy_by_config.update({
        f"synaptic_alpha{r['alpha']}_beta{r['beta']}_threshold{r['threshold']}": r["accuracy"]
        for r in synaptic_results
    })

    best_leaky = max(leaky_results, key=lambda r: r["accuracy"])
    best_synaptic = max(synaptic_results, key=lambda r: r["accuracy"])

    summary = {
        "config": {"thresholds": thresholds, "leaky_betas": leaky_betas, "synaptic_alpha": SYNAPTIC_ALPHA,
                    "synaptic_betas": synaptic_betas, "hidden_size": HIDDEN_SIZE,
                    "num_epochs": num_epochs, "batch_size": BATCH_SIZE, "device": device,
                    "quick": args.quick, "wall_seconds": wall_s},
        "accuracy_by_config": accuracy_by_config,
        "best": {
            "leaky": {"beta": best_leaky["beta"], "threshold": best_leaky["threshold"],
                      "accuracy": best_leaky["accuracy"]},
            "synaptic": {"alpha": best_synaptic["alpha"], "beta": best_synaptic["beta"],
                          "threshold": best_synaptic["threshold"], "accuracy": best_synaptic["accuracy"]},
        },
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plot + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for r in leaky_results:
        print(f"  leaky    threshold={r['threshold']:<5} beta={r['beta']:<4} accuracy={r['accuracy']:.4f}")
    for r in synaptic_results:
        print(f"  synaptic alpha={r['alpha']} beta={r['beta']:<4} threshold={r['threshold']:<5} "
              f"accuracy={r['accuracy']:.4f}")
    print(f"\nBest leaky:    beta={best_leaky['beta']} threshold={best_leaky['threshold']} "
          f"-> {best_leaky['accuracy']:.4f}")
    print(f"Best synaptic: alpha={best_synaptic['alpha']} beta={best_synaptic['beta']} "
          f"threshold={best_synaptic['threshold']} -> {best_synaptic['accuracy']:.4f}")


if __name__ == "__main__":
    main()
