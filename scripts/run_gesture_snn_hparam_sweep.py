"""Beta/alpha hyperparameter sweep for the two LIF neuron models (snn.Leaky,
snn.Synaptic) used by the SNN gesture-classification methods compared in
run_gesture_snn_method_comparison.py.

Both neuron models are swept using the same encoding -- delta modulation + rate
decoding (encode_delta_spike_trains + evaluate_snn's argmax-of-summed-spikes) --
since that's the one encoding shared between the two methods actually being
compared by neuron order (Delta+Rate+Leaky vs. Delta+Rate+Synaptic). The winning
beta (Leaky) and alpha+beta (Synaptic) are then reused for every method that
shares that neuron type in run_gesture_snn_method_comparison.py, including the
Rate+Rate+Leaky method.

Usage:
    python run_gesture_snn_hparam_sweep.py [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs a reduced beta grid with fewer epochs, for a smoke test before
committing a full run.
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
LEAKY_BETAS = [0.9, 0.7, 0.5]
SYNAPTIC_ALPHA = 0.9
SYNAPTIC_BETAS = [0.9, 0.7, 0.5]
NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

ENCODE_FN = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)


def run_leaky_point(beta, train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, ENCODE_FN, model_cls=se.SNNGestureClassifierLeaky,
                          model_kwargs={"hidden_size": HIDDEN_SIZE, "beta": beta},
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, ENCODE_FN,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"neuron": "leaky", "beta": beta, "accuracy": accuracy, "per_class": per_class,
            "train_s": train_s, "eval_s": eval_s}


def run_synaptic_point(alpha, beta, train_by_class, test_by_class, device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, ENCODE_FN, model_cls=se.SNNGestureClassifierSynaptic,
                          model_kwargs={"hidden_size": HIDDEN_SIZE, "alpha": alpha, "beta": beta},
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, ENCODE_FN,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"neuron": "synaptic", "alpha": alpha, "beta": beta, "accuracy": accuracy,
            "per_class": per_class, "train_s": train_s, "eval_s": eval_s}


def plot_beta_sweep(outdir, leaky_results, synaptic_results):
    plt.figure(figsize=(8, 5))
    leaky_betas = [r["beta"] for r in leaky_results]
    plt.plot(leaky_betas, [r["accuracy"] * 100 for r in leaky_results], marker='o',
              label="snn.Leaky (Delta + Rate decoding)")
    synaptic_betas = [r["beta"] for r in synaptic_results]
    plt.plot(synaptic_betas, [r["accuracy"] * 100 for r in synaptic_results], marker='s',
              label=f"snn.Synaptic, alpha={SYNAPTIC_ALPHA} (Delta + Rate decoding)")
    plt.xlabel("beta")
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary SNN accuracy vs. beta")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_beta_snn_hparam.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_gesture_snn_hparam")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced beta grid + fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    leaky_betas = [0.9, 0.5] if args.quick else LEAKY_BETAS
    synaptic_betas = [0.9, 0.5] if args.quick else SYNAPTIC_BETAS
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    wall0 = time.time()

    leaky_results = []
    for beta in leaky_betas:
        print(f"\n[leaky beta={beta}] training...", flush=True)
        res = run_leaky_point(beta, train_by_class, test_by_class, device, num_epochs)
        leaky_results.append(res)
        print(f"[leaky beta={beta}] -> acc={res['accuracy']:.4f}", flush=True)

    synaptic_results = []
    for beta in synaptic_betas:
        print(f"\n[synaptic alpha={SYNAPTIC_ALPHA} beta={beta}] training...", flush=True)
        res = run_synaptic_point(SYNAPTIC_ALPHA, beta, train_by_class, test_by_class, device, num_epochs)
        synaptic_results.append(res)
        print(f"[synaptic alpha={SYNAPTIC_ALPHA} beta={beta}] -> acc={res['accuracy']:.4f}", flush=True)

    wall_s = time.time() - wall0
    print(f"\nAll points done in {wall_s:.1f}s wall.", flush=True)

    plot_beta_sweep(args.outdir, leaky_results, synaptic_results)

    accuracy_by_config = {f"leaky_beta{r['beta']}": r["accuracy"] for r in leaky_results}
    accuracy_by_config.update({
        f"synaptic_alpha{r['alpha']}_beta{r['beta']}": r["accuracy"] for r in synaptic_results
    })

    best_leaky = max(leaky_results, key=lambda r: r["accuracy"])
    best_synaptic = max(synaptic_results, key=lambda r: r["accuracy"])

    summary = {
        "config": {"leaky_betas": leaky_betas, "synaptic_alpha": SYNAPTIC_ALPHA,
                    "synaptic_betas": synaptic_betas, "hidden_size": HIDDEN_SIZE,
                    "num_epochs": num_epochs, "batch_size": BATCH_SIZE, "device": device,
                    "quick": args.quick, "wall_seconds": wall_s},
        "accuracy_by_config": accuracy_by_config,
        "best": {
            "leaky": {"beta": best_leaky["beta"], "accuracy": best_leaky["accuracy"]},
            "synaptic": {"alpha": best_synaptic["alpha"], "beta": best_synaptic["beta"],
                          "accuracy": best_synaptic["accuracy"]},
        },
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plot + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for r in leaky_results:
        print(f"  leaky   beta={r['beta']:<4}          accuracy={r['accuracy']:.4f}")
    for r in synaptic_results:
        print(f"  synaptic alpha={r['alpha']} beta={r['beta']:<4} accuracy={r['accuracy']:.4f}")
    print(f"\nBest leaky:    beta={best_leaky['beta']} -> {best_leaky['accuracy']:.4f}")
    print(f"Best synaptic: alpha={best_synaptic['alpha']} beta={best_synaptic['beta']} "
          f"-> {best_synaptic['accuracy']:.4f}")


if __name__ == "__main__":
    main()
