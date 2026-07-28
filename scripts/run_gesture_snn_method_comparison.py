"""Accuracy comparison of the three SNN encoding/decoding/neuron-model methods:

  1. rate_leaky:     Rate coding      + Rate decoding + snn.Leaky
  2. delta_leaky:    Delta modulation + Rate decoding + snn.Leaky
  3. delta_synaptic: Delta modulation + Rate decoding + snn.Synaptic

Rate decoding (sum of output spikes per class, argmax) is shared by all three --
see snn_encoder.py's evaluate_snn. Neuron hyperparameters (beta+threshold for
Leaky; alpha, beta+threshold for Synaptic) are read from
run_gesture_snn_hparam_sweep.py's results.json ("best" block) rather than
re-tuned here, and reused across every method that shares that neuron type
(including method 1, which never appears in the beta sweep). hidden_size is held
fixed at snn_encoder.DEFAULT_HIDDEN_SIZE for all three methods -- this comparison
doesn't re-sweep model capacity.

Usage:
    python run_gesture_snn_method_comparison.py --hparam-dir DIR [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs fewer epochs, for a smoke test before committing a full run.
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

NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

LABELS = {
    "rate_leaky": "Rate + Rate decoding + Leaky",
    "delta_leaky": "Delta + Rate decoding + Leaky",
    "delta_synaptic": "Delta + Rate decoding + Synaptic",
}
COLORS = {"rate_leaky": "#4C72B0", "delta_leaky": "#DD8452", "delta_synaptic": "#55A868"}


def load_best_hparams(hparam_dir):
    with open(os.path.join(hparam_dir, "results.json")) as f:
        summary = json.load(f)
    best = summary["best"]
    return (best["leaky"]["beta"], best["leaky"]["threshold"],
            best["synaptic"]["alpha"], best["synaptic"]["beta"], best["synaptic"]["threshold"])


def run_method(name, encode_fn, model_cls, model_kwargs, train_by_class, test_by_class,
                device, num_epochs):
    t0 = time.time()
    model = se.train_snn(train_by_class, encode_fn, model_cls=model_cls, model_kwargs=model_kwargs,
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class = se.evaluate_snn(model, test_by_class, encode_fn,
                                           batch_size=BATCH_SIZE, device=device, return_per_class=True)
    eval_s = time.time() - t0

    return {"method": name, "accuracy": accuracy, "per_class": per_class,
            "train_s": train_s, "eval_s": eval_s}


def plot_method_comparison(outdir, results):
    methods = [r["method"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    labels = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, accs, color=colors)
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary: SNN method comparison")
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison_snn_methods.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam-dir", required=True,
                        help="Output dir from run_gesture_snn_hparam_sweep.py.")
    parser.add_argument("--outdir", default="results_gesture_snn_methods")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    num_epochs = 3 if args.quick else NUM_EPOCHS
    leaky_beta, leaky_threshold, synaptic_alpha, synaptic_beta, synaptic_threshold = \
        load_best_hparams(args.hparam_dir)

    print(f"device: {device}", flush=True)
    print(f"leaky beta={leaky_beta} threshold={leaky_threshold}, "
          f"synaptic alpha={synaptic_alpha} beta={synaptic_beta} threshold={synaptic_threshold}", flush=True)
    print("Loading data...", flush=True)
    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    delta_encode_fn = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)

    methods = [
        ("rate_leaky", se.encode_rate_spike_trains, se.SNNGestureClassifierLeaky,
         {"hidden_size": HIDDEN_SIZE, "beta": leaky_beta, "threshold": leaky_threshold}),
        ("delta_leaky", delta_encode_fn, se.SNNGestureClassifierLeaky,
         {"hidden_size": HIDDEN_SIZE, "beta": leaky_beta, "threshold": leaky_threshold}),
        ("delta_synaptic", delta_encode_fn, se.SNNGestureClassifierSynaptic,
         {"hidden_size": HIDDEN_SIZE, "alpha": synaptic_alpha, "beta": synaptic_beta,
          "threshold": synaptic_threshold}),
    ]

    results = []
    wall0 = time.time()
    for name, encode_fn, model_cls, model_kwargs in methods:
        print(f"\n[{name}] training...", flush=True)
        res = run_method(name, encode_fn, model_cls, model_kwargs, train_by_class, test_by_class,
                          device, num_epochs)
        results.append(res)
        print(f"[{name}] -> acc={res['accuracy']:.4f} "
              f"(train {res['train_s']:.1f}s, eval {res['eval_s']:.1f}s)", flush=True)

    wall_s = time.time() - wall0
    print(f"\nAll methods done in {wall_s:.1f}s wall.", flush=True)

    plot_method_comparison(args.outdir, results)

    accuracy_by_method = {r["method"]: r["accuracy"] for r in results}
    per_class_by_method = {r["method"]: r["per_class"] for r in results}

    summary = {
        "config": {"hidden_size": HIDDEN_SIZE, "num_epochs": num_epochs, "batch_size": BATCH_SIZE,
                    "leaky_beta": leaky_beta, "leaky_threshold": leaky_threshold,
                    "synaptic_alpha": synaptic_alpha, "synaptic_beta": synaptic_beta,
                    "synaptic_threshold": synaptic_threshold, "device": device, "quick": args.quick,
                    "wall_seconds": wall_s, "hparam_dir": args.hparam_dir},
        "accuracy_by_method": accuracy_by_method,
        "per_class_by_method": per_class_by_method,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plot + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {LABELS[r['method']]:<35} accuracy={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
