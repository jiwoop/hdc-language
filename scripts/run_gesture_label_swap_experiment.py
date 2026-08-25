"""Checks whether the SNN gesture classifier's lopsided confusion matrices
(see results/results_gesture_snn_temporal_multibeta_3341837/) reflect a
genuine property of the gesture data, or a code/architecture bias toward
always predicting a *fixed output slot* (e.g. output neuron index 2, i.e.
class label "3") regardless of what gesture that slot actually holds.

Original confusion matrices (results_gesture_snn_temporal_multibeta_3341837):
  - baseline   (SNNGestureClassifierSynaptic, rate decode, "known best"):
    most classes' test examples pile up in the predicted="3" column.
  - multibeta  (SNNGestureClassifierMultiBeta, rate decode):
    most classes' test examples pile up in the predicted="1" column.

Test: relabel the raw gesture data by swapping which physical gesture is
called "3" vs "5" (for baseline) or "1" vs "6" (for multibeta) in BOTH the
train and test split, retrain from scratch, and re-evaluate.
  - If the bias is about the *label slot* (a code/architecture bias -- e.g.
    output neuron 2 has an easier-to-hit decision boundary irrespective of
    what's trained into it), swapping should make the SAME slot ("3", or "1")
    stay the over-predicted column even though it now holds different gesture
    data -- i.e. the bias follows the label, not the underlying gesture.
  - If the bias is about the *gesture itself* (e.g. gesture "3"'s
    accelerometer trace is simply harder to distinguish / more central in
    feature space than the others), swapping should make the over-prediction
    follow the relabeled data -- the column that was "3" and is now "5" (or
    vice versa) should carry the bias, not the "3" slot itself.

Both baseline and multibeta configs are retrained TWICE: once on original
labels (reproducing the known confusion matrix as a sanity check) and once on
swapped labels, using the same hyperparameters/seed as
run_gesture_snn_temporal_multibeta.py (the script that produced the original
matrices) via snn_encoder_legacy.py (a copy of src/snn_encoder.py as it stood
at that commit -- current HEAD's snn_encoder.py has since been refactored to
drop SNNGestureClassifierSynaptic/single-MultiBeta in favor of the
MultiBeta3/4/5 family, so this experiment pins the exact original code instead
of trying to approximate it with the newer classes).

Usage:
    python scripts/run_gesture_label_swap_experiment.py [--outdir DIR] [--quick] [--device cuda|cpu]
"""
import argparse
import copy
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

import snn_encoder_legacy as se
import uwave_data as data

NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = se.DEFAULT_HIDDEN_SIZE

# Same "best known" hyperparameters as run_gesture_snn_temporal_multibeta.py
# (results_gesture_snn_comparison_3325060/hparam_sweep/results.json "best" block).
SYNAPTIC_ALPHA, SYNAPTIC_BETA, SYNAPTIC_THRESHOLD = 0.9, 0.5, 0.25
MULTIBETA_FAST, MULTIBETA_SLOW, MULTIBETA_OUT, MULTIBETA_THRESHOLD = 0.5, 0.95, 0.5, 0.25

# (config_name, swap_pair, model class/kwargs, label used for plot titles)
SWAP_EXPERIMENTS = {
    "baseline": {"swap": ("3", "5"), "label": "Synaptic, rate decode (known best)"},
    "multibeta": {"swap": ("1", "6"), "label": "MultiBeta, rate decode"},
}


def swap_labels(by_class, label_a, label_b):
    """{label: [examples]} -> new dict with label_a's and label_b's example
    lists swapped, all other labels untouched. Does not mutate the input."""
    swapped = dict(by_class)
    swapped[label_a], swapped[label_b] = by_class[label_b], by_class[label_a]
    return swapped


def build_model_config(name):
    delta_encode_fn = partial(se.encode_delta_spike_trains, delta_threshold=se.DEFAULT_DELTA_THRESHOLD)
    if name == "baseline":
        model_kwargs = {"hidden_size": HIDDEN_SIZE, "alpha": SYNAPTIC_ALPHA,
                         "beta": SYNAPTIC_BETA, "threshold": SYNAPTIC_THRESHOLD}
        return delta_encode_fn, se.SNNGestureClassifierSynaptic, model_kwargs, SF.ce_rate_loss()
    elif name == "multibeta":
        model_kwargs = {"hidden_size": HIDDEN_SIZE, "beta_fast": MULTIBETA_FAST,
                         "beta_slow": MULTIBETA_SLOW, "beta_out": MULTIBETA_OUT,
                         "threshold": MULTIBETA_THRESHOLD}
        return delta_encode_fn, se.SNNGestureClassifierMultiBeta, model_kwargs, SF.ce_rate_loss()
    raise ValueError(name)


def run_one(name, train_by_class, test_by_class, device, num_epochs):
    encode_fn, model_cls, model_kwargs, loss_fn = build_model_config(name)
    t0 = time.time()
    model = se.train_snn(train_by_class, encode_fn, model_cls=model_cls, model_kwargs=model_kwargs,
                          num_epochs=num_epochs, batch_size=BATCH_SIZE, device=device, loss_fn=loss_fn)
    train_s = time.time() - t0

    t0 = time.time()
    accuracy, per_class, confusion, margins = se.evaluate_snn(
        model, test_by_class, encode_fn, batch_size=BATCH_SIZE, device=device,
        return_per_class=True, return_confusion=True, decode="rate")
    eval_s = time.time() - t0

    return {"accuracy": accuracy, "per_class": per_class, "confusion": confusion,
            "train_s": train_s, "eval_s": eval_s}


def plot_confusion_matrix(outdir, confusion, title, subtitle, filename):
    classes = data.CLASSES
    matrix = np.array([[confusion[t][p] for p in classes] for t in classes])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.suptitle(title, y=0.98)
    plt.title(subtitle, fontsize=9)

    vmax = matrix.max()
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = matrix[i, j]
            color = "white" if count > vmax / 2 else "black"
            plt.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_gesture_label_swap")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for a smoke test.")
    parser.add_argument("--device", default=None,
                         help="cuda or cpu. Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    num_epochs = 3 if args.quick else NUM_EPOCHS

    print(f"device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_orig = data.load_split("TRAIN")
    test_orig = data.load_split("TEST")

    all_results = {}
    wall0 = time.time()
    for name, spec in SWAP_EXPERIMENTS.items():
        label_a, label_b = spec["swap"]
        title_label = spec["label"]

        for variant in ("original", "swapped"):
            torch.manual_seed(42)
            if variant == "original":
                train_by_class, test_by_class = train_orig, test_orig
            else:
                train_by_class = swap_labels(train_orig, label_a, label_b)
                test_by_class = swap_labels(test_orig, label_a, label_b)

            key = f"{name}_{variant}"
            print(f"\n[{key}] training ({title_label}, {variant} labels"
                  f"{'' if variant == 'original' else f', {label_a}<->{label_b} swapped'})...",
                  flush=True)
            res = run_one(name, train_by_class, test_by_class, device, num_epochs)
            all_results[key] = res
            print(f"[{key}] -> acc={res['accuracy']:.4f} "
                  f"(train {res['train_s']:.1f}s, eval {res['eval_s']:.1f}s)", flush=True)

            subtitle = title_label if variant == "original" else \
                f"{title_label}, labels {label_a}<->{label_b} swapped"
            plot_confusion_matrix(
                args.outdir, res["confusion"],
                "UWaveGestureLibrary: confusion matrix (label-swap check)",
                subtitle, f"confusion_matrix_{key}.png")

    wall_s = time.time() - wall0
    print(f"\nAll configs done in {wall_s:.1f}s wall.", flush=True)

    # Side-by-side accuracy comparison, original vs swapped, per config.
    plt.figure(figsize=(8, 5))
    names = list(SWAP_EXPERIMENTS.keys())
    x = np.arange(len(names))
    width = 0.35
    orig_accs = [all_results[f"{n}_original"]["accuracy"] * 100 for n in names]
    swap_accs = [all_results[f"{n}_swapped"]["accuracy"] * 100 for n in names]
    plt.bar(x - width / 2, orig_accs, width, label="original labels", color="#4C72B0")
    plt.bar(x + width / 2, swap_accs, width, label="swapped labels", color="#DD8452")
    plt.xticks(x, [f"{n}\n({'<->'.join(SWAP_EXPERIMENTS[n]['swap'])} swap)" for n in names])
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary: accuracy before/after label swap")
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "accuracy_original_vs_swapped.png"), dpi=150)
    plt.close()

    summary = {
        "config": {"hidden_size": HIDDEN_SIZE, "num_epochs": num_epochs, "batch_size": BATCH_SIZE,
                    "synaptic_alpha": SYNAPTIC_ALPHA, "synaptic_beta": SYNAPTIC_BETA,
                    "synaptic_threshold": SYNAPTIC_THRESHOLD,
                    "multibeta_fast": MULTIBETA_FAST, "multibeta_slow": MULTIBETA_SLOW,
                    "multibeta_out": MULTIBETA_OUT, "multibeta_threshold": MULTIBETA_THRESHOLD,
                    "device": device, "quick": args.quick, "wall_seconds": wall_s,
                    "swaps": {n: spec["swap"] for n, spec in SWAP_EXPERIMENTS.items()}},
        "accuracy": {k: v["accuracy"] for k, v in all_results.items()},
        "per_class": {k: v["per_class"] for k, v in all_results.items()},
        "confusion": {k: v["confusion"] for k, v in all_results.items()},
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plots + results.json to {args.outdir}/", flush=True)
    print("\n===== SUMMARY =====")
    for name in names:
        label_a, label_b = SWAP_EXPERIMENTS[name]["swap"]
        orig = all_results[f"{name}_original"]
        swap = all_results[f"{name}_swapped"]
        # Column totals: how many test examples were predicted into label_a / label_b,
        # before and after the swap -- the direct read on "did the bias follow the slot?"
        orig_col_a = sum(orig["confusion"][t][label_a] for t in data.CLASSES)
        orig_col_b = sum(orig["confusion"][t][label_b] for t in data.CLASSES)
        swap_col_a = sum(swap["confusion"][t][label_a] for t in data.CLASSES)
        swap_col_b = sum(swap["confusion"][t][label_b] for t in data.CLASSES)
        print(f"  [{name}] accuracy original={orig['accuracy']:.4f} swapped={swap['accuracy']:.4f}")
        print(f"    predicted-as-{label_a} count: original={orig_col_a} swapped={swap_col_a}")
        print(f"    predicted-as-{label_b} count: original={orig_col_b} swapped={swap_col_b}")


if __name__ == "__main__":
    main()
