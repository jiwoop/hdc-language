"""Standalone plotter for results_3305116/results.json.

Regenerates the two plots of interest from the saved JSON snapshot (no
re-running experiments), with explicit (x, y) coordinate labels annotated at
each point -- the original run_experiments.py plots leave points unlabeled,
which makes reading exact values off a saved PNG (e.g. for a writeup) require
guessing from gridlines.

Usage:
    python plot_results.py [results.json] [--outdir DIR]

Defaults to results.json / this script's own directory, so it can be run
as-is from inside results_3305116/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _annotate(ax, xs, ys, fmt="{:.3f}", dy=8, fontsize=7):
    """Label every point with its (rounded) y-value, offset above the marker."""
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, dy), ha='center', fontsize=fontsize)


def plot_stratified(outdir, results):
    unstrat = {int(k): v for k, v in results["accuracy_unstratified"].items()}
    strat = {int(k): v for k, v in results["accuracy_stratified"].items()}
    std_u = {int(k): v for k, v in results["accuracy_unstratified_std"].items()}
    std_s = {int(k): v for k, v in results["accuracy_stratified_std"].items()}
    n_runs = len(next(iter(results["accuracy_stratified_runs"].values())))
    K = results["config"]["K"]

    xs = sorted(unstrat)
    ys_u = [unstrat[x] * 100 for x in xs]
    ys_s = [strat[x] * 100 for x in xs]
    yerr_u = [std_u[x] * 100 for x in xs]
    yerr_s = [std_s[x] * 100 for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys_u, yerr=yerr_u, marker='o', capsize=3, label="unstratified")
    ax.errorbar(xs, ys_s, yerr=yerr_s, marker='s', capsize=3, label="stratified")
    for x, y, e in zip(xs, ys_u, yerr_u):
        ax.annotate(f"{y:.2f}±{e:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -16), ha='center', fontsize=7)
    for x, y, e in zip(xs, ys_s, yerr_s):
        ax.annotate(f"{y:.2f}±{e:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel("Vector dimension [bit]"); ax.set_ylabel("Inference accuracy [%]")
    ax.set_title(f"Landmark sampling strategy vs. accuracy (k={K}, n={n_runs} runs)")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_dimension_stratified_comparison.png"), dpi=150)
    plt.close(fig)


def plot_num_landmark_fine(outdir, results):
    acc = {int(k): v for k, v in results["accuracy_by_num_landmark_fine"].items()}
    acc_std = {int(k): v for k, v in results["accuracy_by_num_landmark_fine_std"].items()}
    train_t = {int(k): v for k, v in results["train_time_by_num_landmark_fine"].items()}
    eval_t = {int(k): v for k, v in results["eval_time_by_num_landmark_fine"].items()}
    n_runs = len(next(iter(results["accuracy_by_num_landmark_fine_runs"].values())))
    D_BASELINE = results["config"]["bit_budgets"][0]
    K = results["config"]["K"]
    title_suffix = f" [fine sweep, n={n_runs} runs, mean±std]"

    xs = sorted(acc)
    ys = [acc[x] * 100 for x in xs]
    yerr = [acc_std[x] * 100 for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3)
    _annotate(ax, xs, ys, fmt="{:.2f}", dy=10)
    ax.set_xlabel("Number of landmarks"); ax.set_ylabel("Inference accuracy [%]")
    ax.set_title(f"Accuracy vs. landmark count (D={D_BASELINE}, k={K}){title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_num_landmark_fine.png"), dpi=150)
    plt.close(fig)

    xs_t = sorted(train_t)
    ys_train = [train_t[x] for x in xs_t]
    ys_eval = [eval_t[x] for x in xs_t]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_t, ys_train, marker='o', label="training time")
    ax.plot(xs_t, ys_eval, marker='s', label="evaluation time")
    _annotate(ax, xs_t, ys_train, fmt="{:.0f}s", dy=8)
    _annotate(ax, xs_t, ys_eval, fmt="{:.0f}s", dy=-14)
    ax.set_xlabel("Number of landmarks"); ax.set_ylabel("Time [s]")
    ax.set_title(f"Training vs. evaluation time by landmark count (D={D_BASELINE}, k={K}){title_suffix}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "time_vs_num_landmark_fine.png"), dpi=150)
    plt.close(fig)


PLOTS = [
    plot_stratified,
    plot_num_landmark_fine,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", nargs="?", default=os.path.join(HERE, "results.json"))
    parser.add_argument("--outdir", default=HERE)
    args = parser.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    for plot_fn in PLOTS:
        plot_fn(args.outdir, results)
        print(f"wrote {plot_fn.__name__}")

    print(f"\nAll plots written to {args.outdir}/")


if __name__ == "__main__":
    main()
