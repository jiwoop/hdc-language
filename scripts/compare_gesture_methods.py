"""Combines run_gesture_experiments.py's results.json (permutation + Nystrom+GAK,
swept over vector dimension D) with run_gesture_snn_experiments.py's results.json
(SNN, swept over hidden_size) into two plots + one table.

The three methods don't share an x-axis (D vs. hidden_size are different capacity
knobs -- see run_gesture_snn_experiments.py's module docstring). Two views are
produced: a best-single-point-per-method bar chart (accuracy_comparison_gesture.png)
and a full-sweep-curve overlay with a twinned x-axis, D on bottom / hidden_size on
top (accuracy_sweeps_comparison_gesture.png). Only accuracy is compared, never
wall-clock time, since permutation/GAK run on CPU and the SNN runs on GPU (see
hdc_gesture_snn.slurm's rationale).

Usage:
    python compare_gesture_methods.py --hdc-dir results_gesture --snn-dir results_gesture_snn_<jobid> [--outdir DIR]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt


def load_hdc_sweeps(hdc_dir):
    """results_gesture/results.json -> {"permutation": {D: acc, ...}, "gak": {D: acc, ...}}."""
    with open(os.path.join(hdc_dir, "results.json")) as f:
        summary = json.load(f)

    by_dim_perm = {int(d): acc for d, acc in summary["accuracy_by_dimension_permutation"].items()}
    by_dim_gak = {int(d): acc for d, acc in summary["accuracy_by_dimension_gak"].items()}
    return {"permutation": by_dim_perm, "gak": by_dim_gak}


def load_snn_sweep(snn_dir):
    """results_gesture_snn_<jobid>/results.json -> {"snn": {hidden_size: acc, ...}}."""
    with open(os.path.join(snn_dir, "results.json")) as f:
        summary = json.load(f)

    by_hidden = {int(h): acc for h, acc in summary["accuracy_by_hidden_size"].items()}
    return {"snn": by_hidden}


PARAM_NAME = {"permutation": "D", "gak": "D", "snn": "hidden_size"}


def best_points(sweeps):
    """{"method": {param: acc, ...}, ...} -> {"method": {"param":, "param_name":, "accuracy":}, ...}."""
    best = {}
    for method, by_param in sweeps.items():
        best_param = max(by_param, key=by_param.get)
        best[method] = {"param": best_param, "param_name": PARAM_NAME[method], "accuracy": by_param[best_param]}
    return best


LABELS = {
    "permutation": "Multi-channel permutation",
    "gak": "Nystrom+GAK",
    "snn": "SNN (snnTorch, rate-coded)",
}


def plot_comparison(outdir, best):
    methods = list(best)
    accs = [best[m]["accuracy"] * 100 for m in methods]
    labels = [LABELS[m] for m in methods]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, accs, color=["#4C72B0", "#DD8452", "#55A868"][:len(methods)])
    for bar, m in zip(bars, methods):
        param = best[m]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{param['param_name']}={param['param']}", ha="center", fontsize=9)
    plt.ylabel("Best inference accuracy [%]")
    plt.title("UWaveGestureLibrary: best accuracy by method")
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison_gesture.png"), dpi=150)
    plt.close()


def plot_sweep_curves(outdir, hdc_sweeps, snn_sweep):
    """All three methods' full accuracy sweep curves on one plot: permutation and
    Nystrom+GAK vs. dimension D on the bottom x-axis, SNN vs. hidden_size on a
    twinned top x-axis (both log-scaled) -- distinct capacity knobs, so they can't
    share one axis, but the shared y-axis (accuracy) still makes them comparable
    at a glance."""
    fig, ax_bottom = plt.subplots(figsize=(8, 5))
    ax_top = ax_bottom.twiny()

    xs_perm = sorted(hdc_sweeps["permutation"])
    xs_gak = sorted(hdc_sweeps["gak"])
    xs_snn = sorted(snn_sweep["snn"])

    l1, = ax_bottom.plot(xs_perm, [hdc_sweeps["permutation"][x] * 100 for x in xs_perm],
                          marker='o', color="#4C72B0", label=LABELS["permutation"])
    l2, = ax_bottom.plot(xs_gak, [hdc_sweeps["gak"][x] * 100 for x in xs_gak],
                          marker='s', color="#DD8452", label=LABELS["gak"])
    l3, = ax_top.plot(xs_snn, [snn_sweep["snn"][x] * 100 for x in xs_snn],
                       marker='^', color="#55A868", label=LABELS["snn"])

    ax_bottom.set_xscale('log')
    ax_top.set_xscale('log')
    ax_bottom.set_xlabel("Vector dimension D [bit] (permutation, Nystrom+GAK)")
    ax_top.set_xlabel("Hidden layer size (SNN)")
    ax_bottom.set_ylabel("Inference accuracy [%]")
    ax_bottom.set_ylim(0, 100)
    ax_bottom.set_title("UWaveGestureLibrary: accuracy sweeps, all methods")
    ax_bottom.grid(True, which='both', alpha=0.3)
    ax_bottom.legend(handles=[l1, l2, l3], loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_sweeps_comparison_gesture.png"), dpi=150)
    plt.close(fig)


def print_table(best):
    print("\n===== Best accuracy by method =====")
    print(f"{'Method':<28} {'Best param':<18} {'Accuracy':>10}")
    for m, res in best.items():
        param_str = f"{res['param_name']}={res['param']}"
        print(f"{LABELS[m]:<28} {param_str:<18} {res['accuracy'] * 100:>9.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdc-dir", default="results_gesture",
                        help="Output dir from run_gesture_experiments.py (permutation + Nystrom+GAK).")
    parser.add_argument("--snn-dir", required=True,
                        help="Output dir from run_gesture_snn_experiments.py (SNN).")
    parser.add_argument("--outdir", default="results_gesture_comparison")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hdc_sweeps = load_hdc_sweeps(args.hdc_dir)
    snn_sweep = load_snn_sweep(args.snn_dir)

    best = {}
    best.update(best_points(hdc_sweeps))
    best.update(best_points(snn_sweep))

    plot_comparison(args.outdir, best)
    plot_sweep_curves(args.outdir, hdc_sweeps, snn_sweep)
    print_table(best)

    with open(os.path.join(args.outdir, "comparison.json"), "w") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"\nWrote plot + comparison.json to {args.outdir}/")


if __name__ == "__main__":
    main()
