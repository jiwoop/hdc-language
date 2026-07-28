"""Combines run_gesture_experiments.py's results.json (permutation + Nystrom+GAK,
swept over vector dimension D), run_gesture_snn_experiments.py's results.json
(SNN, swept over hidden_size), and/or run_gesture_snn_method_comparison.py's
results.json (SNN, 3 fixed-hyperparameter methods: rate_leaky, delta_leaky,
delta_synaptic) into two plots + one table.

None of these share an x-axis (D vs. hidden_size vs. "no sweep, one tuned point
per method" -- see run_gesture_snn_experiments.py's and
run_gesture_snn_method_comparison.py's module docstrings). Two views are
produced: a best-single-point-per-method bar chart (accuracy_comparison_gesture.png,
includes all methods from all three sources) and a full-sweep-curve overlay with a
twinned x-axis, D on bottom / hidden_size on top (accuracy_sweeps_comparison_gesture.png,
sweep methods only -- the 3 method-comparison methods are single points, not
sweeps, so they never appear on this second plot). Only accuracy is compared,
never wall-clock time, since permutation/GAK run on CPU and the SNN methods run
on GPU (see hdc_gesture_snn.slurm's rationale).

Usage:
    python compare_gesture_methods.py --hdc-dir results_gesture \\
        [--snn-dir results_gesture_snn_<jobid>] \\
        [--snn-methods-dir results_gesture_snn_methods_<jobid>] \\
        [--outdir DIR]

At least one of --snn-dir / --snn-methods-dir must be given.
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


def load_snn_methods(snn_methods_dir):
    """results_gesture_snn_methods_<jobid>/results.json (from
    run_gesture_snn_method_comparison.py) -> {"rate_leaky": {"param": "-",
    "param_name": "-", "accuracy": acc}, "delta_leaky": {...}, "delta_synaptic": {...}}.
    Each method here is a single tuned point, not a sweep, so it bypasses best_points."""
    with open(os.path.join(snn_methods_dir, "results.json")) as f:
        summary = json.load(f)

    return {
        method: {"param": "-", "param_name": "-", "accuracy": acc}
        for method, acc in summary["accuracy_by_method"].items()
    }


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
    "rate_leaky": "SNN: Rate + Rate decoding + Leaky",
    "delta_leaky": "SNN: Delta + Rate decoding + Leaky",
    "delta_synaptic": "SNN: Delta + Rate decoding + Synaptic",
}

COLORS = {
    "permutation": "#4C72B0",
    "gak": "#DD8452",
    "snn": "#55A868",
    "rate_leaky": "#8172B2",
    "delta_leaky": "#C44E52",
    "delta_synaptic": "#937860",
}


def plot_comparison(outdir, best):
    methods = list(best)
    accs = [best[m]["accuracy"] * 100 for m in methods]
    labels = [LABELS[m] for m in methods]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, accs, color=[COLORS[m] for m in methods])
    for bar, m in zip(bars, methods):
        param = best[m]
        if param["param_name"] != "-":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{param['param_name']}={param['param']}", ha="center", fontsize=9)
    plt.ylabel("Best inference accuracy [%]")
    plt.title("UWaveGestureLibrary: best accuracy by method")
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison_gesture.png"), dpi=150)
    plt.close()


def plot_sweep_curves(outdir, hdc_sweeps, snn_sweep):
    """Sweep-curve methods' full accuracy sweep curves on one plot: permutation and
    Nystrom+GAK vs. dimension D on the bottom x-axis, SNN vs. hidden_size on a
    twinned top x-axis (both log-scaled) -- distinct capacity knobs, so they can't
    share one axis, but the shared y-axis (accuracy) still makes them comparable
    at a glance. snn_sweep is optional (None if --snn-dir wasn't given) -- the 3
    method-comparison methods never appear here, since they're single tuned points,
    not sweeps; see plot_comparison for those."""
    fig, ax_bottom = plt.subplots(figsize=(8, 5))

    xs_perm = sorted(hdc_sweeps["permutation"])
    xs_gak = sorted(hdc_sweeps["gak"])

    l1, = ax_bottom.plot(xs_perm, [hdc_sweeps["permutation"][x] * 100 for x in xs_perm],
                          marker='o', color=COLORS["permutation"], label=LABELS["permutation"])
    l2, = ax_bottom.plot(xs_gak, [hdc_sweeps["gak"][x] * 100 for x in xs_gak],
                          marker='s', color=COLORS["gak"], label=LABELS["gak"])
    handles = [l1, l2]

    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel("Vector dimension D [bit] (permutation, Nystrom+GAK)")
    ax_bottom.set_ylabel("Inference accuracy [%]")
    ax_bottom.set_ylim(0, 100)
    ax_bottom.set_title("UWaveGestureLibrary: accuracy sweeps")
    ax_bottom.grid(True, which='both', alpha=0.3)

    if snn_sweep is not None:
        ax_top = ax_bottom.twiny()
        xs_snn = sorted(snn_sweep["snn"])
        l3, = ax_top.plot(xs_snn, [snn_sweep["snn"][x] * 100 for x in xs_snn],
                           marker='^', color=COLORS["snn"], label=LABELS["snn"])
        ax_top.set_xscale('log')
        ax_top.set_xlabel("Hidden layer size (SNN)")
        handles.append(l3)

    ax_bottom.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_sweeps_comparison_gesture.png"), dpi=150)
    plt.close(fig)


def print_table(best):
    print("\n===== Best accuracy by method =====")
    print(f"{'Method':<38} {'Best param':<18} {'Accuracy':>10}")
    for m, res in best.items():
        param_str = "-" if res["param_name"] == "-" else f"{res['param_name']}={res['param']}"
        print(f"{LABELS[m]:<38} {param_str:<18} {res['accuracy'] * 100:>9.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdc-dir", default="results_gesture",
                        help="Output dir from run_gesture_experiments.py (permutation + Nystrom+GAK).")
    parser.add_argument("--snn-dir", default=None,
                        help="Optional: output dir from run_gesture_snn_experiments.py (SNN hidden_size sweep).")
    parser.add_argument("--snn-methods-dir", default=None,
                        help="Optional: output dir from run_gesture_snn_method_comparison.py "
                             "(SNN rate/delta/synaptic method comparison).")
    parser.add_argument("--outdir", default="results_gesture_comparison")
    args = parser.parse_args()

    if args.snn_dir is None and args.snn_methods_dir is None:
        parser.error("at least one of --snn-dir / --snn-methods-dir must be given")

    os.makedirs(args.outdir, exist_ok=True)

    hdc_sweeps = load_hdc_sweeps(args.hdc_dir)
    snn_sweep = load_snn_sweep(args.snn_dir) if args.snn_dir else None

    best = {}
    best.update(best_points(hdc_sweeps))
    if snn_sweep is not None:
        best.update(best_points(snn_sweep))
    if args.snn_methods_dir:
        best.update(load_snn_methods(args.snn_methods_dir))

    plot_comparison(args.outdir, best)
    plot_sweep_curves(args.outdir, hdc_sweeps, snn_sweep)
    print_table(best)

    with open(os.path.join(args.outdir, "comparison.json"), "w") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"\nWrote plot + comparison.json to {args.outdir}/")


if __name__ == "__main__":
    main()
