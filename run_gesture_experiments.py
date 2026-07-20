"""Batch driver for the UWaveGestureLibrary experiment: multi-channel permutation
HDC baseline vs. Nystrom+GAK HDC, for TACC Lonestar6.

Headline result: accuracy vs. vector dimension D, both methods on one plot.
Secondary: Nystrom+GAK accuracy vs. number of landmarks, at a fixed D.
Tertiary: permutation baseline's num_levels quantization-resolution sensitivity.

Why CPU (not GPU): the permutation baseline's dominant cost is a pure-Python
per-timestep loop (permutation_encoder.encode_example_to_hv), and the Nystrom+GAK
side's dominant cost is tslearn's GAK kernel evaluation (itself already a compiled/
numba-backed routine, but called once per example-landmark pair in a Python loop
by cdist_gak internally) -- neither is dense-matmul-bound, so there is no GPU code
in this experiment either, matching hdc_nystrom.slurm's rationale. The sweep's
points are independent, so we fan out across a multiprocessing.Pool on one LS6
128-core node, one BLAS thread per worker.

GAK is expensive (O(seriesLength^2) per pair; empirically ~1.5ms/pair on this
data), and its cost is set by (num_examples, num_landmark) -- NOT by the
hypervector dimension D. So for Nystrom+GAK, the GAK kernel matrices (train-by-
class and test-by-class, against a given landmark set) are computed ONCE per
landmark-count and then reused across every D in the dimension sweep within the
same worker call (gak_encoder.build_gak_landmark_kernels +
train_class_vectors_gak_from_kernels), rather than recomputing GAK per (D,
landmark) pair -- an (len(DIMENSIONS))x reduction in GAK calls for the headline
sweep.

Usage:
    python run_gesture_experiments.py [--workers N] [--outdir DIR] [--quick]

--quick runs a reduced grid (few D points, small landmark count, few num_levels)
for a smoke test on a login node before committing a full run.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt

import gak_encoder as ge
import permutation_encoder as pe
import uwave_data as data

# ---------------------------------------------------------------------------
# Experiment grids.
# ---------------------------------------------------------------------------
# Keeps {64, 128, 512, 1024} for cross-comparability with the language task's own
# dimension sweep (run_experiments.py's DIMENSIONS), extended to 2048/4096/8192/
# 10000 since binary-HDC time-series literature (record-based/spatiotemporal HDC,
# e.g. Kleyko & Osipov; Rahimi et al.'s EMG-gesture HDC; Moin et al. 2021 Nature
# Electronics hand-gesture HDC) commonly uses D=10000 as the standard hypervector
# width, with accuracy typically plateauing in the low thousands -- this range
# lets the sweep actually reach that plateau.
DIMENSIONS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
D_BASELINE = 1024  # cross-comparable point used for the secondary/tertiary sweeps

NUM_LANDMARK_VALUES = [20, 40, 80, 160, 320]  # secondary sweep, GAK's own core knob
GAK_NUM_LANDMARK_FOR_DIM_SWEEP = 80           # fixed landmark count for the headline D-sweep

NUM_LEVELS_VALUES = [20, 50, 100, 200]        # tertiary sweep, baseline's quantization knob
N_GRAM_SIZE = pe.DEFAULT_N_GRAM_SIZE          # fixed n-gram size throughout


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level so Pool can pickle them).
# ---------------------------------------------------------------------------
def w_permutation_point(job):
    """(dimension, num_levels, n_gram_size) -> result dict."""
    dimension, num_levels, n_gram_size = job
    test_by_class = data.load_split("TEST")

    t0 = time.time()
    class_hvs, channel_memory, level_memory, lo, hi = pe.train_class_vectors_permutation(
        dimension, num_levels=num_levels, n_gram_size=n_gram_size)
    train_s = time.time() - t0

    t0 = time.time()
    acc, per_class = pe.evaluate_permutation(
        class_hvs, channel_memory, level_memory, lo, hi, num_levels, n_gram_size,
        dimension, test_by_class, return_per_class=True)
    eval_s = time.time() - t0

    return {"method": "permutation", "dimension": dimension, "num_levels": num_levels,
            "n_gram_size": n_gram_size, "accuracy": acc, "per_class": per_class,
            "train_s": train_s, "eval_s": eval_s}


def w_gak_dimension_sweep(job):
    """(dimensions, num_landmark) -> list of result dicts, one per dimension.

    Computes the expensive GAK kernel matrices ONCE for this num_landmark, then
    loops the cheap per-dimension eigendecomposition/random-projection/eval step
    over every dimension in `dimensions` -- see module docstring.
    """
    dimensions, num_landmark = job

    t0 = time.time()
    kernels = ge.build_gak_landmark_kernels(num_landmark=num_landmark)
    kernel_s = time.time() - t0

    results = []
    for dimension in dimensions:
        t0 = time.time()
        class_hvs, encoding_matrix = ge.train_class_vectors_gak_from_kernels(kernels, dimension)
        train_s = time.time() - t0

        t0 = time.time()
        acc, per_class = ge.evaluate_gak_from_kernels(
            class_hvs, encoding_matrix, kernels, dimension, return_per_class=True)
        eval_s = time.time() - t0

        results.append({
            "method": "gak", "dimension": dimension, "num_landmark": num_landmark,
            "accuracy": acc, "per_class": per_class,
            "train_s": train_s, "eval_s": eval_s, "kernel_s": kernel_s,
        })
    return results


def w_gak_landmark_point(job):
    """(num_landmark, dimension) -> result dict. Used for the num-landmark sweep
    (one landmark count -> one kernel build -> one dimension, not the whole grid)."""
    num_landmark, dimension = job

    t0 = time.time()
    kernels = ge.build_gak_landmark_kernels(num_landmark=num_landmark)
    kernel_s = time.time() - t0

    t0 = time.time()
    class_hvs, encoding_matrix = ge.train_class_vectors_gak_from_kernels(kernels, dimension)
    train_s = time.time() - t0

    t0 = time.time()
    acc, per_class = ge.evaluate_gak_from_kernels(
        class_hvs, encoding_matrix, kernels, dimension, return_per_class=True)
    eval_s = time.time() - t0

    return {"method": "gak", "dimension": dimension, "num_landmark": num_landmark,
            "accuracy": acc, "per_class": per_class,
            "train_s": train_s, "eval_s": eval_s, "kernel_s": kernel_s}


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def plot_dimension(outdir, by_dim_perm, by_dim_gak):
    """Headline plot: accuracy vs. D, both methods overlaid."""
    xs_perm = sorted(by_dim_perm)
    xs_gak = sorted(by_dim_gak)
    plt.figure(figsize=(8, 5))
    plt.plot(xs_perm, [by_dim_perm[x] * 100 for x in xs_perm], marker='o',
             label="Multi-channel permutation (baseline)")
    plt.plot(xs_gak, [by_dim_gak[x] * 100 for x in xs_gak], marker='s',
             label=f"Nystrom+GAK ({GAK_NUM_LANDMARK_FOR_DIM_SWEEP} landmarks)")
    plt.xscale('log')
    plt.xlabel("Vector dimension [bit]")
    plt.ylabel("Inference accuracy [%]")
    plt.title("UWaveGestureLibrary accuracy vs. vector dimension")
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_dimension_gesture.png"), dpi=150)
    plt.close()


def plot_num_landmark(outdir, acc, kernel_t, train_t, eval_t):
    xs = sorted(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [acc[x] * 100 for x in xs], marker='o')
    plt.xlabel("Number of landmarks")
    plt.ylabel("Inference accuracy [%]")
    plt.title(f"Nystrom+GAK accuracy vs. landmark count (D={D_BASELINE})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_num_landmark_gesture.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [kernel_t[x] for x in xs], marker='^', label="GAK kernel build time")
    plt.plot(xs, [train_t[x] for x in xs], marker='o', label="training time")
    plt.plot(xs, [eval_t[x] for x in xs], marker='s', label="evaluation time")
    plt.xlabel("Number of landmarks")
    plt.ylabel("Time [s]")
    plt.title(f"Nystrom+GAK time vs. landmark count (D={D_BASELINE})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_vs_num_landmark_gesture.png"), dpi=150)
    plt.close()


def plot_num_levels(outdir, acc):
    xs = sorted(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [acc[x] * 100 for x in xs], marker='o')
    plt.xlabel("Number of quantization levels")
    plt.ylabel("Inference accuracy [%]")
    plt.title(f"Permutation baseline accuracy vs. quantization resolution (D={D_BASELINE})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_num_levels_gesture.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or None,
                        help="Pool size. Defaults to SLURM_CPUS_ON_NODE, else all cores.")
    parser.add_argument("--outdir", default="results_gesture")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for a smoke test.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dims = [128, 1024] if args.quick else DIMENSIONS
    num_landmark_values = [10, 20] if args.quick else NUM_LANDMARK_VALUES
    gak_dim_sweep_landmark = 10 if args.quick else GAK_NUM_LANDMARK_FOR_DIM_SWEEP
    num_levels_values = [20, 100] if args.quick else NUM_LEVELS_VALUES

    jobs = []  # (section, worker_fn_name, job_args)

    # Headline: permutation baseline accuracy vs. D.
    for d in dims:
        jobs.append(("dim_perm", "w_permutation_point", (d, pe.DEFAULT_NUM_LEVELS, N_GRAM_SIZE)))

    # Headline: Nystrom+GAK accuracy vs. D, one kernel build shared across all `dims`.
    jobs.append(("dim_gak", "w_gak_dimension_sweep", (dims, gak_dim_sweep_landmark)))

    # Secondary: Nystrom+GAK accuracy vs. num_landmark, at fixed D_BASELINE.
    for nl in num_landmark_values:
        jobs.append(("nlsweep", "w_gak_landmark_point", (nl, D_BASELINE)))

    # Tertiary: permutation baseline accuracy vs. num_levels, at fixed D_BASELINE.
    for nlev in num_levels_values:
        jobs.append(("levelsweep", "w_permutation_point", (D_BASELINE, nlev, N_GRAM_SIZE)))

    print(f"Dispatching {len(jobs)} experiment points over {args.workers or 'all'} workers...", flush=True)
    wall0 = time.time()

    with Pool(processes=args.workers) as pool:
        results = []
        for section, res in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            results.append((section, res))
            _print_point(section, res)

    print(f"\nAll points done in {time.time() - wall0:.1f}s wall.", flush=True)

    # ---- Demultiplex. ----
    by = {}
    for section, res in results:
        if isinstance(res, list):
            by.setdefault(section, []).extend(res)
        else:
            by.setdefault(section, []).append(res)

    by_dim_perm = {r["dimension"]: r["accuracy"] for r in by.get("dim_perm", [])}
    by_dim_gak = {r["dimension"]: r["accuracy"] for r in by.get("dim_gak", [])}
    per_class_dim_perm = {r["dimension"]: r["per_class"] for r in by.get("dim_perm", [])}
    per_class_dim_gak = {r["dimension"]: r["per_class"] for r in by.get("dim_gak", [])}

    nl_acc = {r["num_landmark"]: r["accuracy"] for r in by.get("nlsweep", [])}
    nl_kernel = {r["num_landmark"]: r["kernel_s"] for r in by.get("nlsweep", [])}
    nl_train = {r["num_landmark"]: r["train_s"] for r in by.get("nlsweep", [])}
    nl_eval = {r["num_landmark"]: r["eval_s"] for r in by.get("nlsweep", [])}

    levels_acc = {r["num_levels"]: r["accuracy"] for r in by.get("levelsweep", [])}

    # ---- Plots. ----
    plot_dimension(args.outdir, by_dim_perm, by_dim_gak)
    plot_num_landmark(args.outdir, nl_acc, nl_kernel, nl_train, nl_eval)
    plot_num_levels(args.outdir, levels_acc)

    # ---- JSON. ----
    summary = {
        "config": {"dimensions": dims, "num_landmark_values": num_landmark_values,
                   "gak_dim_sweep_landmark": gak_dim_sweep_landmark,
                   "num_levels_values": num_levels_values, "n_gram_size": N_GRAM_SIZE,
                   "quick": args.quick, "wall_seconds": time.time() - wall0},
        "accuracy_by_dimension_permutation": by_dim_perm,
        "accuracy_by_dimension_gak": by_dim_gak,
        "per_class_by_dimension_permutation": per_class_dim_perm,
        "per_class_by_dimension_gak": per_class_dim_gak,
        "accuracy_by_num_landmark": nl_acc,
        "kernel_time_by_num_landmark": nl_kernel,
        "train_time_by_num_landmark": nl_train,
        "eval_time_by_num_landmark": nl_eval,
        "accuracy_by_num_levels": levels_acc,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plots + results.json to {args.outdir}/", flush=True)
    _print_report(summary)


def _dispatch(tagged):
    section, fn_name, job_args = tagged
    fns = {"w_permutation_point": w_permutation_point,
           "w_gak_dimension_sweep": w_gak_dimension_sweep,
           "w_gak_landmark_point": w_gak_landmark_point}
    return section, fns[fn_name](job_args)


def _print_point(section, res):
    if isinstance(res, list):
        for r in res:
            print(f"[{section}] D={r['dimension']} num_landmark={r['num_landmark']} "
                  f"-> acc={r['accuracy']:.4f} (train {r['train_s']:.1f}s)", flush=True)
        return
    if res.get("method") == "gak":
        print(f"[{section}] D={res['dimension']} num_landmark={res['num_landmark']} "
              f"-> acc={res['accuracy']:.4f} (kernel {res['kernel_s']:.1f}s, "
              f"train {res['train_s']:.1f}s)", flush=True)
    else:
        print(f"[{section}] D={res['dimension']} num_levels={res['num_levels']} "
              f"-> acc={res['accuracy']:.4f} (train {res['train_s']:.1f}s)", flush=True)


def _print_report(summary):
    print("\n===== SUMMARY =====")
    print("Dimension sweep:")
    for d in sorted(summary["accuracy_by_dimension_permutation"]):
        p = summary["accuracy_by_dimension_permutation"][d]
        g = summary["accuracy_by_dimension_gak"].get(d)
        print(f"  D={d:5d}  Permutation={p:.4f}" + (f"  Nystrom+GAK={g:.4f}" if g else ""))


if __name__ == "__main__":
    main()
