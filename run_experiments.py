"""Batch driver reproducing hdc_nystrom.ipynb end-to-end, for TACC Lonestar6.

Runs the same computations the notebook does -- sanity check, D=1024 baseline
(Nystrom + N-gram), precision sweep, dimension sweep, k-sweep {3,4}, stratified-vs-
unstratified landmark comparison, landmark-count sweep, and a bit-budget x ensemble-size
sweep (does splitting a shrinking total bit budget into more independent small-D encoders
ever beat a single full-width encoder at that budget?) -- plus a round-2 finer-grained
landmark-count sweep -- and writes results to JSON + regenerates every plot PNG.

The bit-budget sweep supersedes the original notebook's single-budget (D=1024-only)
small-D-encoder ensembling section: instead of one fixed budget with ensemble sizes
{1, 2, 4}, it sweeps total bit budget over {1024, 512, 256, 128} and, at each budget,
ensemble size over {1, 3, 4, 5} (1 = single full-width encoder, the baseline for that
budget). This is a strict generalization -- same worker, same question -- so it replaces
rather than duplicates the old section.

Why CPU (not GPU) parallelism: the workload's dominant cost is `nystrom_encoder.kmer_spectrum`,
a pure-Python character loop over ~100K training lines x 10 languages per (D,k) point. That's
CPU-bound Python, not dense linear algebra -- there is no GPU code in this project and a GPU
would sit idle on the bottleneck. The experiments are, however, embarrassingly parallel across
(D, k, num_landmark, strategy, ensemble-size) points, so we fan every independent point out
across a multiprocessing.Pool over a 128-core LS6 compute node.

Each worker pins BLAS to a single thread (see the OMP/MKL/OPENBLAS env in the .slurm script):
with N Python workers each doing their own matmuls, letting BLAS also spawn N threads per
worker oversubscribes the cores and slows everything down. One BLAS thread per worker + one
worker per core is the right split for this many-independent-jobs shape.

Usage:
    python run_experiments.py [--workers N] [--outdir DIR] [--quick]

--quick runs a reduced grid for a smoke test (single small D, no k=4) so you can validate the
pipeline on a login node or a short debug allocation before committing a full run.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt

import hdc
import hdc_train as t
import nystrom_encoder as ne


# ---------------------------------------------------------------------------
# Experiment grids -- mirror the notebook's constants exactly.
# ---------------------------------------------------------------------------
K = 3            # k-mer size (Nystrom side)
N = 3            # n-gram size (baseline side), matched to K
D_SANITY = 200
D_BASELINE = 1024
PRECISIONS = [10, 8, 6]
DIMENSIONS = [1024, 512, 128, 64]
K_VALUES = [3, 4]
K_SWEEP_DTYPE = np.float32
K_SWEEP_BATCH_SIZE = 1000
K_SWEEP_NUM_LANDMARK = 500
STRATIFY_NUM_LANDMARK = 500
NUM_LANDMARK_VALUES = [500, 800, 1000, 2000]
ENSEMBLE_NUM_LANDMARK = 500                   # landmark count for the bit-budget/ensemble sweep
BIT_BUDGETS = [1024, 512, 256, 128]           # total bit-budget sweep
ENSEMBLE_SIZES = [1, 3, 4, 5]                 # 1 = single full-width encoder (baseline)

# --- Round 2 addition ---
NUM_LANDMARK_VALUES_FINE = list(range(200, 2001, 200))  # finer landmark sweep, D=1024, k=3


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level so Pool can pickle them).
# Each returns a small plain-data tuple/dict; heavy arrays stay inside the worker.
# test_data is rebuilt inside each worker rather than pickled across -- it's cheap
# to load and avoids shipping the whole test corpus to every process.
# ---------------------------------------------------------------------------
def _test_data():
    return {language: t.load_test_sentences(language) for language in t.LANGUAGES}


def w_nystrom_point(job):
    """(dimension, k, num_landmark, dtype, batch_size, strategy) -> result dict.

    strategy: 'pooled' (default sampler) or 'stratified'. Covers the baseline point,
    dimension sweep, k-sweep, and stratified comparison with one worker body.
    """
    dimension, k, num_landmark, dtype, batch_size, strategy = job
    sampler = ne.sample_landmarks_stratified if strategy == "stratified" else None
    test_data = _test_data()

    t0 = time.time()
    class_hvs, enc, lf, lsk = ne.train_class_vectors_nystrom(
        dimension, k, num_landmark=num_landmark, dtype=dtype,
        batch_size=batch_size, landmark_sampler=sampler)
    train_s = time.time() - t0

    t0 = time.time()
    acc, per_lang = ne.evaluate_nystrom(
        class_hvs, enc, lf, lsk, k, dimension, test_data,
        dtype=dtype, batch_size=batch_size, return_per_language=True)
    eval_s = time.time() - t0

    return {
        "method": "nystrom", "dimension": dimension, "k": k,
        "num_landmark": num_landmark, "strategy": strategy,
        "accuracy": acc, "per_language": per_lang,
        "train_s": train_s, "eval_s": eval_s,
    }


def w_nystrom_precision(job):
    """(dimension, k, precision) -> result dict. Rebuilds class vectors then evaluates
    at the given block-Hamming precision (class vectors don't depend on precision, but
    each worker is independent so it rebuilds rather than sharing state)."""
    dimension, k, precision = job
    test_data = _test_data()
    class_hvs, enc, lf, lsk = ne.train_class_vectors_nystrom(dimension, k)
    acc = ne.evaluate_nystrom(class_hvs, enc, lf, lsk, k, dimension, test_data,
                              precision=precision)
    return {"method": "nystrom", "dimension": dimension, "k": k,
            "precision": precision, "accuracy": acc}


def w_ngram_point(job):
    """(dimension, n, precision) -> result dict. N-gram baseline; precision=None for exact."""
    dimension, n, precision = job
    test_data = _test_data()
    item_memory = hdc.generate_item_memory(t.ALPHABET, dimension, seed=42)
    class_hvs = t.train_class_vectors(item_memory, n, dimension)
    acc = t.evaluate(item_memory, class_hvs, n, dimension, test_data, precision=precision)
    return {"method": "ngram", "dimension": dimension, "n": n,
            "precision": precision, "accuracy": acc}


def w_ensemble(job):
    """(total_dimension, num_encoders, k, num_landmark) -> result dict."""
    total_dimension, num_encoders, k, num_landmark = job
    test_data = _test_data()
    if num_encoders == 1:
        class_hvs, enc, lf, lsk = ne.train_class_vectors_nystrom(
            total_dimension, k, num_landmark=num_landmark)
        acc = ne.evaluate_nystrom(class_hvs, enc, lf, lsk, k, total_dimension, test_data)
    else:
        encoders = ne.train_class_vectors_nystrom_ensemble(
            total_dimension, num_encoders, k, num_landmark=num_landmark)
        encoder_dimension = total_dimension // num_encoders
        acc = ne.evaluate_nystrom_ensemble(encoders, k, encoder_dimension, test_data)
    return {"total_dimension": total_dimension, "num_encoders": num_encoders,
            "encoder_dimension": total_dimension // num_encoders, "accuracy": acc}


# ---------------------------------------------------------------------------
# Plotting -- one function per notebook figure, driven off the collected results.
# ---------------------------------------------------------------------------
def plot_precision(outdir, by_precision_nys, by_precision_ngram):
    xs = sorted(by_precision_nys, reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [by_precision_nys[x] * 100 for x in xs], marker='o',
             label=f"D={D_BASELINE}, k={K} (Nystrom)")
    plt.plot(xs, [by_precision_ngram[x] * 100 for x in xs], marker='s',
             label=f"D={D_BASELINE}, N={N} (N-gram baseline)")
    plt.gca().invert_xaxis()
    plt.xlabel("Per-block Hamming-distance precision, out of 16-bit blocks")
    plt.ylabel("Inference accuracy [%]")
    plt.title("Language recognition accuracy vs. Hamming-distance precision")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_hamming_precision_comparison.png"), dpi=150)
    plt.close()


def plot_dimension(outdir, by_dim_nys, by_dim_ngram):
    xs = sorted(by_dim_nys, reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [by_dim_nys[x] * 100 for x in xs], marker='o', label=f"k={K} (Nystrom)")
    plt.plot(xs, [by_dim_ngram[x] * 100 for x in xs], marker='s',
             label=f"N={N} (N-gram baseline)")
    plt.gca().invert_xaxis(); plt.xscale('log')
    plt.xlabel("Vector dimension [bit]"); plt.ylabel("Inference accuracy [%]")
    plt.title("Language recognition accuracy vs. vector dimension")
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_dimension_comparison.png"), dpi=150)
    plt.close()


def plot_by_k(outdir, accuracy_by_k):
    plt.figure(figsize=(8, 5))
    for k in sorted(accuracy_by_k):
        xs = sorted(accuracy_by_k[k], reverse=True)
        plt.plot(xs, [accuracy_by_k[k][x] * 100 for x in xs], marker='o', label=f"k={k}")
    plt.gca().invert_xaxis(); plt.xscale('log')
    plt.xlabel("Vector dimension [bit]"); plt.ylabel("Inference accuracy [%]")
    plt.title("Nystrom accuracy vs. vector dimension, by k-mer size")
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_dimension_by_k.png"), dpi=150)
    plt.close()


def plot_stratified(outdir, unstrat, strat, per_lang_u, per_lang_s):
    xs = sorted(unstrat, reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [unstrat[x] * 100 for x in xs], marker='o', label="unstratified (pooled random)")
    plt.plot(xs, [strat[x] * 100 for x in xs], marker='s', label="stratified (equal per language)")
    plt.gca().invert_xaxis(); plt.xscale('log')
    plt.xlabel("Vector dimension [bit]"); plt.ylabel("Inference accuracy [%]")
    plt.title(f"Landmark sampling strategy vs. accuracy (k={K})")
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_dimension_stratified_comparison.png"), dpi=150)
    plt.close()


def plot_num_landmark(outdir, acc, train_t, eval_t, suffix="", title_suffix=""):
    """suffix/title_suffix let the finer round-2 sweep reuse this rather than duplicating it
    wholesale -- e.g. suffix="_fine", title_suffix=" (fine sweep)"."""
    xs = sorted(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [acc[x] * 100 for x in xs], marker='o')
    plt.xlabel("Number of landmarks"); plt.ylabel("Inference accuracy [%]")
    plt.title(f"Accuracy vs. landmark count (D={D_BASELINE}, k={K}){title_suffix}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"accuracy_vs_num_landmark{suffix}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [train_t[x] for x in xs], marker='o', label="training time")
    plt.plot(xs, [eval_t[x] for x in xs], marker='s', label="evaluation time")
    plt.xlabel("Number of landmarks"); plt.ylabel("Time [s]")
    plt.title(f"Training vs. evaluation time by landmark count (D={D_BASELINE}, k={K}){title_suffix}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"time_vs_num_landmark{suffix}.png"), dpi=150)
    plt.close()


def plot_bit_budget(outdir, accuracy_by_budget_and_encoders):
    """One line per total bit budget; x-axis = number of independent encoders sharing that
    budget, y-axis = accuracy. Puts every budget on one chart so shrinking the total bit
    budget's effect on the ensembling tradeoff is visible directly, rather than needing a
    separate figure per budget."""
    plt.figure(figsize=(8, 5))
    for budget in sorted(accuracy_by_budget_and_encoders, reverse=True):
        by_encoders = accuracy_by_budget_and_encoders[budget]
        xs = sorted(by_encoders)
        ys = [by_encoders[x] * 100 for x in xs]
        plt.plot(xs, ys, marker='o', label=f"budget={budget}")
        for x, y in zip(xs, ys):
            plt.annotate(f"D={budget // x}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=7)
    all_encoder_counts = sorted({x for by_encoders in accuracy_by_budget_and_encoders.values()
                                  for x in by_encoders})
    plt.xticks(all_encoder_counts)
    plt.xlabel("Number of independent encoders (ensemble size)")
    plt.ylabel("Inference accuracy [%]")
    plt.title("Ensembling accuracy vs. encoder count, by total bit budget")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_bit_budget.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or None,
                        help="Pool size. Defaults to SLURM_CPUS_ON_NODE, else all cores.")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for a smoke test.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dims = [1024, 64] if args.quick else DIMENSIONS
    k_values = [3] if args.quick else K_VALUES
    num_landmark_values = [500] if args.quick else NUM_LANDMARK_VALUES
    bit_budgets = [1024, 128] if args.quick else BIT_BUDGETS
    ensemble_sizes = [1, 3] if args.quick else ENSEMBLE_SIZES
    num_landmark_values_fine = [200] if args.quick else NUM_LANDMARK_VALUES_FINE

    # ---- Build the full flat job list across every experiment, then run once. ----
    # Tagging each job with its section lets us fan the whole sweep out over ONE pool
    # (max core utilization) and demultiplex results afterward, instead of a barrier
    # between sections that would leave cores idle waiting for the slowest point.
    jobs = []  # (section, worker_fn_name, job_args)

    # Section: baseline (D=1024) + dimension sweep -- Nystrom, default (~2000) landmarks.
    for d in dims:
        jobs.append(("dim_nys", "w_nystrom_point", (d, K, None, np.float64, None, "pooled")))
        jobs.append(("dim_ngram", "w_ngram_point", (d, N, None)))

    # Section: precision sweep at D=1024 (both methods).
    for p in PRECISIONS:
        jobs.append(("prec_nys", "w_nystrom_precision", (D_BASELINE, K, p)))
        jobs.append(("prec_ngram", "w_ngram_point", (D_BASELINE, N, p)))

    # Section: k-sweep {3,4}, float32 + batching + 500 landmarks.
    for k in k_values:
        for d in dims:
            jobs.append(("ksweep", "w_nystrom_point",
                         (d, k, K_SWEEP_NUM_LANDMARK, K_SWEEP_DTYPE, K_SWEEP_BATCH_SIZE, "pooled")))

    # Section: stratified vs unstratified, k=3, 500 landmarks.
    for d in dims:
        jobs.append(("strat_u", "w_nystrom_point", (d, K, STRATIFY_NUM_LANDMARK, np.float64, None, "pooled")))
        jobs.append(("strat_s", "w_nystrom_point", (d, K, STRATIFY_NUM_LANDMARK, np.float64, None, "stratified")))

    # Section: landmark-count sweep at fixed D=1024, k=3.
    for nl in num_landmark_values:
        jobs.append(("nlsweep", "w_nystrom_point", (D_BASELINE, K, nl, np.float64, None, "pooled")))

    # Section: bit-budget x ensemble-size sweep, k=3, 500 landmarks. Sweeps total bit budget
    # over `bit_budgets` and, at each budget, ensemble size over `ensemble_sizes` (1 = single
    # full-width encoder, the baseline for that budget). Self-contained at a fixed
    # num_landmark=ENSEMBLE_NUM_LANDMARK -- not reusing the dimension sweep's default (~2000)
    # landmark points above -- so landmark count doesn't confound the budget/encoder-count
    # comparison this section is meant to isolate.
    for budget in bit_budgets:
        for n_encoder in ensemble_sizes:
            jobs.append(("ensemble", "w_ensemble", (budget, n_encoder, K, ENSEMBLE_NUM_LANDMARK)))

    # Section (round 2): finer landmark-count sweep at fixed D=1024, k=3.
    for nl in num_landmark_values_fine:
        jobs.append(("nlsweep_fine", "w_nystrom_point", (D_BASELINE, K, nl, np.float64, None, "pooled")))

    print(f"Dispatching {len(jobs)} experiment points over {args.workers or 'all'} workers...", flush=True)
    wall0 = time.time()

    # imap_unordered so finished points stream back and print as they land, rather
    # than blocking on the slowest. chunksize=1 since points vary widely in cost.
    with Pool(processes=args.workers) as pool:
        results = []
        for section, res in pool.imap_unordered(_dispatch, jobs, chunksize=1):
            results.append((section, res))
            _print_point(section, res)

    print(f"\nAll points done in {time.time() - wall0:.1f}s wall.", flush=True)

    # ---- Demultiplex results by section. ----
    by = {}
    for section, res in results:
        by.setdefault(section, []).append(res)

    def dim_map(section):
        return {r["dimension"]: r["accuracy"] for r in by.get(section, [])}

    # Dimension sweep + baseline.
    by_dim_nys = dim_map("dim_nys")
    by_dim_ngram = dim_map("dim_ngram")

    # Precision sweep (D=1024). Point at precision=16 is the exact-Hamming baseline.
    by_prec_nys = {16: by_dim_nys.get(D_BASELINE)}
    by_prec_nys.update({r["precision"]: r["accuracy"] for r in by.get("prec_nys", [])})
    by_prec_ngram = {16: by_dim_ngram.get(D_BASELINE)}
    by_prec_ngram.update({r["precision"]: r["accuracy"] for r in by.get("prec_ngram", [])})

    # k-sweep.
    accuracy_by_k = {}
    for r in by.get("ksweep", []):
        accuracy_by_k.setdefault(r["k"], {})[r["dimension"]] = r["accuracy"]

    # Stratified comparison.
    unstrat = {r["dimension"]: r["accuracy"] for r in by.get("strat_u", [])}
    strat = {r["dimension"]: r["accuracy"] for r in by.get("strat_s", [])}
    per_lang_u = {r["dimension"]: r["per_language"] for r in by.get("strat_u", [])}
    per_lang_s = {r["dimension"]: r["per_language"] for r in by.get("strat_s", [])}

    # Landmark-count sweep.
    nl_acc = {r["num_landmark"]: r["accuracy"] for r in by.get("nlsweep", [])}
    nl_train = {r["num_landmark"]: r["train_s"] for r in by.get("nlsweep", [])}
    nl_eval = {r["num_landmark"]: r["eval_s"] for r in by.get("nlsweep", [])}

    # Bit-budget x ensemble-size sweep.
    accuracy_by_budget_and_encoders = {}
    for r in by.get("ensemble", []):
        accuracy_by_budget_and_encoders.setdefault(r["total_dimension"], {})[r["num_encoders"]] = r["accuracy"]

    # Round 2: finer landmark-count sweep.
    nl_fine_acc = {r["num_landmark"]: r["accuracy"] for r in by.get("nlsweep_fine", [])}
    nl_fine_train = {r["num_landmark"]: r["train_s"] for r in by.get("nlsweep_fine", [])}
    nl_fine_eval = {r["num_landmark"]: r["eval_s"] for r in by.get("nlsweep_fine", [])}

    # ---- Plots. ----
    plot_dimension(args.outdir, by_dim_nys, by_dim_ngram)
    plot_precision(args.outdir, by_prec_nys, by_prec_ngram)
    plot_by_k(args.outdir, accuracy_by_k)
    plot_stratified(args.outdir, unstrat, strat, per_lang_u, per_lang_s)
    plot_num_landmark(args.outdir, nl_acc, nl_train, nl_eval)
    plot_bit_budget(args.outdir, accuracy_by_budget_and_encoders)
    plot_num_landmark(args.outdir, nl_fine_acc, nl_fine_train, nl_fine_eval,
                      suffix="_fine", title_suffix=" (fine sweep)")

    # ---- Dump every number to JSON for downstream analysis / provenance. ----
    summary = {
        "config": {"K": K, "N": N, "dimensions": dims, "k_values": k_values,
                   "num_landmark_values": num_landmark_values,
                   "bit_budgets": bit_budgets, "ensemble_sizes": ensemble_sizes,
                   "num_landmark_values_fine": num_landmark_values_fine,
                   "quick": args.quick, "wall_seconds": time.time() - wall0},
        "accuracy_by_dimension_nystrom": by_dim_nys,
        "accuracy_by_dimension_ngram": by_dim_ngram,
        "accuracy_by_precision_nystrom": by_prec_nys,
        "accuracy_by_precision_ngram": by_prec_ngram,
        "accuracy_by_k": accuracy_by_k,
        "accuracy_stratified": strat,
        "accuracy_unstratified": unstrat,
        "per_language_stratified": per_lang_s,
        "per_language_unstratified": per_lang_u,
        "accuracy_by_num_landmark": nl_acc,
        "train_time_by_num_landmark": nl_train,
        "eval_time_by_num_landmark": nl_eval,
        "accuracy_by_budget_and_encoders": accuracy_by_budget_and_encoders,
        "accuracy_by_num_landmark_fine": nl_fine_acc,
        "train_time_by_num_landmark_fine": nl_fine_train,
        "eval_time_by_num_landmark_fine": nl_fine_eval,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote plots + results.json to {args.outdir}/", flush=True)
    _print_report(summary)


# Module-level dispatcher (picklable) that Pool actually maps over.
_WORKER_FNS = None


def _dispatch(tagged):
    section, fn_name, job_args = tagged
    fns = {"w_nystrom_point": w_nystrom_point, "w_nystrom_precision": w_nystrom_precision,
           "w_ngram_point": w_ngram_point, "w_ensemble": w_ensemble}
    return section, fns[fn_name](job_args)


def _print_point(section, res):
    if "num_encoders" in res:
        budget = f" budget={res['total_dimension']}" if "total_dimension" in res else ""
        print(f"[{section}]{budget} num_encoders={res['num_encoders']} "
              f"(D_small={res['encoder_dimension']}) -> acc={res['accuracy']:.4f}", flush=True)
    elif "precision" in res:
        print(f"[{section}] D={res['dimension']} precision={res['precision']} "
              f"-> acc={res['accuracy']:.4f}", flush=True)
    elif res.get("method") == "ngram":
        print(f"[{section}] D={res['dimension']} N={res['n']} -> acc={res['accuracy']:.4f}", flush=True)
    else:
        extra = f" landmarks={res.get('num_landmark')}" if res.get("num_landmark") else ""
        print(f"[{section}] D={res['dimension']} k={res['k']}{extra} "
              f"-> acc={res['accuracy']:.4f} (train {res.get('train_s', 0):.1f}s)", flush=True)


def _print_report(summary):
    print("\n===== SUMMARY =====")
    print("Dimension sweep (exact Hamming):")
    for d in sorted(summary["accuracy_by_dimension_nystrom"], reverse=True):
        n = summary["accuracy_by_dimension_nystrom"][d]
        g = summary["accuracy_by_dimension_ngram"].get(d)
        print(f"  D={d:5d}  Nystrom={n:.4f}  N-gram={g:.4f}" if g else f"  D={d:5d}  Nystrom={n:.4f}")


if __name__ == "__main__":
    main()
