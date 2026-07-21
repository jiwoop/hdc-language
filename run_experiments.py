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

# --- Round 3: multi-run averaging ---
# Random landmark sampling (and the random projection it feeds) means a single seed's
# result doesn't generalize -- re-run every point across N_SEEDS distinct seeds and report
# mean +/- std, not just one draw. Applies only to the fine landmark-count sweep and the
# stratified-vs-unstratified comparison, per-request; other sections stay single-run (seed=42).
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level so Pool can pickle them).
# Each returns a small plain-data tuple/dict; heavy arrays stay inside the worker.
# test_data is rebuilt inside each worker rather than pickled across -- it's cheap
# to load and avoids shipping the whole test corpus to every process.
# ---------------------------------------------------------------------------
def _test_data():
    return {language: t.load_test_sentences(language) for language in t.LANGUAGES}


def w_nystrom_point(job):
    """(dimension, k, num_landmark, dtype, batch_size, strategy, seed=42) -> result dict.

    strategy: 'pooled' (default sampler) or 'stratified'. Covers the baseline point,
    dimension sweep, k-sweep, and stratified comparison with one worker body.

    seed: forwarded to train_class_vectors_nystrom (drives both landmark sampling and the
    random projection), defaulting to 42 to preserve existing single-run call sites. Callers
    doing multi-run averaging pass a distinct seed per run.
    """
    if len(job) == 7:
        dimension, k, num_landmark, dtype, batch_size, strategy, seed = job
    else:
        dimension, k, num_landmark, dtype, batch_size, strategy = job
        seed = 42
    sampler = ne.sample_landmarks_stratified if strategy == "stratified" else None
    test_data = _test_data()

    t0 = time.time()
    class_hvs, enc, lf, lsk = ne.train_class_vectors_nystrom(
        dimension, k, num_landmark=num_landmark, dtype=dtype,
        batch_size=batch_size, landmark_sampler=sampler, seed=seed)
    train_s = time.time() - t0

    t0 = time.time()
    acc, per_lang = ne.evaluate_nystrom(
        class_hvs, enc, lf, lsk, k, dimension, test_data,
        dtype=dtype, batch_size=batch_size, return_per_language=True)
    eval_s = time.time() - t0

    return {
        "method": "nystrom", "dimension": dimension, "k": k,
        "num_landmark": num_landmark, "strategy": strategy, "seed": seed,
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
# Multi-run aggregation helpers -- collapse N seeded runs per x-value into mean/std,
# used by the landmark-count sweep and stratified-sampling comparison sections.
# ---------------------------------------------------------------------------
def _runs_by_key(records, key, value_key="accuracy"):
    """[{key: x, value_key: v, ...}, ...] -> {x: [v_seed0, v_seed1, ...]}."""
    out = {}
    for r in records:
        out.setdefault(r[key], []).append(r[value_key])
    return out


def _mean_std_map(runs_by_x):
    """{x: [v, ...]} -> ({x: mean}, {x: std})."""
    mean = {x: float(np.mean(vs)) for x, vs in runs_by_x.items()}
    std = {x: float(np.std(vs)) for x, vs in runs_by_x.items()}
    return mean, std


def _per_language_mean(records, dim_key="dimension"):
    """records with a per-language accuracy dict per seed -> {dimension: {language: mean_acc}}."""
    by_dim = {}
    for r in records:
        by_dim.setdefault(r[dim_key], []).append(r["per_language"])
    out = {}
    for dim, per_lang_list in by_dim.items():
        languages = per_lang_list[0].keys()
        out[dim] = {lang: float(np.mean([pl[lang] for pl in per_lang_list])) for lang in languages}
    return out


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


def plot_stratified(outdir, unstrat, strat, per_lang_u, per_lang_s,
                     unstrat_std=None, strat_std=None, n_runs=None):
    """unstrat_std/strat_std: optional {dimension: std} from multi-run averaging, plotted as
    error bars. n_runs, if given, is noted in the title."""
    xs = sorted(unstrat, reverse=True)
    runs_note = f", mean of {n_runs} runs" if n_runs else ""
    plt.figure(figsize=(8, 5))
    yerr_u = [unstrat_std[x] * 100 for x in xs] if unstrat_std else None
    yerr_s = [strat_std[x] * 100 for x in xs] if strat_std else None
    plt.errorbar(xs, [unstrat[x] * 100 for x in xs], yerr=yerr_u, marker='o', capsize=3,
                 label="unstratified (pooled random)")
    plt.errorbar(xs, [strat[x] * 100 for x in xs], yerr=yerr_s, marker='s', capsize=3,
                 label="stratified (equal per language)")
    plt.gca().invert_xaxis(); plt.xscale('log')
    plt.xlabel("Vector dimension [bit]"); plt.ylabel("Inference accuracy [%]")
    plt.title(f"Landmark sampling strategy vs. accuracy (k={K}){runs_note}")
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_dimension_stratified_comparison.png"), dpi=150)
    plt.close()


def plot_num_landmark(outdir, acc, train_t, eval_t, suffix="", title_suffix="", acc_std=None, n_runs=None):
    """suffix/title_suffix let the finer round-2 sweep reuse this rather than duplicating it
    wholesale -- e.g. suffix="_fine", title_suffix=" (fine sweep)".

    acc_std: optional {num_landmark: std} from multi-run averaging, plotted as error bars.
    n_runs, if given, is noted in the title."""
    xs = sorted(acc)
    runs_note = f", mean of {n_runs} runs" if n_runs else ""
    plt.figure(figsize=(8, 5))
    yerr = [acc_std[x] * 100 for x in xs] if acc_std else None
    plt.errorbar(xs, [acc[x] * 100 for x in xs], yerr=yerr, marker='o', capsize=3)
    plt.xlabel("Number of landmarks"); plt.ylabel("Inference accuracy [%]")
    plt.title(f"Accuracy vs. landmark count (D={D_BASELINE}, k={K}){title_suffix}{runs_note}")
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
                        help="Pool size before the memory cap is applied. Defaults to "
                             "SLURM_CPUS_ON_NODE, else all cores.")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for a smoke test.")
    parser.add_argument("--mem-per-worker-gb", type=float, default=6.0,
                        help="Assumed worst-case peak RSS per worker (GB), used to cap Pool "
                             "size against total node memory. The largest unbatched float64 "
                             "point (num_landmark=2000, D=1024) measured ~3.4GB+ at "
                             "num_landmark=500 and grows with num_landmark, so 6GB leaves "
                             "headroom. One process running 128-wide previously OOM'd a "
                             "256GB node (job 3305101) because 128 workers x ~3-4GB+ each "
                             "exceeds available memory once several heavy points overlap.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dims = [1024, 64] if args.quick else DIMENSIONS
    k_values = [3] if args.quick else K_VALUES
    num_landmark_values = [500] if args.quick else NUM_LANDMARK_VALUES
    bit_budgets = [1024, 128] if args.quick else BIT_BUDGETS
    ensemble_sizes = [1, 3] if args.quick else ENSEMBLE_SIZES
    num_landmark_values_fine = [200] if args.quick else NUM_LANDMARK_VALUES_FINE
    seeds = SEEDS[:2] if args.quick else SEEDS

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

    # Section: stratified vs unstratified, k=3, 500 landmarks. Random landmark sampling
    # needs multiple runs to generalize -- 10 seeds per (dimension, strategy) point.
    for d in dims:
        for seed in seeds:
            jobs.append(("strat_u", "w_nystrom_point", (d, K, STRATIFY_NUM_LANDMARK, np.float64, None, "pooled", seed)))
            jobs.append(("strat_s", "w_nystrom_point", (d, K, STRATIFY_NUM_LANDMARK, np.float64, None, "stratified", seed)))

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

    # Section (round 2/3): finer landmark-count sweep at fixed D=1024, k=3. 10 seeds per
    # num_landmark point since random landmark sampling needs averaging to generalize.
    for nl in num_landmark_values_fine:
        for seed in seeds:
            jobs.append(("nlsweep_fine", "w_nystrom_point", (D_BASELINE, K, nl, np.float64, None, "pooled", seed)))

    # ---- Cap Pool size by available memory, not just core count. ----
    # job 3305101 OOM'd a 128-core/256GB node: 128 concurrent unbatched float64
    # train_class_vectors_nystrom workers (each peaking several GB, see --mem-per-worker-gb
    # help) overran available memory once enough heavy points landed in the pool at once.
    # One core per worker is necessary but not sufficient -- also bound by memory/worker.
    requested_workers = args.workers or os.cpu_count() or 1
    try:
        total_mem_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
        mem_capped_workers = max(1, int(total_mem_gb // args.mem_per_worker_gb))
    except (ValueError, OSError):
        total_mem_gb = None
        mem_capped_workers = requested_workers
    workers = min(requested_workers, mem_capped_workers)
    if workers < requested_workers:
        print(f"Capping workers {requested_workers} -> {workers}: "
              f"{total_mem_gb:.0f}GB total memory / {args.mem_per_worker_gb}GB per worker "
              f"(override with --mem-per-worker-gb)", flush=True)

    print(f"Dispatching {len(jobs)} experiment points over {workers} workers...", flush=True)
    wall0 = time.time()

    # imap_unordered so finished points stream back and print as they land, rather
    # than blocking on the slowest. chunksize=1 since points vary widely in cost.
    with Pool(processes=workers) as pool:
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

    # Stratified comparison -- 10 seeded runs per (dimension, strategy); mean/std for
    # plotting, raw per-seed accuracy kept in the JSON dump for provenance.
    unstrat_runs = _runs_by_key(by.get("strat_u", []), "dimension")
    strat_runs = _runs_by_key(by.get("strat_s", []), "dimension")
    unstrat, unstrat_std = _mean_std_map(unstrat_runs)
    strat, strat_std = _mean_std_map(strat_runs)
    per_lang_u = _per_language_mean(by.get("strat_u", []))
    per_lang_s = _per_language_mean(by.get("strat_s", []))

    # Landmark-count sweep.
    nl_acc = {r["num_landmark"]: r["accuracy"] for r in by.get("nlsweep", [])}
    nl_train = {r["num_landmark"]: r["train_s"] for r in by.get("nlsweep", [])}
    nl_eval = {r["num_landmark"]: r["eval_s"] for r in by.get("nlsweep", [])}

    # Bit-budget x ensemble-size sweep.
    accuracy_by_budget_and_encoders = {}
    for r in by.get("ensemble", []):
        accuracy_by_budget_and_encoders.setdefault(r["total_dimension"], {})[r["num_encoders"]] = r["accuracy"]

    # Round 2/3: finer landmark-count sweep -- 10 seeded runs per num_landmark point.
    nl_fine_runs = _runs_by_key(by.get("nlsweep_fine", []), "num_landmark")
    nl_fine_acc, nl_fine_acc_std = _mean_std_map(nl_fine_runs)
    nl_fine_train_runs = _runs_by_key(by.get("nlsweep_fine", []), "num_landmark", value_key="train_s")
    nl_fine_eval_runs = _runs_by_key(by.get("nlsweep_fine", []), "num_landmark", value_key="eval_s")
    nl_fine_train, _ = _mean_std_map(nl_fine_train_runs)
    nl_fine_eval, _ = _mean_std_map(nl_fine_eval_runs)

    # ---- Plots. ----
    plot_dimension(args.outdir, by_dim_nys, by_dim_ngram)
    plot_precision(args.outdir, by_prec_nys, by_prec_ngram)
    plot_by_k(args.outdir, accuracy_by_k)
    plot_stratified(args.outdir, unstrat, strat, per_lang_u, per_lang_s,
                     unstrat_std=unstrat_std, strat_std=strat_std, n_runs=len(seeds))
    plot_num_landmark(args.outdir, nl_acc, nl_train, nl_eval)
    plot_bit_budget(args.outdir, accuracy_by_budget_and_encoders)
    plot_num_landmark(args.outdir, nl_fine_acc, nl_fine_train, nl_fine_eval,
                      suffix="_fine", title_suffix=" (fine sweep)",
                      acc_std=nl_fine_acc_std, n_runs=len(seeds))

    # ---- Dump every number to JSON for downstream analysis / provenance. ----
    summary = {
        "config": {"K": K, "N": N, "dimensions": dims, "k_values": k_values,
                   "num_landmark_values": num_landmark_values,
                   "bit_budgets": bit_budgets, "ensemble_sizes": ensemble_sizes,
                   "num_landmark_values_fine": num_landmark_values_fine,
                   "seeds": seeds,
                   "quick": args.quick, "wall_seconds": time.time() - wall0},
        "accuracy_by_dimension_nystrom": by_dim_nys,
        "accuracy_by_dimension_ngram": by_dim_ngram,
        "accuracy_by_precision_nystrom": by_prec_nys,
        "accuracy_by_precision_ngram": by_prec_ngram,
        "accuracy_by_k": accuracy_by_k,
        # Stratified comparison: mean + std across `seeds` runs, plus every raw per-seed
        # point (dimension -> [accuracy_seed0, accuracy_seed1, ...]) for provenance.
        "accuracy_stratified": strat,
        "accuracy_stratified_std": strat_std,
        "accuracy_stratified_runs": strat_runs,
        "accuracy_unstratified": unstrat,
        "accuracy_unstratified_std": unstrat_std,
        "accuracy_unstratified_runs": unstrat_runs,
        "per_language_stratified": per_lang_s,
        "per_language_unstratified": per_lang_u,
        "accuracy_by_num_landmark": nl_acc,
        "train_time_by_num_landmark": nl_train,
        "eval_time_by_num_landmark": nl_eval,
        "accuracy_by_budget_and_encoders": accuracy_by_budget_and_encoders,
        # Fine landmark-count sweep: mean + std across `seeds` runs, plus every raw
        # per-seed point (num_landmark -> [accuracy_seed0, accuracy_seed1, ...]).
        "accuracy_by_num_landmark_fine": nl_fine_acc,
        "accuracy_by_num_landmark_fine_std": nl_fine_acc_std,
        "accuracy_by_num_landmark_fine_runs": nl_fine_runs,
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
        strategy = f" strategy={res['strategy']}" if res.get("strategy") not in (None, "pooled") else ""
        seed = f" seed={res['seed']}" if "seed" in res else ""
        print(f"[{section}] D={res['dimension']} k={res['k']}{extra}{strategy}{seed} "
              f"-> acc={res['accuracy']:.4f} (train {res.get('train_s', 0):.1f}s)", flush=True)


def _print_report(summary):
    print("\n===== SUMMARY =====")
    print("Dimension sweep (exact Hamming):")
    for d in sorted(summary["accuracy_by_dimension_nystrom"], reverse=True):
        n = summary["accuracy_by_dimension_nystrom"][d]
        g = summary["accuracy_by_dimension_ngram"].get(d)
        print(f"  D={d:5d}  Nystrom={n:.4f}  N-gram={g:.4f}" if g else f"  D={d:5d}  Nystrom={n:.4f}")

    n_seeds = len(summary["config"]["seeds"])
    print(f"\nStratified vs. unstratified landmark sampling (mean +/- std over {n_seeds} runs):")
    for d in sorted(summary["accuracy_stratified"], reverse=True):
        s, s_std = summary["accuracy_stratified"][d], summary["accuracy_stratified_std"][d]
        u, u_std = summary["accuracy_unstratified"][d], summary["accuracy_unstratified_std"][d]
        print(f"  D={d:5d}  stratified={s:.4f}+/-{s_std:.4f}  unstratified={u:.4f}+/-{u_std:.4f}")

    print(f"\nLandmark-count sweep, fine (mean +/- std over {n_seeds} runs):")
    for nl in sorted(summary["accuracy_by_num_landmark_fine"]):
        a = summary["accuracy_by_num_landmark_fine"][nl]
        a_std = summary["accuracy_by_num_landmark_fine_std"][nl]
        print(f"  num_landmark={nl:5d}  acc={a:.4f}+/-{a_std:.4f}")


if __name__ == "__main__":
    main()
