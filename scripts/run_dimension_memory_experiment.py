"""Measures memory usage as a function of hypervector dimension D, to make the
"larger D -> more memory" tradeoff concrete rather than just asserted.

Two independent measurements per D, both cheap to get on TACC (no external
profiler needed):
  1. Theoretical bytes/hypervector: each item-memory entry and each class
     hypervector is a NumPy array. generate_item_memory (hdc.py) stores dtype
     uint8, so this is exactly D bytes/vector. encode_text_to_hv's accumulator
     is int32 (4 bytes/element) but is freed once the function returns; the
     dtype/itemsize-based count captures the actual dtypes used at every stage
     without relying on OS-level attribution.
  2. Measured process peak RSS (resource.getrusage(RUSAGE_SELF).ru_maxrss, KB
     on Linux): runs the SAME single-language train+evaluate pipeline
     (train_class_vectors + evaluate, matching hdc_train.py) as a fresh
     subprocess per D, so each measurement's peak RSS isn't polluted by a
     previous, larger D still being cached/not-yet-garbage-collected in the
     same process. This is the actual OS-reported high-water-mark memory use
     of one full run at that D, including the Python interpreter's own
     baseline footprint (numpy import, dataset text in memory, etc.) -- so it
     is not zero at D=0 and does not scale as cleanly linear as the
     theoretical count, but it is what you would actually see in `sacct
     --format=MaxRSS` or `seff` for a real job.

Subprocess design: run_one_dimension() is invoked via `python -m
scripts.run_dimension_memory_experiment --single-d D` (self-re-exec), so each
measurement gets its own address space and getrusage() reading. The parent
process sweeps D, shells out once per D, and parses the child's peak-RSS line
from stdout.

Usage:
    python scripts/run_dimension_memory_experiment.py [--outdir DIR]
    python scripts/run_dimension_memory_experiment.py --single-d 1000   # internal, one measurement
"""
import argparse
import json
import os
import resource
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DIMENSIONS = list(range(100, 1001, 100)) + list(range(2000, 10001, 1000)) + [20000, 50000]
N_GRAM_SIZE = 3
LANGUAGE = "eng"  # one language is enough to characterize per-D memory scaling
ALPHABET = "abcdefghijklmnopqrstuvwxyz "


def run_one_dimension(dimension):
    """Runs train_class_vectors (all 10 languages, matching hdc_train.py) +
    evaluate for a single dimension in THIS process, then reports this
    process's peak RSS. Meant to be invoked as a fresh subprocess per D."""
    import re
    import time

    import numpy as np

    import hdc
    from encoder import encode_text_to_hv

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "language_recognition_dataset")
    languages = ["dan", "deu", "eng", "fra", "ita", "nld", "pol", "por", "spa", "swe"]

    def clean_text(text):
        text = text.replace('\n', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def load_training_text(language):
        path = os.path.join(data_dir, "training", f"{language}.txt")
        with open(path, 'r', encoding='utf-8') as file:
            return clean_text(file.read())

    def load_test_sentences(language):
        path = os.path.join(data_dir, "testing", f"{language}.txt")
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [clean_text(line) for line in lines if line.strip()]

    def classify(sentence_hv, class_hvs):
        return min(class_hvs, key=lambda language: np.count_nonzero(sentence_hv != class_hvs[language]))

    t0 = time.time()
    item_memory = hdc.generate_item_memory(ALPHABET, dimension, seed=42)

    class_hvs = {}
    for language in languages:
        text = load_training_text(language)
        class_hvs[language] = encode_text_to_hv(text, item_memory, N_GRAM_SIZE, dimension)

    correct, total = 0, 0
    for sentence in load_test_sentences(LANGUAGE):
        sentence_hv = encode_text_to_hv(sentence, item_memory, N_GRAM_SIZE, dimension)
        correct += int(classify(sentence_hv, class_hvs) == LANGUAGE)
        total += 1
    accuracy = correct / total if total else 0.0
    wall_s = time.time() - t0

    # ru_maxrss is KB on Linux, bytes on macOS -- TACC nodes are Linux.
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Theoretical bytes actually allocated for the D-sized arrays that persist
    # past this function: item_memory (len(ALPHABET) vectors, uint8) + one
    # class hypervector per language (uint8).
    item_memory_bytes = len(item_memory) * dimension * np.dtype(np.uint8).itemsize
    class_hv_bytes = len(class_hvs) * dimension * np.dtype(np.uint8).itemsize
    theoretical_bytes = item_memory_bytes + class_hv_bytes

    print(f"RESULT dimension={dimension} accuracy={accuracy:.4f} "
          f"peak_rss_kb={peak_rss_kb} theoretical_bytes={theoretical_bytes} "
          f"wall_s={wall_s:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--single-d", type=int, default=None,
                         help="internal: run only this dimension in-process and print RESULT line")
    args = parser.parse_args()

    if args.single_d is not None:
        run_one_dimension(args.single_d)
        return

    os.makedirs(args.outdir, exist_ok=True)
    script_path = os.path.abspath(__file__)

    records = []
    for dimension in DIMENSIONS:
        proc = subprocess.run(
            [sys.executable, script_path, "--single-d", str(dimension)],
            capture_output=True, text=True, check=True,
        )
        line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
        fields = dict(kv.split("=") for kv in line[len("RESULT "):].split())
        record = {
            "dimension": int(fields["dimension"]),
            "accuracy": float(fields["accuracy"]),
            "peak_rss_kb": int(fields["peak_rss_kb"]),
            "theoretical_bytes": int(fields["theoretical_bytes"]),
            "wall_s": float(fields["wall_s"]),
        }
        records.append(record)
        print(f"D={record['dimension']:6d} -> peak_rss={record['peak_rss_kb'] / 1024:.1f} MiB "
              f"theoretical={record['theoretical_bytes'] / 1024:.2f} KiB "
              f"accuracy={record['accuracy']:.4f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [r["dimension"] for r in records]
    rss_mib = [r["peak_rss_kb"] / 1024 for r in records]
    theoretical_kib = [r["theoretical_bytes"] / 1024 for r in records]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dims, rss_mib, marker='o', color="#4C72B0", label="Measured peak RSS (subprocess)")
    ax1.set_xlabel("Vector dimension D [bit]")
    ax1.set_ylabel("Peak RSS [MiB]", color="#4C72B0")
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor="#4C72B0")
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(dims, theoretical_kib, marker='s', color="#DD8452",
              label="Theoretical item-memory + class-HV bytes")
    ax2.set_ylabel("Theoretical size [KiB]", color="#DD8452")
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor="#DD8452")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    plt.title("Memory usage vs. hypervector dimension (TACC)")
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "memory_vs_dimension.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}", flush=True)

    results_path = os.path.join(args.outdir, "results_memory_vs_dimension.json")
    with open(results_path, "w") as f:
        json.dump({"n_gram_size": N_GRAM_SIZE, "language": LANGUAGE, "records": records}, f, indent=2)
    print(f"Wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
