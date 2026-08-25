"""Extends hdc_train.py's N-gram sweep (N=3,4,5) down to N=1 and N=2, to see
whether the N=3 > N=4,5 trend (more context helps, up to a point) continues to
hold or reverses when N drops below 3.

N=1 discards all sequential/context information -- encode_text_to_hv's N-gram
hypervector construction (bind(V_0, permute(V_1,1), ..., permute(V_{n-1},n-1)))
degenerates to just item_memory[char] itself (no bind, no permute), so the class
hypervector becomes a plain bag-of-characters majority vote: this should behave
like a per-language character-frequency classifier, which still works reasonably
well for language ID (different languages have different letter frequencies) but
loses any word-shape/digraph signal.

N=2 is the smallest N that actually exercises bind+permute (digraphs), so it's
the natural midpoint between the frequency-only N=1 baseline and the N=3 result
already measured on this branch (hdc_train.py's DIMENSIONS x [3,4,5] sweep).

Reuses encode_text_to_hv/hdc.py unchanged -- same encoder, just swept over a
wider N range -- and the same dimension grid as hdc_train.py so the two plots
are directly comparable (N=3,4,5 curves should match hdc_train.py's output up to
RNG noise from the multiprocessing Pool scheduling order, since generate_item_memory
is seeded per (dimension, n_gram_size) task).

Usage:
    python scripts/run_ngram_low_n_experiment.py [--outdir DIR]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hdc
from encoder import encode_text_to_hv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "language_recognition_dataset")
LANGUAGES = ["dan", "deu", "eng", "fra", "ita", "nld", "pol", "por", "spa", "swe"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz "

DIMENSIONS = list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
N_GRAM_SIZES = [1, 2, 3, 4, 5]  # extends hdc_train.py's [3, 4, 5] down to 1, 2


def clean_text(text):
    import re
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def load_training_text(language):
    path = os.path.join(DATA_DIR, "training", f"{language}.txt")
    with open(path, 'r', encoding='utf-8') as file:
        return clean_text(file.read())


def load_test_sentences(language):
    path = os.path.join(DATA_DIR, "testing", f"{language}.txt")
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [clean_text(line) for line in lines if line.strip()]


def train_class_vectors(item_memory, n_gram_size, dimension):
    class_hvs = {}
    for language in LANGUAGES:
        text = load_training_text(language)
        class_hvs[language] = encode_text_to_hv(text, item_memory, n_gram_size, dimension)
    return class_hvs


def classify(sentence_hv, class_hvs):
    return min(class_hvs, key=lambda language: np.count_nonzero(sentence_hv != class_hvs[language]))


def evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data):
    correct, total = 0, 0
    for true_language, sentences in test_data.items():
        for sentence in sentences:
            sentence_hv = encode_text_to_hv(sentence, item_memory, n_gram_size, dimension)
            correct += int(classify(sentence_hv, class_hvs) == true_language)
            total += 1
    return correct / total


def run_experiment(args):
    dimension, n_gram_size, test_data = args
    item_memory = hdc.generate_item_memory(ALPHABET, dimension, seed=42)
    class_hvs = train_class_vectors(item_memory, n_gram_size, dimension)
    accuracy = evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data)
    print(f"D={dimension:5d} N={n_gram_size} -> accuracy={accuracy:.4f}", flush=True)
    return dimension, n_gram_size, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading test sentences...", flush=True)
    test_data = {language: load_test_sentences(language) for language in LANGUAGES}

    tasks = [(dimension, n_gram_size, test_data)
             for n_gram_size in N_GRAM_SIZES for dimension in DIMENSIONS]

    with Pool(processes=args.workers) as pool:
        results = pool.map(run_experiment, tasks)

    accuracy_by_n = {n: {} for n in N_GRAM_SIZES}
    for dimension, n_gram_size, accuracy in results:
        accuracy_by_n[n_gram_size][dimension] = accuracy

    plt.figure(figsize=(8, 5))
    for n_gram_size in N_GRAM_SIZES:
        xs = sorted(accuracy_by_n[n_gram_size])
        ys = [accuracy_by_n[n_gram_size][d] * 100 for d in xs]
        plt.plot(xs, ys, marker='o', label=f"N={n_gram_size}")

    plt.xscale('log')
    plt.xlabel("Vector dimension [bit]")
    plt.ylabel("Inference accuracy [%]")
    plt.title("Language recognition accuracy vs. vector dimension (N=1..5)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "accuracy_vs_dimension_low_n.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}", flush=True)

    # Best-D accuracy per N, isolating the effect of N alone.
    plt.figure(figsize=(7, 5))
    best_by_n = {n: max(accuracy_by_n[n].values()) for n in N_GRAM_SIZES}
    plt.bar([str(n) for n in N_GRAM_SIZES], [best_by_n[n] * 100 for n in N_GRAM_SIZES],
            color="#4C72B0")
    plt.xlabel("N-gram size")
    plt.ylabel("Best inference accuracy across D [%]")
    plt.title("Language recognition: best accuracy vs. N-gram size")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(args.outdir, "accuracy_vs_n_best.png")
    plt.savefig(bar_path, dpi=150)
    print(f"Saved plot to {bar_path}", flush=True)

    summary = {
        "dimensions": DIMENSIONS,
        "n_gram_sizes": N_GRAM_SIZES,
        "accuracy_by_n_then_d": {str(n): {str(d): accuracy_by_n[n][d] for d in accuracy_by_n[n]}
                                  for n in N_GRAM_SIZES},
        "best_accuracy_by_n": {str(n): best_by_n[n] for n in N_GRAM_SIZES},
    }
    results_path = os.path.join(args.outdir, "results_low_n.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {results_path}", flush=True)

    print("\n===== SUMMARY (best accuracy per N) =====")
    for n in N_GRAM_SIZES:
        print(f"  N={n}: {best_by_n[n]:.4f}")


if __name__ == "__main__":
    main()
