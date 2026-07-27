import os
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

import hdc
from encoder import encode_text_to_hv

DATA_DIR = "language_recognition_dataset"
LANGUAGES = ["dan", "deu", "eng", "fra", "ita", "nld", "pol", "por", "spa", "swe"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz "


def clean_text(text):
    """Collapses a chunk of text (already lowercase a-z + space) into one line."""
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


def train_class_vectors(item_memory, n_gram_size, dimension, max_val=None):
    """Encode each language's full training text into one class hypervector."""
    class_hvs = {}
    for language in LANGUAGES:
        text = load_training_text(language)
        class_hvs[language] = encode_text_to_hv(text, item_memory, n_gram_size, dimension, max_val)
    return class_hvs


def block_hamming_distance(hv1, hv2, precision, block_size=16):
    """Hamming distance as a fixed-width N-bit machine would compute it:
    split into `block_size`-bit blocks, saturate each block's local
    mismatch count at `precision`, then sum the (possibly-saturated)
    per-block counts."""
    mismatches = (hv1 != hv2)
    num_full_blocks = len(mismatches) // block_size
    trimmed = mismatches[:num_full_blocks * block_size].reshape(-1, block_size)
    block_hds = np.count_nonzero(trimmed, axis=1)
    total = np.clip(block_hds, a_min=None, a_max=precision).sum()

    remainder = mismatches[num_full_blocks * block_size:]
    if remainder.size:
        total += min(np.count_nonzero(remainder), precision)

    return total


def classify(sentence_hv, class_hvs, precision=None, block_size=16):
    """Nearest class by Hamming distance."""
    if precision is None:
        return min(class_hvs, key=lambda language: np.count_nonzero(sentence_hv != class_hvs[language]))
    return min(class_hvs, key=lambda language: block_hamming_distance(
        sentence_hv, class_hvs[language], precision, block_size))


def evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data, max_val=None, precision=None, block_size=16):
    correct, total = 0, 0
    for true_language, sentences in test_data.items():
        for sentence in sentences:
            sentence_hv = encode_text_to_hv(sentence, item_memory, n_gram_size, dimension, max_val)
            correct += int(classify(sentence_hv, class_hvs, precision, block_size) == true_language)
            total += 1
    return correct / total


def run_experiment(args):
    dimension, n_gram_size, test_data = args
    item_memory = hdc.generate_item_memory(ALPHABET, dimension, seed=42)
    class_hvs = train_class_vectors(item_memory, n_gram_size, dimension)
    accuracy = evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data)
    print(f"D={dimension:5d} N={n_gram_size} -> accuracy={accuracy:.4f}")
    return dimension, n_gram_size, accuracy


def run_precision_experiment(args):
    """Sweep the accumulator's saturating-counter bound (max_val) at a fixed D, N."""
    max_val, dimension, n_gram_size, test_data = args
    item_memory = hdc.generate_item_memory(ALPHABET, dimension, seed=42)
    class_hvs = train_class_vectors(item_memory, n_gram_size, dimension, max_val)
    accuracy = evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data, max_val)
    print(f"max_val={max_val:3d} -> accuracy={accuracy:.4f}")
    return max_val, accuracy


def run_hamming_precision_experiment(args):
    """Sweep the block-wise saturating Hamming distance precision (bits
    of local mismatch detectable per 16-bit block) across (D, N) scenarios."""
    precision, dimension, n_gram_size, test_data = args
    item_memory = hdc.generate_item_memory(ALPHABET, dimension, seed=42)
    class_hvs = train_class_vectors(item_memory, n_gram_size, dimension)
    accuracy = evaluate(item_memory, class_hvs, n_gram_size, dimension, test_data, precision=precision)
    print(f"D={dimension:5d} N={n_gram_size} precision={precision:3d} -> accuracy={accuracy:.4f}")
    return precision, dimension, n_gram_size, accuracy


def main():
    dimensions = list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
    n_gram_sizes = [3, 4, 5]

    print("Loading test sentences...")
    test_data = {language: load_test_sentences(language) for language in LANGUAGES}

    tasks = [(dimension, n_gram_size, test_data)
             for n_gram_size in n_gram_sizes for dimension in dimensions]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_experiment, tasks)

    accuracy_by_n = {n_gram_size: {} for n_gram_size in n_gram_sizes}
    for dimension, n_gram_size, accuracy in results:
        accuracy_by_n[n_gram_size][dimension] = accuracy

    plt.figure(figsize=(8, 5))
    for n_gram_size in n_gram_sizes:
        xs = sorted(accuracy_by_n[n_gram_size])
        ys = [accuracy_by_n[n_gram_size][d] * 100 for d in xs]
        plt.plot(xs, ys, marker='o', label=f"N={n_gram_size}")

    plt.xscale('log')
    plt.xlabel("Vector dimension [bit]")
    plt.ylabel("Inference accuracy [%]")
    plt.title("Language recognition accuracy vs. vector dimension")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_dimension.png", dpi=150)
    print("Saved plot to accuracy_vs_dimension.png")


if __name__ == "__main__":
    main()
