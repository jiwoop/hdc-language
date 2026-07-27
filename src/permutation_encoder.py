"""Multi-channel permutation-based HDC encoding for gesture time series.

Generalizes encoder.py's character n-gram encoding (bind consecutive
permuted item-memory vectors, stream into a bipolar accumulator) from a 1D
discrete character sequence to a multi-channel continuous signal:

  1. Quantize each channel's real-valued samples into `num_levels` discrete
     levels, using a *correlated* level item memory (each level differs from
     the previous one by a small fraction of flipped bits) rather than the
     fully independent per-symbol vectors generate_item_memory produces --
     nearby levels need to be close in Hamming distance since, unlike
     characters, quantization levels are numerically ordered.
  2. At each timestep, bind each channel's level vector with that channel's
     ID vector, then bundle (majority vote) across channels into one
     per-timestep hypervector.
  3. Slide an n_gram_size window of per-timestep hypervectors, bind each with
     a permute()-shifted copy keyed by its position in the window (exactly
     encoder.encode_text_to_hv's pattern), and stream into a bipolar
     accumulator over the whole example -> one hypervector per example.
  4. Bundle every training example's hypervector, per class, into one class
     hypervector. Classification reuses hdc_train.classify unchanged.
"""
import numpy as np

import hdc
import hdc_train as t
import uwave_data as data

DEFAULT_NUM_LEVELS = 100
DEFAULT_N_GRAM_SIZE = 5
LEVEL_FLIP_FRACTION = 0.02  # fraction of bits flipped between adjacent levels


def generate_channel_memory(n_channels, dimension, seed=None):
    """One independent random binary hypervector per channel (e.g. X, Y, Z)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_channels, dimension), dtype=np.uint8)


def generate_level_memory(num_levels, dimension, seed=None,
                           flip_fraction=LEVEL_FLIP_FRACTION):
    """Correlated ("thermometer-style") level hypervectors: level 0 is a random
    base vector; each subsequent level flips a small fraction of the previous
    level's bits, so Hamming distance grows roughly with the numeric gap
    between levels (unlike independently-random symbol vectors)."""
    rng = np.random.default_rng(seed)
    n_flip = max(1, int(round(dimension * flip_fraction)))

    levels = np.empty((num_levels, dimension), dtype=np.uint8)
    levels[0] = rng.integers(0, 2, size=dimension, dtype=np.uint8)
    for i in range(1, num_levels):
        levels[i] = levels[i - 1].copy()
        flip_idx = rng.choice(dimension, size=n_flip, replace=False)
        levels[i, flip_idx] ^= 1
    return levels


def fit_quantizer(train_examples, num_levels):
    """Compute per-channel-pooled [min, max] over training data, returning
    (lo, hi) scalars used to bin every channel's values into num_levels bins.
    Fixed at train time, applied unchanged to test data."""
    all_values = np.concatenate([ex.ravel() for ex in train_examples])
    return float(all_values.min()), float(all_values.max())


def quantize(values, lo, hi, num_levels):
    """Map real values -> integer level indices in [0, num_levels)."""
    if hi <= lo:
        return np.zeros_like(values, dtype=np.int64)
    scaled = (values - lo) / (hi - lo)
    idx = np.floor(scaled * num_levels).astype(np.int64)
    return np.clip(idx, 0, num_levels - 1)


def encode_example_to_hv(channels, channel_memory, level_memory, lo, hi,
                          num_levels, n_gram_size, dimension):
    """channels: (n_channels, seriesLength) real-valued array -> one binary
    hypervector of length `dimension`."""
    n_channels, series_length = channels.shape
    level_idx = quantize(channels, lo, hi, num_levels)  # (n_channels, series_length)

    # Per-timestep hypervector: bundle across channels of bind(level_hv, channel_hv).
    timestep_hvs = np.empty((series_length, dimension), dtype=np.uint8)
    for tstep in range(series_length):
        bound = np.empty((n_channels, dimension), dtype=np.uint8)
        for c in range(n_channels):
            bound[c] = hdc.bind(level_memory[level_idx[c, tstep]], channel_memory[c])
        bipolar = 2 * bound.astype(np.int32) - 1
        timestep_hvs[tstep] = (bipolar.sum(axis=0) > 0).astype(np.uint8)

    # N-gram over time: bind permuted window members, stream into accumulator.
    accumulator = np.zeros(dimension, dtype=np.int32)
    for i in range(series_length - n_gram_size + 1):
        window = timestep_hvs[i:i + n_gram_size]
        ngram_hv = window[0]
        for pos in range(1, n_gram_size):
            permuted_hv = hdc.permute(window[pos], steps=pos)
            ngram_hv = hdc.bind(ngram_hv, permuted_hv)
        accumulator += 2 * ngram_hv.astype(np.int32) - 1

    return (accumulator > 0).astype(np.uint8)


def train_class_vectors_permutation(dimension, num_levels=DEFAULT_NUM_LEVELS,
                                     n_gram_size=DEFAULT_N_GRAM_SIZE, seed=42,
                                     data_dir=data.DATA_DIR):
    """Build one bundled binary class hypervector per gesture class.

    Returns (class_hvs, channel_memory, level_memory, lo, hi) -- the memories
    and quantization range must be reused unchanged at test time.
    """
    train_by_class = data.load_split("TRAIN", data_dir)
    all_train = [ex for exs in train_by_class.values() for ex in exs]

    channel_memory = generate_channel_memory(data.N_CHANNELS, dimension, seed=seed)
    level_memory = generate_level_memory(num_levels, dimension, seed=seed + 1)
    lo, hi = fit_quantizer(all_train, num_levels)

    class_hvs = {}
    for label, examples in train_by_class.items():
        example_hvs = np.array([
            encode_example_to_hv(ex, channel_memory, level_memory, lo, hi,
                                  num_levels, n_gram_size, dimension)
            for ex in examples
        ], dtype=np.uint8)
        bipolar = 2 * example_hvs.astype(np.int32) - 1
        class_hvs[label] = (bipolar.sum(axis=0) > 0).astype(np.uint8)

    return class_hvs, channel_memory, level_memory, lo, hi


def evaluate_permutation(class_hvs, channel_memory, level_memory, lo, hi,
                          num_levels, n_gram_size, dimension, test_by_class,
                          precision=None, return_per_class=False):
    """Encode each test example via the same memories/quantizer, classify via
    hdc_train.classify, report accuracy. Mirrors nystrom_encoder.evaluate_nystrom."""
    correct, total = 0, 0
    per_class_accuracy = {}
    for true_label, examples in test_by_class.items():
        if not examples:
            continue
        class_correct = 0
        for ex in examples:
            example_hv = encode_example_to_hv(
                ex, channel_memory, level_memory, lo, hi, num_levels, n_gram_size, dimension)
            is_correct = int(t.classify(example_hv, class_hvs, precision=precision) == true_label)
            correct += is_correct
            class_correct += is_correct
            total += 1
        per_class_accuracy[true_label] = class_correct / len(examples)

    accuracy = correct / total
    if return_per_class:
        return accuracy, per_class_accuracy
    return accuracy


if __name__ == "__main__":
    dimension = 1024
    class_hvs, channel_memory, level_memory, lo, hi = train_class_vectors_permutation(dimension)
    test_by_class = data.load_split("TEST")
    accuracy, per_class = evaluate_permutation(
        class_hvs, channel_memory, level_memory, lo, hi,
        DEFAULT_NUM_LEVELS, DEFAULT_N_GRAM_SIZE, dimension, test_by_class,
        return_per_class=True)
    print(f"D={dimension} -> accuracy={accuracy:.4f}")
    for label in sorted(per_class):
        print(f"  class {label}: {per_class[label]:.4f}")
