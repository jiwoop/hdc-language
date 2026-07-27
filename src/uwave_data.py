"""Loader for the UWaveGestureLibraryAll sktime .ts files.

UWaveGestureLibraryAll_{TRAIN,TEST}.ts stores each gesture as one univariate,
945-length series that is the X, Y, Z accelerometer channels concatenated end
to end (confirmed by the file's own header comment: "The files are the X, Y
and Z dimensions. UWaveGestureLibraryAll is the concatenation of all three."
-- @univariate true, @seriesLength 945). This module undoes that concatenation,
reshaping each row back into (3 channels, 315 samples) so it can be fed to a
genuinely multi-channel encoder (permutation baseline or GAK).

Each data row has the form "v1,v2,...,v945:label" -- comma-separated values,
then a colon, then the trailing class label (1..8). This is NOT pure
comma-separated like the sibling .txt file.
"""
import os

import numpy as np

DATA_DIR = "UWaveGestureLibraryAll"
N_CHANNELS = 3
SERIES_LENGTH = 315
CLASSES = [str(i) for i in range(1, 9)]


def _parse_ts_file(path):
    """Parse one .ts file -> list of (channels, label) pairs.

    channels: float64 array of shape (N_CHANNELS, SERIES_LENGTH).
    label: the class label string, exactly as stored (e.g. "1".."8").
    """
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data_start = next(i for i, line in enumerate(lines) if line.startswith('@data')) + 1

    examples = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        series_str, label = line.rsplit(':', 1)
        values = np.array([float(v) for v in series_str.split(',')], dtype=np.float64)
        if values.size != N_CHANNELS * SERIES_LENGTH:
            raise ValueError(
                f"{path}: expected {N_CHANNELS * SERIES_LENGTH} values, got {values.size}")
        channels = values.reshape(N_CHANNELS, SERIES_LENGTH)
        examples.append((channels, label))

    return examples


def load_split(split, data_dir=DATA_DIR):
    """split: "TRAIN" or "TEST" -> {label: [ (N_CHANNELS, SERIES_LENGTH) array, ... ]}."""
    path = os.path.join(data_dir, f"UWaveGestureLibraryAll_{split}.ts")
    examples = _parse_ts_file(path)

    by_class = {label: [] for label in CLASSES}
    for channels, label in examples:
        by_class[label].append(channels)
    return by_class


def load_train(data_dir=DATA_DIR):
    return load_split("TRAIN", data_dir)


def load_test(data_dir=DATA_DIR):
    return load_split("TEST", data_dir)


if __name__ == "__main__":
    train = load_train()
    test = load_test()

    n_train = sum(len(v) for v in train.values())
    n_test = sum(len(v) for v in test.values())
    print(f"train: {n_train} examples across {len(train)} classes")
    for label in CLASSES:
        print(f"  class {label}: {len(train[label])} train, {len(test[label])} test")
    print(f"test: {n_test} examples across {len(test)} classes")

    sample = train[CLASSES[0]][0]
    print(f"sample shape: {sample.shape} (expected ({N_CHANNELS}, {SERIES_LENGTH}))")
