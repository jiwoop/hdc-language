"""Nystrom-approximated Global Alignment Kernel (GAK) encoding for gesture time series.

Mirrors nystrom_encoder.py's structure exactly, replacing only its string/k-mer
kernel top layer (kmer_spectrum / kmer_kernel_against_landmarks) with a GAK
Gram matrix computed via tslearn.metrics.cdist_gak over multi-channel gesture
examples. Everything from the eigendecomposition onward -- build_nystrom_encoding
_matrix, nystrom_encode, binarize, bundle, and landmark sampling -- is reused
unchanged by importing from nystrom_encoder.py; there is no string-specific
code in that half of the pipeline to begin with.

Unlike the spectrum kernel (a raw dot product needing cosine normalization via
nystrom_encoder.normalize_kernel_matrix), tslearn's cdist_gak already returns a
self-normalized kernel -- K(x, x) == 1 for every x -- so no separate
normalization step is needed here.
"""
import numpy as np
from tslearn.metrics import cdist_gak, sigma_gak

import hdc_train as t
import nystrom_encoder as ne
import uwave_data as data

DEFAULT_NUM_LANDMARK = 100


def _to_tslearn_shape(examples):
    """(n_channels, seriesLength) per-example arrays -> tslearn's expected
    (n_samples, seriesLength, n_channels) stacked array."""
    return np.stack([ex.T for ex in examples])


def estimate_gak_sigma(examples, n_samples=100, seed=42):
    """tslearn's own bandwidth heuristic (median-ish pairwise distance), estimated
    from a subsample of `examples` for speed. Fixed once at train time and reused
    for every later kernel computation (train, landmark, and test-time)."""
    X = _to_tslearn_shape(examples)
    return sigma_gak(X, n_samples=min(n_samples, len(examples)), random_state=seed)


def gak_kernel_against_landmarks(examples, landmarks, sigma):
    """Normalized (len(examples) x len(landmarks)) GAK Gram matrix.

    cdist_gak self-normalizes internally (diagonal of a self-Gram matrix is
    always 1), so no separate cosine-normalization step is required here,
    unlike kmer_kernel_against_landmarks's raw spectrum dot product.
    """
    X = _to_tslearn_shape(examples)
    L = _to_tslearn_shape(landmarks)
    return cdist_gak(X, L, sigma=sigma)


def build_gak_landmark_kernels(num_landmark=None, seed=42, landmark_sampler=None,
                                data_dir=data.DATA_DIR):
    """Compute the expensive, dimension-independent part of Nystrom+GAK once:
    landmark sampling, sigma estimation, and every example's GAK kernel row against
    the landmarks (train examples by class + full test set). This is the part that
    actually costs O(examples x landmarks) GAK evaluations -- everything downstream
    (per-dimension eigendecomposition + random projection) is cheap linear algebra,
    so callers sweeping many dimensions at one landmark count should call this ONCE
    and reuse its return value across every dimension via train_class_vectors_gak_from_kernels.

    Returns a dict: {landmarks, sigma, landmark_kernel_matrix,
    train_kernel_by_class: {label: (n_examples, num_landmark) matrix},
    test_kernel_by_class: {label: (n_examples, num_landmark) matrix}}.
    """
    examples_by_class = data.load_split("TRAIN", data_dir)
    test_by_class = data.load_split("TEST", data_dir)

    if landmark_sampler is None:
        all_examples = [ex for exs in examples_by_class.values() for ex in exs]
        landmarks = ne.sample_landmarks(all_examples, num_landmark=num_landmark, seed=seed)
    else:
        landmarks = landmark_sampler(examples_by_class, num_landmark, seed)

    sigma = estimate_gak_sigma(landmarks, seed=seed)
    landmark_kernel_matrix = gak_kernel_against_landmarks(landmarks, landmarks, sigma)

    train_kernel_by_class = {
        label: gak_kernel_against_landmarks(examples, landmarks, sigma)
        for label, examples in examples_by_class.items()
    }
    test_kernel_by_class = {
        label: gak_kernel_against_landmarks(examples, landmarks, sigma)
        for label, examples in test_by_class.items() if examples
    }

    return {
        "landmarks": landmarks, "sigma": sigma,
        "landmark_kernel_matrix": landmark_kernel_matrix,
        "train_kernel_by_class": train_kernel_by_class,
        "test_kernel_by_class": test_kernel_by_class,
    }


def train_class_vectors_gak_from_kernels(kernels, dimension, seed=42):
    """Cheap per-dimension step: eigendecompose + random-project the already-computed
    landmark kernel matrix, encode every training example's already-computed kernel
    row, bundle into class hypervectors. Returns (class_hvs, encoding_matrix)."""
    encoding_matrix = ne.build_nystrom_encoding_matrix(
        kernels["landmark_kernel_matrix"], dimension, seed=seed)

    class_hvs = {}
    for label, kernel_matrix in kernels["train_kernel_by_class"].items():
        example_encodings = ne.nystrom_encode(kernel_matrix, encoding_matrix, dimension)
        example_binary = ne.binarize(example_encodings)
        class_hvs[label] = ne.bundle(example_binary)

    return class_hvs, encoding_matrix


def evaluate_gak_from_kernels(class_hvs, encoding_matrix, kernels, dimension,
                               precision=None, return_per_class=False):
    """Classify every test example using its already-computed GAK kernel row (from
    build_gak_landmark_kernels) -- no GAK re-computation at evaluation time."""
    correct, total = 0, 0
    per_class_accuracy = {}
    for true_label, kernel_matrix in kernels["test_kernel_by_class"].items():
        example_encodings = ne.nystrom_encode(kernel_matrix, encoding_matrix, dimension)
        example_binary = ne.binarize(example_encodings)

        class_correct = 0
        for example_hv in example_binary:
            is_correct = int(t.classify(example_hv, class_hvs, precision=precision) == true_label)
            correct += is_correct
            class_correct += is_correct
            total += 1
        per_class_accuracy[true_label] = class_correct / len(kernel_matrix)

    accuracy = correct / total
    if return_per_class:
        return accuracy, per_class_accuracy
    return accuracy


def train_class_vectors_gak(dimension, num_landmark=None, seed=42,
                             landmark_sampler=None, data_dir=data.DATA_DIR):
    """Convenience one-shot wrapper (computes kernels AND trains at a single dimension).
    For sweeping many dimensions at one landmark count, prefer
    build_gak_landmark_kernels + train_class_vectors_gak_from_kernels directly to avoid
    recomputing the expensive GAK kernels per dimension.

    Returns (class_hvs, encoding_matrix, kernels) -- kernels must be reused (via
    evaluate_gak_from_kernels) for test-time evaluation.
    """
    kernels = build_gak_landmark_kernels(num_landmark=num_landmark, seed=seed,
                                          landmark_sampler=landmark_sampler, data_dir=data_dir)
    class_hvs, encoding_matrix = train_class_vectors_gak_from_kernels(kernels, dimension, seed=seed)
    return class_hvs, encoding_matrix, kernels


def evaluate_gak(class_hvs, encoding_matrix, kernels, dimension,
                  precision=None, return_per_class=False):
    """One-shot wrapper around evaluate_gak_from_kernels, matching train_class_vectors_gak's
    (class_hvs, encoding_matrix, kernels) return shape."""
    return evaluate_gak_from_kernels(class_hvs, encoding_matrix, kernels, dimension,
                                      precision=precision, return_per_class=return_per_class)


def sample_landmarks_stratified_gesture(examples_by_class, num_landmark=None, seed=None):
    """Thin wrapper: ne.sample_landmarks_stratified is already generic over any
    {class: chunks} dict, so this just documents the gesture-specific call site."""
    return ne.sample_landmarks_stratified(examples_by_class, num_landmark=num_landmark, seed=seed)


if __name__ == "__main__":
    dimension = 1024
    class_hvs, encoding_matrix, kernels = train_class_vectors_gak(
        dimension, num_landmark=DEFAULT_NUM_LANDMARK)
    accuracy, per_class = evaluate_gak(
        class_hvs, encoding_matrix, kernels, dimension, return_per_class=True)
    print(f"D={dimension} num_landmark={DEFAULT_NUM_LANDMARK} -> accuracy={accuracy:.4f}")
    for label in sorted(per_class):
        print(f"  class {label}: {per_class[label]:.4f}")
