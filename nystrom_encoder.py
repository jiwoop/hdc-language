"""Nystrom-approximated spectrum-kernel encoding for language recognition.

Ports the minimal pieces of NysHD (NysHD/Framework/NysHD/gen_kernel_matrix.py's k-mer
spectrum extractor and NysHD_train.py's Nystrom eigendecomposition + random-projection
encoding) needed for this project's 10-language text-classification task, reimplemented
in pure numpy rather than importing NysHD's own modules directly (those pull in torch,
grakel, mnist, and Bio.Seq at module scope purely for unrelated dataset loaders).

Classification stays on this repo's existing architecture (hdc_train.classify /
hdc_train.evaluate): one binary class hypervector per language, built by bundling many
Nystrom-encoded training chunks via majority vote, classified by (optionally
precision-limited) Hamming distance. Only the *encoding* step is Nystrom-based.
"""
import numpy as np

import hdc_train as t

LANDMARK_FRACTION = 0.02
MIN_LANDMARKS = 300


def kmer_spectrum(text, k, alphabet=t.ALPHABET, dtype=np.float64):
    """Count of each possible k-mer in `text`, using this project's alphabet.

    Each k-mer is treated as a base-len(alphabet) number (character -> its index in
    `alphabet`); windows containing a character outside `alphabet` are skipped, matching
    encoder.encode_text_to_hv's existing "any(char not in item_memory ...)" behavior.

    dtype defaults to float64 (matching original behavior); pass float32 to halve memory
    for larger k (e.g. k=4), where the dense spectrum vector itself is large.
    """
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    base = len(alphabet)
    spectrum = np.zeros(base ** k, dtype=dtype)
    powers = base ** np.arange(k - 1, -1, -1)

    for i in range(len(text) - k + 1):
        window = text[i:i + k]
        if any(char not in char_to_idx for char in window):
            continue
        digits = np.fromiter((char_to_idx[char] for char in window), dtype=np.int64, count=k)
        spectrum[int(np.dot(digits, powers))] += 1

    return spectrum


def kmer_spectrum_matrix(texts, k, alphabet=t.ALPHABET, dtype=np.float64):
    """Stack kmer_spectrum(text, k) for a list of texts -> shape (len(texts), len(alphabet)**k)."""
    return np.array([kmer_spectrum(text, k, alphabet, dtype) for text in texts], dtype=dtype)


def normalize_kernel_matrix(kernel_matrix, row_self_kernel, col_self_kernel):
    """Cosine-normalize a kernel matrix: K_hat(x, x') = K(x, x') / sqrt(K(x,x) * K(x',x')).

    Required for the Nystrom encoding's similarity-preservation guarantee (paper's Theorem 1,
    Eq. 4/9), which is stated for the *normalized* kernel, not the raw one. Without this, the
    raw spectrum-kernel dot product conflates k-mer-profile similarity with document length
    (a longer document has larger self-dot-products in every direction).

    row_self_kernel / col_self_kernel: 1D arrays of K(x,x) for the matrix's rows/columns
    (e.g. row_self_kernel = sum(features**2, axis=1) for the row-side feature matrix).
    """
    denom = np.sqrt(np.outer(row_self_kernel, col_self_kernel))
    denom[denom == 0] = 1e-15  # guard an all-zero (e.g. empty/out-of-alphabet) spectrum
    return kernel_matrix / denom


def kmer_kernel_against_landmarks(texts, landmark_features, landmark_self_kernel, k,
                                   alphabet=t.ALPHABET, dtype=np.float64, batch_size=None):
    """Normalized (len(texts) x num_landmark) kernel matrix between `texts` and the landmarks.

    batch_size=None (default): identical to computing kmer_spectrum_matrix(texts, k) once and
    dot-producting the whole thing against landmark_features -- one dense (len(texts) x 27**k)
    array is held in memory at once.

    batch_size=<int>: processes `texts` in slices of that size, computing and discarding each
    slice's (small) spectrum matrix immediately after it's been dot-producted against the
    landmarks, so peak memory is bounded by (batch_size x 27**k) rather than
    (len(texts) x 27**k). Only the small (len(texts) x num_landmark) kernel-matrix row-blocks
    accumulate across batches. Purely a memory-management choice -- produces identical values
    to batch_size=None for any batch size, including ones that don't evenly divide len(texts).
    """
    if batch_size is None:
        features = kmer_spectrum_matrix(texts, k, alphabet, dtype)
        self_kernel = np.sum(features * features, axis=1)
        kernel = features @ landmark_features.T
        return normalize_kernel_matrix(kernel, self_kernel, landmark_self_kernel)

    kernel_blocks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_features = kmer_spectrum_matrix(batch, k, alphabet, dtype)
        batch_self_kernel = np.sum(batch_features * batch_features, axis=1)
        batch_kernel = batch_features @ landmark_features.T
        kernel_blocks.append(normalize_kernel_matrix(batch_kernel, batch_self_kernel, landmark_self_kernel))

    return np.concatenate(kernel_blocks, axis=0)


def chunk_training_text_by_line(language):
    """Split a language's raw training file into per-line documents.

    Mirrors hdc_train.load_test_sentences's own line-splitting for the testing files,
    applied to the training files instead of hdc_train.load_training_text's single
    whitespace-collapsed blob -- gives many landmark-eligible documents per language
    instead of just one.
    """
    path = f"{t.DATA_DIR}/training/{language}.txt"
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [t.clean_text(line) for line in lines if line.strip()]


def sample_landmarks(chunks, num_landmark=None, seed=None):
    """Sample landmark documents from a pooled list of chunks.

    num_landmark defaults to NysHD's own recipe: max(2% of chunks, 300), capped at
    len(chunks).
    """
    if num_landmark is None:
        num_landmark = max(int(len(chunks) * LANDMARK_FRACTION), MIN_LANDMARKS)
    num_landmark = min(num_landmark, len(chunks))

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(chunks), size=num_landmark, replace=False)
    return [chunks[i] for i in indices]


def sample_landmarks_stratified(chunks_by_language, num_landmark=None, seed=None):
    """Sample landmarks with equal representation across languages, instead of
    sample_landmarks' single pooled uniform draw (which today happens to end up roughly
    balanced only because each language contributes a near-equal chunk count, not by design).

    Takes the pre-pooled per-language dict (not the flat list sample_landmarks takes), since
    stratifying requires knowing each chunk's language. Returns a flat list, same shape as
    sample_landmarks' return.

    Allocates num_landmark // len(languages) per language; any remainder
    (num_landmark % len(languages)) goes one-per-language to the first `remainder` languages
    in chunks_by_language's iteration order (deterministic: the same num_landmark always
    splits the same way; only which chunks are drawn within each language's share is
    seed-controlled).
    """
    languages = list(chunks_by_language.keys())
    if num_landmark is None:
        total_chunks = sum(len(chunks) for chunks in chunks_by_language.values())
        num_landmark = max(int(total_chunks * LANDMARK_FRACTION), MIN_LANDMARKS)

    base_count, remainder = divmod(num_landmark, len(languages))

    rng = np.random.default_rng(seed)
    landmarks = []
    for i, language in enumerate(languages):
        chunks = chunks_by_language[language]
        this_count = base_count + (1 if i < remainder else 0)
        this_count = min(this_count, len(chunks))
        indices = rng.choice(len(chunks), size=this_count, replace=False)
        landmarks.extend(chunks[idx] for idx in indices)
    return landmarks


def build_nystrom_encoding_matrix(landmark_kernel_matrix, dimension, seed=None):
    """Nystrom eigendecomposition + random projection, matching NysHD_train.py::encode.

    encoding_matrix has shape (dimension, num_landmark); a kernel-vs-landmark row vector
    k(x, landmarks) maps to HD space via `k(x, landmarks) @ encoding_matrix.T`.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(landmark_kernel_matrix)
    eigen_values = eigen_values.real
    eigen_values[eigen_values <= 0] = 1e-15
    eigen_vectors = eigen_vectors.real

    num_landmark = landmark_kernel_matrix.shape[0]
    diag = np.diag(1 / np.sqrt(eigen_values))

    rng = np.random.default_rng(seed)
    proj = rng.uniform(-1, 1, size=(dimension, num_landmark))
    proj_norm = proj / np.linalg.norm(proj, axis=1, keepdims=True)

    return proj_norm @ diag @ eigen_vectors.T


def nystrom_encode(kernel_matrix_rows, encoding_matrix, dimension):
    """Sign-random-projection binarization of the Nystrom feature map (NysHD_train.py::encode).

    Returns real-valued (bipolar-scaled) vectors of shape (num_examples, dimension); the
    sign is the meaningful part (see binarize()), the sqrt(pi/2)*sqrt(1/dimension) factor
    is a fixed positive scalar that estimates the original kernel's cosine similarity.
    """
    scale = np.sqrt(np.pi / 2) * np.sqrt(1 / dimension)
    return scale * np.sign(kernel_matrix_rows @ encoding_matrix.T)


def binarize(encoded):
    """Strip the positive scaling constant from nystrom_encode's output, keeping the sign,
    to match this repo's binary {0,1} hypervector convention (hdc.py/encoder.py)."""
    return (encoded > 0).astype(np.uint8)


def bundle(binary_hvs):
    """Majority-vote bundle of binary hypervectors into one, matching
    encoder.encode_text_to_hv's final two steps (bipolar-map, sum, threshold)."""
    bipolar = 2 * binary_hvs.astype(np.int32) - 1
    accumulator = bipolar.sum(axis=0)
    return (accumulator > 0).astype(np.uint8)


def train_class_vectors_nystrom(dimension, k, num_landmark=None, seed=42,
                                 dtype=np.float64, batch_size=None, landmark_sampler=None):
    """Build one Nystrom-encoded, bundled binary class hypervector per language.

    Returns (class_hvs, encoding_matrix, landmarks): class_hvs is a dict
    {language: binary_hv} with the same shape hdc_train.classify/evaluate already expect.
    encoding_matrix + landmarks are also returned since test-time encoding must be
    projected through the identical landmark-derived map.

    dtype/batch_size: forwarded to kmer_spectrum_matrix/kmer_kernel_against_landmarks;
    default to float64/None, reproducing original (unbatched, double-precision) behavior.

    landmark_sampler: None (default) -> today's sample_landmarks over the pooled all_chunks
    list, unchanged. Otherwise a callable (chunks_by_language, num_landmark, seed) -> flat
    landmark list, e.g. sample_landmarks_stratified, to switch sampling strategy.
    """
    chunks_by_language = {language: chunk_training_text_by_line(language) for language in t.LANGUAGES}

    if landmark_sampler is None:
        all_chunks = [chunk for chunks in chunks_by_language.values() for chunk in chunks]
        landmarks = sample_landmarks(all_chunks, num_landmark=num_landmark, seed=seed)
    else:
        landmarks = landmark_sampler(chunks_by_language, num_landmark, seed)

    landmark_features = kmer_spectrum_matrix(landmarks, k, dtype=dtype)
    landmark_self_kernel = np.sum(landmark_features * landmark_features, axis=1)

    landmark_kernel_matrix = landmark_features @ landmark_features.T
    landmark_kernel_matrix = normalize_kernel_matrix(landmark_kernel_matrix, landmark_self_kernel, landmark_self_kernel)

    encoding_matrix = build_nystrom_encoding_matrix(landmark_kernel_matrix, dimension, seed=seed)

    class_hvs = {}
    for language in t.LANGUAGES:
        chunk_kernel_matrix = kmer_kernel_against_landmarks(
            chunks_by_language[language], landmark_features, landmark_self_kernel, k,
            dtype=dtype, batch_size=batch_size)
        chunk_encodings = nystrom_encode(chunk_kernel_matrix, encoding_matrix, dimension)
        chunk_binary = binarize(chunk_encodings)
        class_hvs[language] = bundle(chunk_binary)

    return class_hvs, encoding_matrix, landmark_features, landmark_self_kernel


def evaluate_nystrom(class_hvs, encoding_matrix, landmark_features, landmark_self_kernel,
                      k, dimension, test_data, precision=None,
                      dtype=np.float64, batch_size=None, return_per_language=False):
    """Encode each test sentence via the same Nystrom map, classify via hdc_train.classify,
    and report accuracy. Mirrors hdc_train.evaluate's loop/accounting.

    return_per_language=False (default): returns a single aggregate accuracy float, as before.
    return_per_language=True: additionally returns a {language: accuracy} dict.
    """
    correct, total = 0, 0
    per_language_accuracy = {}
    for true_language, sentences in test_data.items():
        if not sentences:
            continue
        sentence_kernel_matrix = kmer_kernel_against_landmarks(
            sentences, landmark_features, landmark_self_kernel, k,
            dtype=dtype, batch_size=batch_size)
        sentence_encodings = nystrom_encode(sentence_kernel_matrix, encoding_matrix, dimension)
        sentence_binary = binarize(sentence_encodings)

        lang_correct = 0
        for sentence_hv in sentence_binary:
            is_correct = int(t.classify(sentence_hv, class_hvs, precision=precision) == true_language)
            correct += is_correct
            lang_correct += is_correct
            total += 1
        per_language_accuracy[true_language] = lang_correct / len(sentences)

    accuracy = correct / total
    if return_per_language:
        return accuracy, per_language_accuracy
    return accuracy


def run_nystrom_experiment(args):
    """args = (dimension, k, test_data). Returns (dimension, k, accuracy)."""
    dimension, k, test_data = args
    class_hvs, encoding_matrix, landmark_features, landmark_self_kernel = train_class_vectors_nystrom(dimension, k)
    accuracy = evaluate_nystrom(class_hvs, encoding_matrix, landmark_features, landmark_self_kernel, k, dimension, test_data)
    print(f"D={dimension:5d} k={k} -> accuracy={accuracy:.4f}")
    return dimension, k, accuracy


def run_nystrom_precision_experiment(args):
    """args = (precision, dimension, k, test_data). Returns (precision, dimension, k, accuracy).
    Class hypervectors don't depend on precision -- only the classification-time distance
    metric does (same as hdc_train.run_hamming_precision_experiment)."""
    precision, dimension, k, test_data = args
    class_hvs, encoding_matrix, landmark_features, landmark_self_kernel = train_class_vectors_nystrom(dimension, k)
    accuracy = evaluate_nystrom(class_hvs, encoding_matrix, landmark_features, landmark_self_kernel,
                                 k, dimension, test_data, precision=precision)
    print(f"D={dimension:5d} k={k} precision={precision:3d} -> accuracy={accuracy:.4f}")
    return precision, dimension, k, accuracy


def train_class_vectors_nystrom_ensemble(total_dimension, num_encoders, k, num_landmark=None, seed=42):
    """Build `num_encoders` independent Nystrom encoders, each at dimension
    (total_dimension // num_encoders), sharing ONE landmark kernel eigendecomposition (the
    expensive step) -- only the cheap random-projection step (build_nystrom_encoding_matrix)
    repeats per encoder, each with a different seed.

    Tests whether splitting a fixed total bit budget into several independent narrower
    hypervectors (each voting on the classification) can approach a single wide hypervector's
    accuracy -- see the ensembling section of the notebook for the full rationale.

    Returns a list of length num_encoders, each element a tuple
    (class_hvs, encoding_matrix, landmark_features, landmark_self_kernel) with the same shape
    train_class_vectors_nystrom returns, so each can be independently evaluated via
    evaluate_nystrom or classified via hdc_train.classify.
    """
    encoder_dimension = total_dimension // num_encoders

    chunks_by_language = {language: chunk_training_text_by_line(language) for language in t.LANGUAGES}
    all_chunks = [chunk for chunks in chunks_by_language.values() for chunk in chunks]
    landmarks = sample_landmarks(all_chunks, num_landmark=num_landmark, seed=seed)
    landmark_features = kmer_spectrum_matrix(landmarks, k)
    landmark_self_kernel = np.sum(landmark_features * landmark_features, axis=1)

    landmark_kernel_matrix = landmark_features @ landmark_features.T
    landmark_kernel_matrix = normalize_kernel_matrix(landmark_kernel_matrix, landmark_self_kernel, landmark_self_kernel)

    encoders = []
    for i in range(num_encoders):
        encoding_matrix = build_nystrom_encoding_matrix(landmark_kernel_matrix, encoder_dimension, seed=seed + i)

        class_hvs = {}
        for language in t.LANGUAGES:
            chunk_kernel_matrix = kmer_kernel_against_landmarks(
                chunks_by_language[language], landmark_features, landmark_self_kernel, k)
            chunk_encodings = nystrom_encode(chunk_kernel_matrix, encoding_matrix, encoder_dimension)
            chunk_binary = binarize(chunk_encodings)
            class_hvs[language] = bundle(chunk_binary)

        encoders.append((class_hvs, encoding_matrix, landmark_features, landmark_self_kernel))

    return encoders


def classify_ensemble(sentence_kernel_row, encoders, encoder_dimension, precision=None):
    """Classify one test example by majority vote across `encoders`' independent predictions.

    sentence_kernel_row: this example's (1 x num_landmark) kernel row against the SHARED
    landmark_features every encoder in `encoders` was built from (all encoders share the same
    landmark_features/landmark_self_kernel, so this is computed once, not once per encoder).

    Ties are broken deterministically: the language with the most votes wins; if tied, the
    first such language in hdc_train.LANGUAGES's fixed order is chosen.
    """
    votes = {}
    for class_hvs, encoding_matrix, _, _ in encoders:
        encoded = nystrom_encode(sentence_kernel_row, encoding_matrix, encoder_dimension)
        binary = binarize(encoded)[0]
        predicted = t.classify(binary, class_hvs, precision=precision)
        votes[predicted] = votes.get(predicted, 0) + 1

    best_count = max(votes.values())
    for language in t.LANGUAGES:
        if votes.get(language) == best_count:
            return language


def evaluate_nystrom_ensemble(encoders, k, encoder_dimension, test_data, precision=None):
    """Evaluate an ensemble of encoders (as returned by train_class_vectors_nystrom_ensemble)
    via majority-vote classification. Mirrors evaluate_nystrom's loop/accounting."""
    _, _, landmark_features, landmark_self_kernel = encoders[0]

    correct, total = 0, 0
    for true_language, sentences in test_data.items():
        if not sentences:
            continue
        sentence_kernel_matrix = kmer_kernel_against_landmarks(
            sentences, landmark_features, landmark_self_kernel, k)

        for row in sentence_kernel_matrix:
            predicted = classify_ensemble(row[np.newaxis, :], encoders, encoder_dimension, precision=precision)
            correct += int(predicted == true_language)
            total += 1

    return correct / total
