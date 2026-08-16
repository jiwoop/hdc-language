"""
Tier-2 verification: builds the network via src/pynn_lif_model.py against a
LOCAL PyNN backend (pyNN.nest) and reports the spike-count-argmax vs.
softmax-argmax agreement rate on a handful of real SHD test examples --
comparing the PyNN/NEST simulation against the tier-1 numpy reference
(verification/reference_lif_numpy.py), which is itself already confirmed to
match sparch's real PyTorch output exactly (see sparch/verify_export.py).

Skips gracefully (prints a message, exits 0) if pyNN or pyNN.nest aren't
importable -- this must never hard-fail or block the rest of the pipeline.

Usage (from hdc-language repo root, hdc-env conda env):
    python verification/pynn_smoke_test.py [--weights PATH] [--data PATH]
        [--n_examples N]
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.pynn_lif_model import DT_MS  # noqa: E402
from src.pynn_lif_model import SIM_TIMESTEP_MS  # noqa: E402
from src.pynn_lif_model import build_network  # noqa: E402
from src.pynn_lif_model import load_weights  # noqa: E402
from src.pynn_lif_model import spike_count_argmax  # noqa: E402
from src.pynn_lif_model import spike_times_to_pynn  # noqa: E402
from verification.reference_lif_numpy import load_weights as load_weights_ref  # noqa: E402
from verification.reference_lif_numpy import predict as predict_ref  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=os.path.join(REPO_ROOT, "src", "data", "shd_lif_weights.npz"),
    )
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(REPO_ROOT), "sparch", "shd_dataset", "shd_test.h5"
        ),
        help="Path to sparch's shd_test.h5 (raw spike times, read directly "
        "-- no sparch/torch import needed).",
    )
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, randomly sample --n_examples from the full dataset "
        "instead of taking the first N in file order (SHD's test.h5 is "
        "sorted by label, so the first N are a skewed, unrepresentative "
        "slice unless shuffled).",
    )
    return parser.parse_args()


def load_shd_examples(h5_path, n_examples, seed=None):
    import h5py

    f = h5py.File(h5_path, "r")
    times = f["spikes"]["times"]
    units = f["spikes"]["units"]
    labels = np.array(f["labels"])

    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(labels), size=n_examples, replace=False)
    else:
        indices = range(n_examples)

    examples = []
    for i in indices:
        examples.append((np.array(times[i]), np.array(units[i]), int(labels[i])))
    return examples


def dense_input_from_spikes(times_sec, units, nb_units, nb_steps=100, max_time=1.4):
    """Reproduces sparch's SpikingDataset.__getitem__ binning (np.digitize
    into nb_steps bins over max_time), for feeding the numpy reference
    module (which expects a dense (time, nb_units) array, same as sparch's
    own dataloader produces).

    sparch builds this via torch.sparse.FloatTensor(...).to_dense(), which
    *sums* values at duplicate (time, unit) indices rather than overwriting
    them -- multiple raw spikes commonly land in the same digitized bin for
    the same input channel (SHD's spike timing is much finer than the
    100-step binning), so plain `x[idx] = 1.0` silently clamps those cells
    to 1 and understates the input current. Use np.add.at to match sparch's
    accumulate-on-collision semantics exactly."""
    time_bins = np.linspace(0, max_time, num=nb_steps)
    time_idx = np.digitize(times_sec, time_bins)
    x = np.zeros((nb_steps, nb_units), dtype=np.float64)
    valid = time_idx < nb_steps
    np.add.at(x, (time_idx[valid], units[valid]), 1.0)
    return x


def main():
    args = parse_args()

    try:
        import pyNN.nest as pynn
    except ImportError as e:
        print(f"pyNN/NEST not importable ({e}) -- skipping tier-2 verification.")
        sys.exit(0)

    layers_pynn = load_weights(args.weights)
    layers_ref = load_weights_ref(args.weights)
    nb_inputs = layers_pynn[0]["W_eff"].shape[1]
    nb_outputs = layers_pynn[-1]["W_eff"].shape[0]

    examples = load_shd_examples(args.data, args.n_examples, seed=args.seed)

    agree = 0
    ref_correct = 0
    pynn_correct = 0
    for i, (times_sec, units, label) in enumerate(examples):
        duration_ms = DT_MS * 100  # matches sparch's nb_steps=100 window

        pynn.setup(timestep=SIM_TIMESTEP_MS, min_delay=SIM_TIMESTEP_MS)
        input_spike_times = spike_times_to_pynn(times_sec, units, nb_inputs)
        populations, projections, readout_pop = build_network(
            pynn, layers_pynn, input_spike_times
        )
        pynn.run(duration_ms)
        pynn_pred, spike_counts = spike_count_argmax(readout_pop, nb_outputs)
        pynn.end()

        x_dense = dense_input_from_spikes(times_sec, units, nb_inputs)[None, :, :]
        ref_pred = int(predict_ref(x_dense, layers_ref)[0])

        match = pynn_pred == ref_pred
        agree += match
        ref_correct += ref_pred == label
        pynn_correct += pynn_pred == label
        print(
            f"example {i}: label={label} ref(softmax-argmax)_pred={ref_pred} "
            f"pynn(spike-count-argmax)_pred={pynn_pred} match={match} "
            f"spike_counts={spike_counts.tolist()}"
        )

    n = len(examples)
    print(f"\nAgreement (spike-count-argmax vs softmax-argmax): {agree}/{n}")
    print(f"Reference (sparch/numpy) accuracy vs true label: {ref_correct}/{n}")
    print(f"PyNN (spike-count-argmax) accuracy vs true label: {pynn_correct}/{n}")


if __name__ == "__main__":
    main()
