"""
Pure-numpy reimplementation of sparch's exact discrete-time LIF forward pass.

Purpose: confirm export_weights.py's extraction + BatchNorm folding is
correct, entirely decoupled from any PyNN/hardware translation. This
reproduces sparch/sparch/models/snns.py's LIFLayer._lif_cell and
ReadoutLayer._readout_cell math exactly, given the folded (W_eff, b_eff)
weights exported by sparch/export_weights.py.

No torch dependency -- only numpy.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def lif_layer_forward(x, W_eff, b_eff, alpha, threshold, rng):
    """
    x: (batch, time, input) binary spike train
    W_eff, b_eff: folded feedforward weights, shapes (hidden, input), (hidden,)
    alpha: (hidden,) decay factor, already clamped
    threshold: scalar
    rng: numpy Generator, used to match sparch's torch.rand(...) init of ut/st

    Returns spike train s: (batch, time, hidden)
    """
    batch, time, _ = x.shape
    hidden = W_eff.shape[0]

    Wx = x @ W_eff.T + b_eff  # (batch, time, hidden), all timesteps at once

    ut = rng.uniform(0, 1, size=(batch, hidden))
    st = rng.uniform(0, 1, size=(batch, hidden))
    s = np.zeros((batch, time, hidden), dtype=np.float64)

    for t in range(time):
        ut = alpha * (ut - st) + (1 - alpha) * Wx[:, t, :]
        st = (ut - threshold > 0).astype(np.float64)
        s[:, t, :] = st

    return s


def readout_layer_forward(x, W_eff, b_eff, alpha, rng):
    """
    x: (batch, time, input) spike train from last hidden layer
    Returns out: (batch, hidden) accumulated softmax, no time dim
    """
    batch, time, _ = x.shape
    hidden = W_eff.shape[0]

    Wx = x @ W_eff.T + b_eff

    ut = rng.uniform(0, 1, size=(batch, hidden))
    out = np.zeros((batch, hidden), dtype=np.float64)

    for t in range(time):
        ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
        out = out + softmax(ut, axis=1)

    return out


def load_weights(npz_path):
    data = np.load(npz_path)
    nb_layers = int(data["nb_layers"])
    layers = []
    for i in range(nb_layers):
        layers.append(
            {
                "W_eff": data[f"layer{i}_W_eff"],
                "b_eff": data[f"layer{i}_b_eff"],
                "alpha": data[f"layer{i}_alpha"],
                "threshold": float(data[f"layer{i}_threshold"]),
                "is_readout": bool(data[f"layer{i}_is_readout"]),
            }
        )
    return layers


def forward(x, layers, seed=0):
    """
    x: (batch, time, nb_inputs) binary spike train (e.g. SHD digitized input)
    layers: list of layer dicts from load_weights()

    Returns: (batch, nb_outputs) accumulated readout output (softmax-argmax
    over axis=1 gives sparch's predicted class).
    """
    rng = np.random.default_rng(seed)
    h = x
    for layer in layers:
        if layer["is_readout"]:
            h = readout_layer_forward(
                h, layer["W_eff"], layer["b_eff"], layer["alpha"], rng
            )
        else:
            h = lif_layer_forward(
                h,
                layer["W_eff"],
                layer["b_eff"],
                layer["alpha"],
                layer["threshold"],
                rng,
            )
    return h


def predict(x, layers, seed=0):
    out = forward(x, layers, seed=seed)
    return np.argmax(out, axis=1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <weights.npz>")
        sys.exit(1)

    layers = load_weights(sys.argv[1])
    print(f"Loaded {len(layers)} layers:")
    for i, layer in enumerate(layers):
        kind = "readout" if layer["is_readout"] else "LIF"
        print(f"  layer{i} ({kind}): W_eff {layer['W_eff'].shape}")

    # Smoke test on random binary input, just to confirm the math runs
    nb_inputs = layers[0]["W_eff"].shape[1]
    x = (np.random.default_rng(1).random((4, 100, nb_inputs)) < 0.05).astype(
        np.float64
    )
    preds = predict(x, layers)
    print(f"Predictions on random input: {preds}")
