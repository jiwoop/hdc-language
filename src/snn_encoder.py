"""Spiking neural network baselines for gesture classification, using snnTorch.
    feedforward SNNs trained end to end.

Pipeline:
  1. Encoding -- either:
     - Rate coding (spikegen.rate): each channel's raw 315-sample real-valued series is
       min-max normalized into [0, 1] then used as per-timestep Bernoulli spike
       probability.
     - Delta-modulation coding (spikegen.delta): each channel's raw 315-sample series
       becomes a spike train of the same length, independently per channel (spikes at
       (t, channel) whenever |x[t] - x[t-1]| crosses `threshold`, signed via
       off_spike=True).
     Either way, uwave_data's (3, 315) arrays are transposed to (315, 3) first --
     both spikegen functions take time along dim 0.
  2. A small feedforward SNN: fc1 (Linear) -> lif1 -> fc2 (Linear) -> lif2, lif2 having
     8 output neurons (one per gesture class). lif1/lif2 are either snn.Leaky
     (1st-order LIF, membrane potential only) or snn.Synaptic (2nd-order LIF, adds a
     synaptic current state with its own decay `alpha`). Run once per timestep across
     all 315 steps, hidden state(s) carried across steps, output spikes recorded at
     every step -- the standard snntorch feedforward pattern (tutorial 3's
     architecture, extended with the training/classification tutorial 3 itself skips).
  3. Classification: rate decoding. Sum each class's output spikes over all 315
     timesteps; argmax of the 8 per-class counts is the prediction.
  4. Training: snntorch.functional.ce_rate_loss (cross-entropy over the per-timestep
     output spike rate), backprop through time via the ATan surrogate gradient
     (explicit spike_grad=surrogate.atan(), matching snnTorch's own default), Adam
     optimizer.
"""
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen
from snntorch import surrogate

import uwave_data as data

"""
Adjustable parameters.
"""
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_BETA = 0.9
DEFAULT_ALPHA = 0.9             # snn.Synaptic synaptic-current decay
DEFAULT_SPIKE_THRESHOLD = 0.9   # membrane firing threshold (Leaky and Synaptic)
DEFAULT_DELTA_THRESHOLD = 0.1   # spikegen.delta's per-step change threshold
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
NUM_CLASSES = len(data.CLASSES)


def _normalize_to_unit_interval(batch):
    """batch: (seriesLength, batch, n_channels) float32 tensor -> same shape, each
    (example, channel) trace min-max scaled into [0, 1] independently. Needed because
    the raw data is already z-scored (mean 0, std 1, range roughly [-4, 8]) and
    spikegen.rate's rate_conv clamps to [0, 1] before treating values as spike
    probabilities -- without this, ~70% of values get clipped to the same 0 or 1."""
    mins = batch.amin(dim=0, keepdim=True)
    maxs = batch.amax(dim=0, keepdim=True)
    span = (maxs - mins).clamp_min(1e-8)
    return (batch - mins) / span


"""examples: list of (n_channels, seriesLength) arrays -> float32 tensor of
    shape (seriesLength, batch, n_channels), spikes in {0, 1} per (t, channel)."""
def encode_rate_spike_trains(examples):
    batch = torch.stack([
        torch.from_numpy(ex.T.astype(np.float32)) for ex in examples
    ], dim=1)  # (seriesLength, batch, n_channels)
    batch = _normalize_to_unit_interval(batch)
    return spikegen.rate(batch, time_var_input=True)

def encode_delta_spike_trains(examples, delta_threshold=DEFAULT_DELTA_THRESHOLD):
    batch = torch.stack([
        torch.from_numpy(ex.T.astype(np.float32)) for ex in examples
    ], dim=1)  # (seriesLength, batch, n_channels)
    return spikegen.delta(batch, threshold=delta_threshold, off_spike=True)


class SNNGestureClassifierLeaky(nn.Module):
    """fc1 -> lif1 -> fc2 -> lif2 (1st-order LIF), lif2 = 8 output neurons (one per class)."""

    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 beta=DEFAULT_BETA, threshold=DEFAULT_SPIKE_THRESHOLD):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.atan())

    def forward(self, spk_in):
        """spk_in: (num_steps, batch, n_channels) -> spk2_rec: (num_steps, batch, num_classes)."""
        num_steps = spk_in.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)


class SNNGestureClassifierSynaptic(nn.Module):
    """fc1 -> lif1 -> fc2 -> lif2 (2nd-order LIF, snn.Synaptic), lif2 = 8 output
    neurons (one per class). Same shape as SNNGestureClassifierLeaky, but each
    neuron also carries a synaptic current state (decay `alpha`) alongside the
    membrane potential (decay `beta`)."""

    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, threshold=DEFAULT_SPIKE_THRESHOLD):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, hidden_size)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=surrogate.atan())

    def forward(self, spk_in):
        """spk_in: (num_steps, batch, n_channels) -> spk2_rec: (num_steps, batch, num_classes)."""
        num_steps = spk_in.shape[0]
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spk2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(spk_in[step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)


def _flatten_by_class(by_class):
    """{label: [examples]} -> (examples list, integer-label tensor), label order
    fixed by data.CLASSES ("1".."8" -> 0..7) so class indices match lif2's 8 outputs."""
    label_to_idx = {label: i for i, label in enumerate(data.CLASSES)}
    examples, labels = [], []
    for label in data.CLASSES:
        for ex in by_class[label]:
            examples.append(ex)
            labels.append(label_to_idx[label])
    return examples, torch.tensor(labels, dtype=torch.long)


def train_snn(train_by_class, encode_fn, model_cls=SNNGestureClassifierLeaky, model_kwargs=None,
              num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
              device="cpu", seed=42):
    """Trains a model_cls (SNNGestureClassifierLeaky or SNNGestureClassifierSynaptic)
    via SF.ce_rate_loss + Adam, encoding each batch with encode_fn (examples -> spk_in).
    Returns the trained model."""
    torch.manual_seed(seed)
    examples, labels = _flatten_by_class(train_by_class)
    labels = labels.to(device)

    model = model_cls(data.N_CHANNELS, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SF.ce_rate_loss()

    n = len(examples)
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_examples = [examples[i] for i in idx]
            batch_labels = labels[idx]

            spk_in = encode_fn(batch_examples).to(device)
            spk2_rec = model(spk_in)
            loss = loss_fn(spk2_rec, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        print(f"  epoch {epoch + 1:3d}/{num_epochs} -> loss={epoch_loss / n:.4f}", flush=True)

    return model


def evaluate_snn(model, test_by_class, encode_fn,
                  batch_size=DEFAULT_BATCH_SIZE, device="cpu", return_per_class=False):
    """Rate-decoded classification: argmax of each class's summed output-spike count
    over all timesteps, encoding each batch with encode_fn (examples -> spk_in). Mirrors
    evaluate_permutation/evaluate_gak_from_kernels's (accuracy[, per_class]) return shape."""
    model.eval()
    idx_to_label = {i: label for i, label in enumerate(data.CLASSES)}

    correct, total = 0, 0
    class_correct = {label: 0 for label in data.CLASSES}
    class_total = {label: 0 for label in data.CLASSES}

    with torch.no_grad():
        for label, exs in test_by_class.items():
            for start in range(0, len(exs), batch_size):
                batch_examples = exs[start:start + batch_size]
                spk_in = encode_fn(batch_examples).to(device)
                spk2_rec = model(spk_in)
                spike_counts = spk2_rec.sum(dim=0)  # (batch, num_classes)
                preds = spike_counts.argmax(dim=1).tolist()

                for pred_idx in preds:
                    is_correct = int(idx_to_label[pred_idx] == label)
                    correct += is_correct
                    total += 1
                    class_correct[label] += is_correct
                    class_total[label] += 1

    accuracy = correct / total
    per_class_accuracy = {
        label: (class_correct[label] / class_total[label] if class_total[label] else 0.0)
        for label in data.CLASSES
    }
    if return_per_class:
        return accuracy, per_class_accuracy
    return accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_by_class = data.load_split("TRAIN")
    test_by_class = data.load_split("TEST")

    model = train_snn(train_by_class, encode_rate_spike_trains,
                       model_kwargs={"hidden_size": DEFAULT_HIDDEN_SIZE},
                       device=device, num_epochs=10)
    accuracy, per_class = evaluate_snn(model, test_by_class, encode_rate_spike_trains,
                                        device=device, return_per_class=True)
    print(f"hidden_size={DEFAULT_HIDDEN_SIZE} -> accuracy={accuracy:.4f}")
    for label in sorted(per_class):
        print(f"  class {label}: {per_class[label]:.4f}")
