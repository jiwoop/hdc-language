"""Spiking neural network baseline for gesture classification, using snnTorch.

A third point of comparison alongside the two HDC encoders (permutation_encoder.py,
gak_encoder.py) on the same UWaveGestureLibrary train/test split and accuracy metric --
not an HDC encoding at all, but a small feedforward SNN trained end to end.

Pipeline:
  1. Delta-modulation encoding (spikegen.delta): each channel's raw 315-sample real-
     valued series becomes a spike train of the same length, independently per channel
     (spikes at (t, channel) whenever |x[t] - x[t-1]| crosses `threshold`, signed via
     off_spike=True). uwave_data's (3, 315) arrays are transposed to (315, 3) first --
     spikegen.delta takes its deltas along dim 0 (time).
  2. A small feedforward SNN: fc1 (Linear) -> lif1 (Leaky) -> fc2 (Linear) -> lif2
     (Leaky), lif2 having 8 output neurons (one per gesture class). Run once per
     timestep across all 315 steps, membrane potentials carried across steps, output
     spikes recorded at every step -- the standard snntorch feedforward pattern
     (tutorial 3's architecture, extended with the training/classification tutorial 3
     itself skips).
  3. Classification: rate coding. Sum each class's output spikes over all 315
     timesteps; argmax of the 8 per-class counts is the prediction.
  4. Training: snntorch.functional.ce_rate_loss (cross-entropy over the per-timestep
     output spike rate), backprop through time via snn.Leaky's default ATan surrogate
     gradient, Adam optimizer.
"""
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen

import uwave_data as data

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_BETA = 0.9
DEFAULT_SPIKE_THRESHOLD = 0.9   # snn.Leaky membrane firing threshold
DEFAULT_DELTA_THRESHOLD = 0.1   # spikegen.delta's per-step change threshold
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
NUM_CLASSES = len(data.CLASSES)


def encode_spike_trains(examples, delta_threshold=DEFAULT_DELTA_THRESHOLD):
    """examples: list of (n_channels, seriesLength) arrays -> float32 tensor of
    shape (seriesLength, batch, n_channels), spikes in {-1, 0, 1} per (t, channel)."""
    batch = torch.stack([
        torch.from_numpy(ex.T.astype(np.float32)) for ex in examples
    ], dim=1)  # (seriesLength, batch, n_channels)
    return spikegen.delta(batch, threshold=delta_threshold, off_spike=True)


class SNNGestureClassifier(nn.Module):
    """fc1 -> lif1 -> fc2 -> lif2, lif2 = 8 output neurons (one per class)."""

    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 beta=DEFAULT_BETA, threshold=DEFAULT_SPIKE_THRESHOLD):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

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


def train_snn(train_by_class, hidden_size=DEFAULT_HIDDEN_SIZE, beta=DEFAULT_BETA,
              spike_threshold=DEFAULT_SPIKE_THRESHOLD, delta_threshold=DEFAULT_DELTA_THRESHOLD,
              num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
              device="cpu", seed=42):
    """Trains an SNNGestureClassifier via SF.ce_rate_loss + Adam. Returns the trained model."""
    torch.manual_seed(seed)
    examples, labels = _flatten_by_class(train_by_class)
    labels = labels.to(device)

    model = SNNGestureClassifier(data.N_CHANNELS, hidden_size, beta=beta,
                                  threshold=spike_threshold).to(device)
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

            spk_in = encode_spike_trains(batch_examples, delta_threshold).to(device)
            spk2_rec = model(spk_in)
            loss = loss_fn(spk2_rec, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        print(f"  epoch {epoch + 1:3d}/{num_epochs} -> loss={epoch_loss / n:.4f}", flush=True)

    return model


def evaluate_snn(model, test_by_class, delta_threshold=DEFAULT_DELTA_THRESHOLD,
                  batch_size=DEFAULT_BATCH_SIZE, device="cpu", return_per_class=False):
    """Rate-coded classification: argmax of each class's summed output-spike count
    over all timesteps. Mirrors evaluate_permutation/evaluate_gak_from_kernels's
    (accuracy[, per_class]) return shape."""
    model.eval()
    idx_to_label = {i: label for i, label in enumerate(data.CLASSES)}

    correct, total = 0, 0
    class_correct = {label: 0 for label in data.CLASSES}
    class_total = {label: 0 for label in data.CLASSES}

    with torch.no_grad():
        for label, exs in test_by_class.items():
            for start in range(0, len(exs), batch_size):
                batch_examples = exs[start:start + batch_size]
                spk_in = encode_spike_trains(batch_examples, delta_threshold).to(device)
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

    model = train_snn(train_by_class, device=device, num_epochs=10)
    accuracy, per_class = evaluate_snn(model, test_by_class, device=device, return_per_class=True)
    print(f"hidden_size={DEFAULT_HIDDEN_SIZE} -> accuracy={accuracy:.4f}")
    for label in sorted(per_class):
        print(f"  class {label}: {per_class[label]:.4f}")
