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
DEFAULT_ALPHA = 0.5             # snn.Synaptic synaptic-current decay
DEFAULT_SPIKE_THRESHOLD = 0.9   # membrane firing threshold (Leaky and Synaptic)
DEFAULT_DELTA_THRESHOLD = 0.1   # spikegen.delta's per-step change threshold
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
NUM_CLASSES = len(data.CLASSES)

# dataloader
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def _stack_examples(examples):
    """list of (n_channels, seriesLength) arrays -> float32 tensor of shape
    (seriesLength, batch, n_channels), raw values, no encoding applied yet."""
    return torch.stack([
        torch.from_numpy(ex.T.astype(np.float32)) for ex in examples
    ], dim=1)


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


def encode_rate_spike_trains(examples):
    batch = _normalize_to_unit_interval(_stack_examples(examples))
    return spikegen.rate(batch, time_var_input=True)


def encode_delta_spike_trains(examples, delta_threshold=DEFAULT_DELTA_THRESHOLD):
    return spikegen.delta(_stack_examples(examples), threshold=delta_threshold, off_spike=True)


class _MultiBetaSNN(nn.Module):
    """Feedforward SNN with `num_hidden` hidden snn.Synaptic layers followed by one
    output snn.Synaptic layer, each hidden layer's width taken from `hidden_sizes`
    (len == num_hidden) and its beta from `betas` (len == num_hidden + 1, last entry
    is the output layer's beta). Every fc_i -> hidden_i, then a final fc that maps
    the last hidden width to num_classes -> output layer. Subclasses just fix
    num_hidden and expose one beta_i kwarg per layer for backward-compatible configs."""

    def __init__(self, n_channels, hidden_sizes, num_classes, alpha, betas, threshold):
        super().__init__()
        assert len(hidden_sizes) == len(betas) - 1, "need one beta per hidden layer plus one for output"
        widths_in = [n_channels] + list(hidden_sizes)
        widths_out = list(hidden_sizes) + [num_classes]

        self.fcs = nn.ModuleList([
            nn.Linear(widths_in[i], widths_out[i]) for i in range(len(widths_out))
        ])
        self.lifs = nn.ModuleList([
            snn.Synaptic(alpha=alpha, beta=betas[i], threshold=threshold, spike_grad=surrogate.atan())
            for i in range(len(betas))
        ])

    def forward(self, spk_in):
        """spk_in: (num_steps, batch, n_channels) -> spk_rec: (num_steps, batch, num_classes)."""
        num_steps = spk_in.shape[0]
        states = [lif.init_synaptic() for lif in self.lifs]

        spk_rec = []
        for step in range(num_steps):
            spk = spk_in[step]
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(spk)
                spk, syn, mem = lif(cur, *states[i])
                states[i] = (syn, mem)
            spk_rec.append(spk)

        return torch.stack(spk_rec, dim=0)


def _hidden_sizes_from_scalar(hidden_size, num_hidden):
    """hidden_size (int, first layer's width) -> list of num_hidden widths, halving
    each layer (min 1), e.g. hidden_size=512, num_hidden=3 -> [512, 256, 128]."""
    sizes = [hidden_size]
    for _ in range(num_hidden - 1):
        sizes.append(max(1, sizes[-1] // 2))
    return sizes


# 1. Synaptic + MultiBeta, 3 FC layers, 2 hidden, 1 output
class SNNGestureClassifierMultiBeta3(_MultiBetaSNN):
    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 alpha=DEFAULT_ALPHA,
                 beta_1=0.9, beta_2=0.8, beta_out=0.7,
                 threshold=DEFAULT_SPIKE_THRESHOLD):
        hidden_sizes = _hidden_sizes_from_scalar(hidden_size, num_hidden=2)
        super().__init__(n_channels, hidden_sizes, num_classes, alpha,
                          betas=[beta_1, beta_2, beta_out], threshold=threshold)


# 2. Synaptic + MultiBeta, 4 FC layers, 3 hidden, 1 output
class SNNGestureClassifierMultiBeta4(_MultiBetaSNN):
    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 alpha=DEFAULT_ALPHA,
                 beta_1=0.9, beta_2=0.83, beta_3=0.77, beta_out=0.7,
                 threshold=DEFAULT_SPIKE_THRESHOLD):
        hidden_sizes = _hidden_sizes_from_scalar(hidden_size, num_hidden=3)
        super().__init__(n_channels, hidden_sizes, num_classes, alpha,
                          betas=[beta_1, beta_2, beta_3, beta_out], threshold=threshold)


# 3. Synaptic + MultiBeta, 5 FC layers, 4 hidden, 1 output
class SNNGestureClassifierMultiBeta5(_MultiBetaSNN):
    def __init__(self, n_channels, hidden_size, num_classes=NUM_CLASSES,
                 alpha=DEFAULT_ALPHA,
                 beta_1=0.9, beta_2=0.85, beta_3=0.8, beta_4=0.75, beta_out=0.7,
                 threshold=DEFAULT_SPIKE_THRESHOLD):
        hidden_sizes = _hidden_sizes_from_scalar(hidden_size, num_hidden=4)
        super().__init__(n_channels, hidden_sizes, num_classes, alpha,
                          betas=[beta_1, beta_2, beta_3, beta_4, beta_out], threshold=threshold)


def _flatten_by_class(by_class):
    """{label: [examples]} -> (examples list, integer-label tensor), label order
    fixed by data.CLASSES ("1".."8" -> 0..7) so class indices match lif2's 8 outputs."""
    label_to_idx = {label: i for i, label in enumerate(data.CLASSES)}   # label: i dict
    examples, labels = [], []
    for label in data.CLASSES:
        for ex in by_class[label]:
            examples.append(ex)
            labels.append(label_to_idx[label])
    return examples, torch.tensor(labels, dtype=torch.long)


def train_snn(train_by_class, encode_fn, model_cls=SNNGestureClassifierMultiBeta3, model_kwargs=None,
              num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
              device=DEVICE, seed=42, loss_fn=None):
    """train_by_class looks like this:
    {
        "1": [array(3,315), array(3,315), ...],
        "2": [array(3,315), array(3,315), ...],
        ...
        "8": [array(3,315), array(3,315), ...],
    }"""

    torch.manual_seed(seed)
    examples, labels = _flatten_by_class(train_by_class)
    labels = labels.to(device)

    model = model_cls(data.N_CHANNELS, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn if loss_fn is not None else SF.ce_rate_loss()

    n = len(examples)
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_examples = [examples[i] for i in idx]
            batch_labels = labels[idx]

            spk_in = encode_fn(batch_examples).to(device)
            spk_rec= model(spk_in)
            loss = loss_fn(spk_rec, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        print(f"  epoch {epoch + 1:3d}/{num_epochs} -> loss={epoch_loss / n:.4f}", flush=True)

    return model


def _first_spike_time(spk_rec):
    """spk_rec: (num_steps, batch, num_classes) -> (batch, num_classes) float tensor of
    each neuron's first spike step index (0-indexed); neurons that never spike get
    num_steps - 1 (i.e. tied for latest possible, so argmin over classes treats
    "never fired" as the worst outcome). Matches snntorch.functional.acc.accuracy_temporal's
    first-spike extraction."""
    num_steps = spk_rec.shape[0]
    device = spk_rec.device
    step_idx = (torch.arange(1, num_steps + 1, device=device)).view(-1, 1, 1)
    spk_time = spk_rec * step_idx  # nonzero only at spike steps, value = step index + 1

    first_spike_time = torch.zeros_like(spk_time[0])
    for step in range(num_steps):
        first_spike_time += spk_time[step] * (first_spike_time == 0)

    never_spiked = first_spike_time == 0
    first_spike_time = first_spike_time + never_spiked * num_steps
    return first_spike_time - 1


def evaluate_snn(model, test_by_class, encode_fn,
                  batch_size=DEFAULT_BATCH_SIZE, device="cpu", return_per_class=False,
                  return_confusion=False, decode="rate"):
    """Encodes each batch with encode_fn (examples -> spk_in), then decodes model's
    output spike train spk_rec (num_steps, batch, num_classes) one of two ways:
      - decode="rate" (default): argmax of each class's summed output-spike count
        over all timesteps. Pair with the default SF.ce_rate_loss() in train_snn.
      - decode="temporal": argmin of each class's first-spike time (via
        _first_spike_time); a class that never spikes is scored as if it fired on
        the very last step. Pair with SF.ce_temporal_loss() in train_snn.
    In both cases a per-class "score" is produced where higher = more preferred, so
    the rest of the bookkeeping (confusion, margins) is decode-agnostic.

    return_confusion=True additionally returns:
      - confusion: dict[true_label][pred_label] -> count, from the same per-example
        predictions used for per_class_accuracy.
      - margins: list of (is_correct, margin) per test example, margin = the true
        class's score minus the top *other* class's score (spike-count difference for
        rate decoding, negative first-spike-time difference for temporal decoding) --
        positive margins that are still wrong mean the true class was runner-up; very
        negative margins mean confidently wrong.
    """
    if decode not in ("rate", "temporal"):
        raise ValueError(f"decode must be 'rate' or 'temporal', got {decode!r}")
    model.eval()
    idx_to_label = {i: label for i, label in enumerate(data.CLASSES)}
    label_to_idx = {label: i for i, label in idx_to_label.items()}

    correct, total = 0, 0
    class_correct = {label: 0 for label in data.CLASSES}
    class_total = {label: 0 for label in data.CLASSES}
    confusion = {t: {p: 0 for p in data.CLASSES} for t in data.CLASSES}
    margins = []

    with torch.no_grad():
        for label, exs in test_by_class.items():
            true_idx = label_to_idx[label]
            for start in range(0, len(exs), batch_size):
                batch_examples = exs[start:start + batch_size]
                spk_in = encode_fn(batch_examples).to(device)
                spk_rec = model(spk_in)
                if decode == "rate":
                    scores = spk_rec.sum(dim=0)  # (batch, num_classes), higher = better
                else:
                    scores = -_first_spike_time(spk_rec)  # earlier spike -> higher score
                preds = scores.argmax(dim=1).tolist()

                for row, pred_idx in enumerate(preds):
                    pred_label = idx_to_label[pred_idx]
                    is_correct = int(pred_label == label)
                    correct += is_correct
                    total += 1
                    class_correct[label] += is_correct
                    class_total[label] += 1
                    confusion[label][pred_label] += 1

                    row_scores = scores[row]
                    true_score = row_scores[true_idx].item()
                    other_scores = row_scores.clone()
                    other_scores[true_idx] = float("-inf")
                    top_other_score = other_scores.max().item()
                    margins.append((bool(is_correct), true_score - top_other_score))

    accuracy = correct / total
    per_class_accuracy = {
        label: (class_correct[label] / class_total[label] if class_total[label] else 0.0)
        for label in data.CLASSES
    }

    result = [accuracy]
    if return_per_class:
        result.append(per_class_accuracy)
    if return_confusion:
        result.append(confusion)
        result.append(margins)
    return tuple(result) if len(result) > 1 else result[0]
