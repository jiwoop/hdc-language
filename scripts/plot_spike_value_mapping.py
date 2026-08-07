"""Visualizes what a raw value maps to under each of snn_encoder.py's two spike
encodings (src/snn_encoder.py: encode_rate_spike_trains, encode_delta_spike_trains).

Rate coding treats a value in [0, 1] as a per-timestep firing *probability*
(spikegen.rate) -- a constant level L sampled 0..1 fires roughly L fraction of the
time, independent of anything else. Delta coding instead spikes on *change*
(spikegen.delta): a constant value at any level never spikes; only a jump larger
than `threshold` between consecutive timesteps fires (+1 on a rise, -1 on a fall).
That's why "what does 0, 0.1, ..., 1 map to" means something different for each --
for rate it's a held level, for delta it's a step size -- and why delta is run
unnormalized (raw scale) here while rate needs its [0, 1] input.

Produces 4 PNGs in --outdir:
  1. rate_value_sweep.png   -- constant levels 0.0..1.0 -> Bernoulli spike rasters,
     with measured empirical firing rate per level.
  2. rate_real_example.png  -- one real, min-max-normalized UWave channel and the
     spike raster encode_rate_spike_trains actually produces for it.
  3. delta_jump_sweep.png   -- step pulses of height 0.0..1.0 (raw scale, unnormalized)
     -> on/off spike rasters, showing the threshold cutoff.
  4. delta_real_example.png -- one real, raw (unnormalized) UWave channel and the
     spike raster encode_delta_spike_trains actually produces for it.

Usage:
    python plot_spike_value_mapping.py [--outdir DIR] [--class-label 1] [--example-idx 0]
        [--channel 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import snntorch.spikegen as spikegen

import snn_encoder as se
import uwave_data as data

ON_COLOR = "#55A868"    # matches plot_snn_error_analysis.py's "correct"/positive color
OFF_COLOR = "#C44E52"   # matches its "incorrect"/negative color
TRACE_COLOR = "#4C72B0"

LEVELS = np.round(np.arange(0.0, 1.0001, 0.1), 2)  # 0.0, 0.1, ..., 1.0


def plot_rate_value_sweep(outdir, num_steps=60, seed=0):
    """Constant value L held for all `num_steps` timesteps -> spikegen.rate fires
    each step with probability L, independent of prior steps. Sweeps L over LEVELS
    and shows the resulting Bernoulli raster plus the empirically measured rate."""
    torch.manual_seed(seed)
    level_tensor = torch.tensor(LEVELS, dtype=torch.float32).view(1, 1, -1).repeat(num_steps, 1, 1)
    spikes = spikegen.rate(level_tensor, time_var_input=True)[:, 0, :].numpy()  # (num_steps, n_levels)
    measured_rate = spikes.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for li, level in enumerate(LEVELS):
        steps = np.nonzero(spikes[:, li])[0]
        ax.scatter(steps, np.full(steps.shape, li), marker="|", s=180, linewidths=1.5, color=ON_COLOR)
        ax.text(num_steps + 1, li, f"L={level:.1f}  measured rate={measured_rate[li]:.2f}",
                va="center", fontsize=8)

    ax.set_yticks(range(len(LEVELS)))
    ax.set_yticklabels([f"{l:.1f}" for l in LEVELS])
    ax.set_xlim(-1, num_steps + 24)
    ax.set_ylim(-0.7, len(LEVELS) - 0.3)
    ax.set_xlabel("timestep")
    ax.set_ylabel("constant input value (held for all timesteps)")
    ax.set_title("encode_rate_spike_trains: value = per-timestep firing probability\n"
                  "(spikegen.rate, Bernoulli-sampled each step -- no memory of prior spikes)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rate_value_sweep.png"), dpi=150)
    plt.close()


def plot_delta_jump_sweep(outdir, total_steps=40, rise=12, fall=28):
    """A step pulse -- 0 for [0, rise), height L for [rise, fall), back to 0 after --
    for each jump size L in LEVELS (raw/unnormalized scale, no [0, 1] clamp; delta
    coding never normalizes). Shows spikegen.delta's threshold cutoff: only jumps
    that clear DEFAULT_DELTA_THRESHOLD fire, +1 on the rise and -1 on the fall."""
    pulse = torch.zeros(total_steps, 1, len(LEVELS))
    for li, level in enumerate(LEVELS):
        pulse[rise:fall, 0, li] = level
    spikes = spikegen.delta(pulse, threshold=se.DEFAULT_DELTA_THRESHOLD, off_spike=True)[:, 0, :].numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5))
    for li, level in enumerate(LEVELS):
        on_steps = np.nonzero(spikes[:, li] > 0)[0]
        off_steps = np.nonzero(spikes[:, li] < 0)[0]
        ax.scatter(on_steps, np.full(on_steps.shape, li), marker="|", s=220, linewidths=2, color=ON_COLOR)
        ax.scatter(off_steps, np.full(off_steps.shape, li), marker="|", s=220, linewidths=2, color=OFF_COLOR)
        fired = "fires" if level > se.DEFAULT_DELTA_THRESHOLD else "below threshold"
        ax.text(total_steps + 1, li, f"jump={level:.1f}  ({fired})", va="center", fontsize=8)

    ax.axvline(rise, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(fall, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(LEVELS)))
    ax.set_yticklabels([f"{l:.1f}" for l in LEVELS])
    ax.set_xlim(-1, total_steps + 16)
    ax.set_ylim(-0.7, len(LEVELS) - 0.3)
    ax.set_xlabel("timestep")
    ax.set_ylabel("pulse height L (raw scale, unnormalized)")
    ax.set_title(
        f"encode_delta_spike_trains: value = size of the step change, not a level\n"
        f"(spikegen.delta, threshold={se.DEFAULT_DELTA_THRESHOLD}: green=on-spike at rise,\n"
        "red=off-spike at fall; a constant value never spikes)")
    legend_handles = [
        plt.Line2D([0], [0], marker="|", color=ON_COLOR, linestyle="", markersize=12, markeredgewidth=2, label="on-spike (+1)"),
        plt.Line2D([0], [0], marker="|", color=OFF_COLOR, linestyle="", markersize=12, markeredgewidth=2, label="off-spike (-1)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delta_jump_sweep.png"), dpi=150)
    plt.close()


def _load_example(class_label, example_idx):
    train_by_class = data.load_split("TRAIN")
    return train_by_class[class_label][example_idx]  # (n_channels, series_length)


def plot_rate_real_example(outdir, example, channel):
    """Same value->probability mapping as plot_rate_value_sweep, but on one real
    channel: min-max normalized to [0, 1] (encode_rate_spike_trains's own
    normalization) then rate-encoded step by step."""
    raw = example[channel]
    spk = se.encode_rate_spike_trains([example])[:, 0, channel].numpy()  # (series_length,)
    span = raw.max() - raw.min()
    norm = (raw - raw.min()) / span if span > 1e-8 else np.zeros_like(raw)
    t = np.arange(len(raw))

    fig, (ax_trace, ax_raster) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_trace.plot(t, norm, color=TRACE_COLOR, linewidth=1.2)
    ax_trace.set_ylabel("value, min-max normalized to [0, 1]\n(= per-step spike probability)")
    ax_trace.set_title("encode_rate_spike_trains on a real UWave channel")
    ax_trace.grid(True, alpha=0.3)

    spike_steps = np.nonzero(spk)[0]
    ax_raster.scatter(spike_steps, np.zeros(spike_steps.shape), marker="|", s=100, linewidths=1, color=ON_COLOR)
    ax_raster.set_yticks([])
    ax_raster.set_ylabel("spikes")
    ax_raster.set_xlabel("timestep")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rate_real_example.png"), dpi=150)
    plt.close()


def plot_delta_real_example(outdir, example, channel):
    """Same on/off-at-change mapping as plot_delta_jump_sweep, but on one real,
    raw (unnormalized) channel -- exactly what encode_delta_spike_trains sees."""
    raw = example[channel]
    spk = se.encode_delta_spike_trains([example])[:, 0, channel].numpy()  # (series_length,)
    t = np.arange(len(raw))

    fig, (ax_trace, ax_raster) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_trace.plot(t, raw, color=TRACE_COLOR, linewidth=1.2)
    ax_trace.set_ylabel("raw value (unnormalized)")
    ax_trace.set_title(
        f"encode_delta_spike_trains on a real UWave channel (threshold={se.DEFAULT_DELTA_THRESHOLD})")
    ax_trace.grid(True, alpha=0.3)

    on_steps = np.nonzero(spk > 0)[0]
    off_steps = np.nonzero(spk < 0)[0]
    ax_raster.scatter(on_steps, np.zeros(on_steps.shape), marker="|", s=100, linewidths=1, color=ON_COLOR, label="on (+1)")
    ax_raster.scatter(off_steps, np.zeros(off_steps.shape), marker="|", s=100, linewidths=1, color=OFF_COLOR, label="off (-1)")
    ax_raster.set_yticks([])
    ax_raster.set_ylabel("spikes")
    ax_raster.set_xlabel("timestep")
    ax_raster.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delta_real_example.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_spike_value_mapping")
    parser.add_argument("--class-label", default="1", choices=data.CLASSES)
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0, choices=range(data.N_CHANNELS))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    plot_rate_value_sweep(args.outdir)
    plot_delta_jump_sweep(args.outdir)

    example = _load_example(args.class_label, args.example_idx)
    plot_rate_real_example(args.outdir, example, args.channel)
    plot_delta_real_example(args.outdir, example, args.channel)

    print(f"Wrote 4 PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
