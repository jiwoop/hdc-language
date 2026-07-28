"""Spike-train visualizations for the three SNN gesture-classification methods
compared in run_gesture_snn_method_comparison.py, using the very first UWave
training example (data.CLASSES[0]'s first example).

Two kinds of plot, per snntorch tutorial conventions:
  1. Input spike-train rasters (all 3 channels x 315 steps) for rate coding and
     delta modulation, via snntorch.spikeplot.raster.
  2. Single-neuron current/membrane(/synaptic-current)/spike-output traces for
     snn.Leaky and snn.Synaptic, feeding channel 0's delta-encoded spike train
     directly into one freshly-initialized neuron (beta/alpha from
     run_gesture_snn_hparam_sweep.py's results.json). snntorch.spikeplot has no
     tutorial-style plot_cur_mem_spk/plot_spk_cur_mem_spk helpers (those are
     notebook-only, not part of the pip package), so small local equivalents are
     defined below.

Usage:
    python visualize_gesture_snn_spikes.py --hparam-dir DIR [--outdir DIR] [--device cuda|cpu]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import torch
import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import surrogate

import snn_encoder as se
import uwave_data as data


def load_best_hparams(hparam_dir):
    with open(os.path.join(hparam_dir, "results.json")) as f:
        summary = json.load(f)
    best = summary["best"]
    return best["leaky"]["beta"], best["synaptic"]["alpha"], best["synaptic"]["beta"]


def plot_raster(spk, title, outpath):
    """spk: (num_steps, n_channels) tensor -> raster scatter, one row per channel."""
    fig = plt.figure(facecolor="w", figsize=(9, 4))
    ax = fig.add_subplot(111)
    splt.raster(spk, ax, s=6, c="black")
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Channel")
    ax.set_yticks(range(spk.shape[1]))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line, title, outpath):
    """3-row plot: input current, membrane potential (with threshold line), output spikes.
    Mirrors the snntorch tutorial's plot_cur_mem_spk helper (not shipped in the package)."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))

    axes[0].plot(cur_in, color="tab:orange")
    axes[0].set_ylabel("Input current")
    axes[0].set_title(title)

    axes[1].plot(mem_rec, color="tab:blue")
    axes[1].axhline(y=thr_line, linestyle='--', color="black", alpha=0.6, label="threshold")
    axes[1].set_ylabel("Membrane potential")
    axes[1].legend(loc="upper right")

    axes[2].plot(spk_rec, color="tab:green")
    axes[2].set_ylabel("Output spikes")
    axes[2].set_xlabel("Time step")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, title, outpath):
    """4-row plot: input spikes, synaptic current, membrane potential, output spikes.
    Mirrors the snntorch tutorial's plot_spk_cur_mem_spk helper (not shipped in the
    package)."""
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 9))

    axes[0].plot(spk_in, color="black")
    axes[0].set_ylabel("Input spikes")
    axes[0].set_title(title)

    axes[1].plot(syn_rec, color="tab:orange")
    axes[1].set_ylabel("Synaptic current")

    axes[2].plot(mem_rec, color="tab:blue")
    axes[2].set_ylabel("Membrane potential")

    axes[3].plot(spk_rec, color="tab:green")
    axes[3].set_ylabel("Output spikes")
    axes[3].set_xlabel("Time step")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def simulate_leaky(spk_in_1d, beta, threshold):
    """spk_in_1d: (num_steps,) 1D tensor fed as scalar input current into one
    freshly-initialized snn.Leaky neuron. Mirrors the pasted tutorial snippet."""
    lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.atan())
    mem = torch.zeros(1)
    spk = torch.zeros(1)
    mem_rec, spk_rec = [], []

    for step in range(len(spk_in_1d)):
        spk, mem = lif1(spk_in_1d[step].unsqueeze(0), mem)
        mem_rec.append(mem)
        spk_rec.append(spk)

    return torch.stack(mem_rec).squeeze(-1), torch.stack(spk_rec).squeeze(-1)


def simulate_synaptic(spk_in_1d, alpha, beta, threshold):
    """spk_in_1d: (num_steps,) 1D tensor fed as scalar input current into one
    freshly-initialized snn.Synaptic neuron. Mirrors the pasted tutorial snippet."""
    lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=surrogate.atan())
    syn, mem = lif1.init_synaptic()
    spk = torch.zeros(1)
    syn_rec, mem_rec, spk_rec = [], [], []

    for step in range(len(spk_in_1d)):
        spk, syn, mem = lif1(spk_in_1d[step].unsqueeze(0), syn, mem)
        syn_rec.append(syn)
        mem_rec.append(mem)
        spk_rec.append(spk)

    return torch.stack(syn_rec).squeeze(-1), torch.stack(mem_rec).squeeze(-1), torch.stack(spk_rec).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam-dir", required=True,
                        help="Output dir from run_gesture_snn_hparam_sweep.py.")
    parser.add_argument("--outdir", default="results_gesture_snn_spikes")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    leaky_beta, synaptic_alpha, synaptic_beta = load_best_hparams(args.hparam_dir)
    print(f"leaky beta={leaky_beta}, synaptic alpha={synaptic_alpha} beta={synaptic_beta}", flush=True)

    train_by_class = data.load_split("TRAIN")
    example = train_by_class[data.CLASSES[0]][0]  # very first training sample, (3, 315)

    rate_spk = se.encode_rate_spike_trains([example])[:, 0, :]    # (315, 3)
    delta_spk = se.encode_delta_spike_trains([example])[:, 0, :]  # (315, 3)

    plot_raster(rate_spk, "Rate-coded input spikes (first training sample)",
                os.path.join(args.outdir, "spike_raster_rate.png"))
    plot_raster(delta_spk, "Delta-modulated input spikes (first training sample)",
                os.path.join(args.outdir, "spike_raster_delta.png"))

    channel0 = delta_spk[:, 0]  # (315,), values in {-1, 0, 1}

    mem_rec, spk_rec = simulate_leaky(channel0, leaky_beta, se.DEFAULT_SPIKE_THRESHOLD)
    plot_cur_mem_spk(channel0.numpy(), mem_rec.detach().numpy(), spk_rec.detach().numpy(),
                      thr_line=se.DEFAULT_SPIKE_THRESHOLD,
                      title=f"snn.Leaky neuron response (beta={leaky_beta}, channel 0, delta-coded input)",
                      outpath=os.path.join(args.outdir, "neuron_response_leaky.png"))

    syn_rec, mem_rec, spk_rec = simulate_synaptic(channel0, synaptic_alpha, synaptic_beta,
                                                    se.DEFAULT_SPIKE_THRESHOLD)
    plot_spk_cur_mem_spk(channel0.numpy(), syn_rec.detach().numpy(), mem_rec.detach().numpy(),
                          spk_rec.detach().numpy(),
                          title=(f"snn.Synaptic neuron response (alpha={synaptic_alpha}, "
                                 f"beta={synaptic_beta}, channel 0, delta-coded input)"),
                          outpath=os.path.join(args.outdir, "neuron_response_synaptic.png"))

    print(f"\nWrote 4 plots to {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
