# Deploying a sparch LIF SNN to SpiNNaker (PyNN batch deployment)

This branch (`heidelberg-snn`) is a hardware-profiling exercise: deploying a **trained `sparch`
LIF model** (a separate PyTorch SNN codebase, vendored here as a git submodule) on the **Spiking
Heidelberg Digits (SHD)** dataset to real **SpiNNaker neuromorphic silicon**, via PyNN and the
EBRAINS Neuromorphic Computing Platform's batch-submission API (`nmpi`).

It does not use HDC (hyperdimensional computing) or snnTorch — earlier work on this branch
explored those (HDC language/gesture classification, an snnTorch gesture baseline), but that
code was removed by a "Clean out structure" commit before this branch pivoted to the SpiNNaker
port below. If you're looking for that HDC/snnTorch work, check history before commit
`134326b` or other branches.

## Project layout

```
sparch/         git submodule (github.com/idiap/sparch, upstream, unmodified) -- the trained
                PyTorch LIF SNN codebase. Local scripts (export_weights.py, verify_export.py)
                live inside this checkout as untracked files, not part of the submodule's own
                git history.
src/            pynn_lif_model.py (builds the PyNN network from exported weights),
                generate_spinnaker_script.py (assembles the self-contained SpiNNaker batch job
                script), src/data/ (committed exported weights, e.g. shd_lif_weights.npz)
slurm/          train_lif_shd.slurm -- TACC batch job for a longer sparch training run (submit
                from the hdc-language repo root; the script cds into sparch/ itself)
verification/   tiered correctness checks, run before spending SpiNNaker hardware queue time --
                see verification/README.md
notebooks/      spinnaker_lif_shd_deploy.ipynb -- the EBRAINS batch-submission notebook, run on
                EBRAINS Collaboratory's Jupyter (via "nmc-test-Collaboratory" access), not locally
env.yml         conda env `hdc-env` -- PyNN and NEST (for local verification) are installed via
                pip/conda directly, not tracked here (see verification/README.md)
```

`slurm/snn_multibeta.slurm`, `logs/`, `results/`, and `scripts/` are leftover from the earlier
HDC/snnTorch work and are currently unused by anything on this branch.

---

## Why LIF, not adLIF

`sparch` also has an adaptive-LIF (adLIF) variant with a subthreshold-voltage-coupling term that
has no equivalent in standard PyNN neuron models — plain LIF maps directly onto PyNN's
`IF_curr_exp` (current-based synapses, matching sparch's current-like, non-conductance-based
formulation), so it's the lower-risk model to port first.

## Why SHD, not HD

SHD's input is already a spike train (precomputed spike times, binned by `sparch`'s own
dataloader into 100 steps over 1.4s — an exact, known 14ms/step), so every layer, including the
input layer, can be a genuine PyNN `Population` connected by `Projection`s. HD's input is
continuous mel-filterbank features that would need a host-side current-injection workaround for
the first layer alone.

## The pipeline

```
sparch/ (submodule)                       hdc-language (this repo)
────────────────────                      ─────────────────────────
1. train LIF model on SHD
   (run_exp.py)
2. export_weights.py         ──.npz──▶    src/data/shd_lif_weights.npz
                                           3. src/pynn_lif_model.py builds
                                              the equivalent PyNN network
                                           4. verification/ checks it locally
                                              (see verification/README.md)
                                           5. notebooks/spinnaker_lif_shd_deploy.ipynb
                                              submits it as a real SpiNNaker batch job
```

**1-2. Train + export** (inside the `sparch/` submodule):

```bash
conda activate sparch
cd sparch
python run_exp.py --model_type LIF --dataset_name shd \
    --data_folder shd_dataset --new_exp_folder exp/lif_shd_run1
python export_weights.py exp/lif_shd_run1/checkpoints/best_model.pth shd_lif_weights.npz
cp shd_lif_weights.npz ../src/data/
```

A longer 50-epoch training run is available as a TACC batch job:
`sbatch slurm/train_lif_shd.slurm` (from the `hdc-language` repo root; CPU queue — this training loop has no GPU code,
~27s/epoch observed locally).

`export_weights.py` folds each layer's trained `BatchNorm1d` into an effective `(W_eff, b_eff)`
pair, so the exported `.npz` is self-contained plain numpy arrays — no torch, no BatchNorm
concept needed downstream.

**3. `src/pynn_lif_model.py`** builds the PyNN network from the exported weights. The key
translation work (discrete-time sparch update ↔ continuous-time PyNN/SpiNNaker neuron dynamics)
required deriving the correct current-injection scaling empirically, not just analytically — see
the module's docstrings (`bias_to_current`, `weight_to_scaled_weight`) for the derivations, and
`verification/README.md` for how they were validated against sparch's real output.

**Two named, deliberate approximations** (not bugs — there is no exact PyNN/hardware equivalent):
- **Reset semantics**: sparch's spike reset subtracts a fixed 1.0 from the membrane potential
  before that step's decay; PyNN's `IF_curr_exp` resets to an absolute `v_reset`. Approximated as
  `v_thresh=1.0, v_reset=0.0, v_rest=0.0`.
- **Readout**: sparch's readout layer is non-spiking, accumulating `softmax(potential)` across
  all 20 output neurons jointly every timestep — no PyNN neuron primitive computes softmax across
  a population. Approximated as a spiking population classified by **spike-count argmax** (rate
  coding) instead.

**4. `verification/`** — three tiers of correctness checks, cheapest first, before spending
SpiNNaker queue time. See `verification/README.md` for the full breakdown; headline results:
tier 1 (pure numpy vs. real sparch/PyTorch output) is exact (20/20 same-random-draw prediction
match). Tier 2 (local `pyNN.nest`) required finding and fixing two real bugs — an uninitialized
membrane potential default, and a current-scaling error that caused degenerate,
input-independent predictions — documented in the verification README, along with why
exact-prediction agreement with the reference isn't the right headline metric once the two named
approximations above are accounted for (raw accuracy-vs-true-label is more informative and was
comparable between the two methods on the samples tested).

**5. `notebooks/spinnaker_lif_shd_deploy.ipynb`** — run this on EBRAINS Collaboratory's Jupyter
(via "nmc-test-Collaboratory" access), not locally. Mirrors the batch-submission pattern used
elsewhere on the platform: since the SpiNNaker job runner sandbox only sees a single uploaded
script, `src/generate_spinnaker_script.py` inlines the exported weights and a sample of real SHD
test examples as base64-encoded bytes directly into a self-contained job script before
submission, rather than relying on any external file access from within the job.

**Is this a faithful hardware deployment?** Worth being honest about: every layer is a genuine
spiking `Population` connected by `Projection`s (not a precomputed-current-replay workaround),
which is the architecturally meaningful choice for profiling real event-driven neuromorphic
hardware — but getting there required an empirically-derived current-scaling correction, not a
clean analytic translation, and the two approximations above mean predictions will diverge from
sparch's exact trained behavior on some inputs. The right way to think about SpiNNaker's output
here is "a plausible, input-driven approximation of the trained model, deployed in a way that's
meaningful for hardware profiling" — not "the trained model, verbatim, on different silicon."
