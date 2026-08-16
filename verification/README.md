# Verification tiers

Checks that `src/pynn_lif_model.py` correctly reproduces the trained `sparch` LIF
model (see repo root `README.md` for the full pipeline), at increasing cost and
increasing fidelity to the real SpiNNaker deployment target. Run each tier in
order — no point spending SpiNNaker queue time on a bug catchable locally.

## Tier 1 — `reference_lif_numpy.py` (always run, numpy only)

Pure-numpy reimplementation of sparch's exact discrete-time forward pass
(`ut = alpha*(ut-st) + (1-alpha)*Wx_t`, and the softmax-accumulation readout),
given the weights exported by `sparch/export_weights.py`. No PyNN, no torch.

This checks that the export + BatchNorm-fold (`export_weights.py`) is correct,
fully decoupled from any PyNN/hardware translation.

**Status: passing.** Cross-checked against real sparch/PyTorch output with
same-random-draw parity (`sparch/verify_export.py`, which monkeypatches
`torch.rand` to capture the exact values sparch's forward pass draws for
`ut`/`st` initialization, then feeds those same draws into this module) —
**20/20 exact prediction agreement**, max output difference ~5e-6
(float32/float64 rounding noise only).

```bash
conda activate sparch
cd /path/to/sparch
python verify_export.py exp/lif_shd_run1/checkpoints/best_model.pth shd_lif_weights.npz
```

## Tier 2 — `pynn_smoke_test.py` (if local PyNN+NEST installs)

Builds the network via `src/pynn_lif_model.py` against a local `pyNN.nest`
backend, and compares its spike-count-argmax predictions against tier 1's
numpy reference on real SHD test examples.

Requires `pyNN` and `nest-simulator` in the active conda env (`hdc-env`).
Install attempt (arm64 Mac, confirmed working — PyNN 0.13.0, NEST 3.9.0):

```bash
conda activate hdc-env
pip install pyNN
conda install -c conda-forge nest-simulator -y
```

If this install fails or is flaky, `pynn_smoke_test.py` prints a message and
exits 0 rather than hard-failing — it must never block the rest of the
pipeline.

```bash
conda activate hdc-env
cd /path/to/hdc-language
python verification/pynn_smoke_test.py --n_examples 25 --seed 42
```

**Use `--seed`.** SHD's `shd_test.h5` is sorted by label, so the first N
examples in file order are a skewed, unrepresentative sample (e.g. the first
20 examples contain 4x label-10 and 0x several other labels). Always pass
`--seed` for a real read on accuracy; omit it only when you want a fixed,
reproducible small smoke test regardless of representativeness.

**Status: three real bugs found. Two fixed (degenerate-network init/scaling,
and a numpy-reference input-binning bug); one open (PyNN readout firing-rate
imbalance across output classes).** See the project plan file for the full
derivation.

Fixed:

1. `IF_curr_exp` defaults to `v=-65.0` at `t=0` regardless of `v_rest` —
   `pop.initialize(v=0.5)` must be called explicitly.
2. Naive current-injection scaling (tried both raw `W_eff`/`b_eff` and a flat
   `(1-alpha)` multiplier) was wrong by 1-2 orders of magnitude, causing a
   sustained bias current to make neurons fire periodically forever
   regardless of input — the network's prediction was the same output neuron
   on every single test example. Fixed by deriving the correct bias
   (`I = b_eff / tau_m`, matching sparch's and PyNN's respective fixed
   points) and weight (`w_scaled = w * (1-alpha) / r`, where `r` is the
   numerically-integrated unit-weight impulse response of the actual
   `IF_curr_exp` + `tau_syn` dynamics) scalings, verified against the tier-1
   numpy reference on isolated single-neuron cases before reapplying to the
   full network.
3. `pynn_smoke_test.py`'s `dense_input_from_spikes` (a from-scratch
   reimplementation of sparch's `SpikingDataset.__getitem__` binning, used
   only to feed the tier-1 numpy reference inside this script) built the
   dense input via plain fancy-indexed assignment (`x[idx] = 1.0`), which
   silently clamps to 1 when multiple raw spikes land in the same digitized
   `(time, unit)` bin. Sparch's real dataloader builds the same array via
   `torch.sparse.FloatTensor(...).to_dense()`, which *sums* colliding
   entries instead — and collisions are common, not rare (every SHD test
   example has 1,000+ colliding `(time, unit)` cells out of ~100k, with up
   to 5 spikes stacked in one cell, since SHD's raw spike timing is much
   finer than the 100-step binning). This silently understated input current
   throughout the network and alone was enough to sink the tier-1 reference's
   own measured accuracy to 26/100 on a 100-example sample, even though
   tier-1's actual forward-pass math is proven bit-exact elsewhere
   (`sparch/verify_export.py`, which goes through sparch's real dataloader
   and never had this bug). Fixed by replacing the assignment with
   `np.add.at`, matching sparse-tensor accumulate-on-collision semantics.
   After the fix, the tier-1 reference scores 82/100 on the same
   100-example/seed=42 sample — consistent with sparch's real 84.03% test
   accuracy (see repo root README commit history for that training run).

**Open — PyNN readout has a systematic per-class firing-rate imbalance,**
found after fix #3 gave a trustworthy tier-1 baseline to compare against. On
the same 100-example/seed=42 sample: tier-1 reference 82/100, but **PyNN only
37/100**, and PyNN's predictions are not merely noisier — they collapse onto
a handful of classes. Out of 100 examples, PyNN predicted only 9 of the 20
possible classes, with 91/100 predictions landing on just 6 classes (9, 10,
12, 14, 17, 18), while true labels were roughly evenly spread across all 20.

Root cause: the readout layer's per-neuron `alpha` (sparch's learned decay)
varies across output classes, converting via `alpha_to_tau_m` to `tau_m`
ranging from 70ms (classes 9, 12, 14, 17, 18 — exactly the over-predicted
set) to 350ms (classes 2, 3, 5, 15, 19 — never predicted once in 100
examples). `weight_to_scaled_weight`'s per-neuron scale factor
`(1-alpha)/r` is ~5x larger for the fast (`tau_m=70ms`) neurons than the slow
(`tau_m=350ms`) ones (0.218 vs. 0.041, measured directly on this layer),
because `r` (the numerically-integrated unit-impulse response) is calibrated
to match sparch's discrete-step delta at exactly `t=dt_ms=14ms` — a
calibration that was verified correct in isolated single-neuron cases (see
bug #2 above) but doesn't hold once many neurons with different `tau_m` are
firing together over the full 100-step/1.4s window: fast neurons get both a
larger injected-current gain *and* faster recovery from reset, compounding
into an input-independent firing-rate advantage that dominates spike-count
argmax regardless of which class the input actually belongs to. This is
larger than, and distinct from, the two previously-named approximation
sources (softmax-argmax vs. spike-count-argmax readout; subtractive vs.
absolute reset) — those cause prediction *disagreement*, not concentration
into a fixed neuron subset.

Not yet fixed — needs either a per-neuron rate-normalization term (e.g.
calibrate `w_scaled` against total accumulated current over the full
simulation window instead of a single `dt_ms` step) or a different readout
decision rule less sensitive to absolute per-neuron rate (e.g. z-scoring
spike counts by each neuron's baseline/bias-only firing rate before argmax).

**If you want to properly quantify the approximation gap's real accuracy
cost:** run both methods on the same larger (100+) random sample and compare
accuracy-vs-true-label directly, rather than exact-agreement-with-reference —
but note the current PyNN number is now known to be dominated by the
readout-scaling bug above, not just the two originally-named approximations.

## Tier 3 — real SpiNNaker batch job (`notebooks/spinnaker_lif_shd_deploy.ipynb`)

Only attempt once tiers 1–2 look reasonable, to avoid spending hardware
queue time on bugs catchable locally. See the notebook and the repo root
`README.md` for how to run this on EBRAINS Collaboratory.
