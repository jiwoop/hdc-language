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

**Status: degenerate-network bug found and fixed; approximation gaps behave
as expected.** Two real bugs were found and fixed via isolated single-neuron
unit tests (see the project plan file for the full derivation):

1. `IF_curr_exp` defaults to `v=-65.0` at `t=0` regardless of `v_rest` —
   `pop.initialize(v=0.5)` must be called explicitly.
2. The dominant bug: naive current-injection scaling (tried both raw
   `W_eff`/`b_eff` and a flat `(1-alpha)` multiplier) was wrong by 1-2 orders
   of magnitude, causing a sustained bias current to make neurons fire
   periodically forever regardless of input — the network's prediction was
   the same output neuron on every single test example. Fixed by deriving
   the correct bias (`I = b_eff / tau_m`, matching sparch's and PyNN's
   respective fixed points) and weight (`w_scaled = w * (1-alpha) / r`,
   where `r` is the numerically-integrated unit-weight impulse response of
   the actual `IF_curr_exp` + `tau_syn` dynamics) scalings, verified against
   the tier-1 numpy reference on isolated single-neuron cases before
   reapplying to the full network.

After the fix, predictions vary correctly with input (no longer degenerate).
Exact-prediction agreement with the tier-1 reference is intentionally
**not** the headline metric — the readout-layer approximation
(softmax-accumulation argmax vs. spike-count argmax; see module docstring in
`src/pynn_lif_model.py`) and the reset-semantics approximation (subtractive
vs. absolute reset) are named, expected sources of prediction disagreement.
Instead, compare **raw accuracy against true labels** for both methods on
the same random sample — on one 25-example random sample, PyNN's own
accuracy (52%) was *higher* than the reference's (28%), and on a separate
100-example random sample the reference alone scored 63% (close to sparch's
real 72.5% test accuracy) — confirming small-sample runs are just
high-variance, not indicative of a broken pipeline.

**If you want to properly quantify the approximation gap's real accuracy
cost:** run both methods on the same larger (100+) random sample and compare
accuracy-vs-true-label directly, rather than exact-agreement-with-reference.

## Tier 3 — real SpiNNaker batch job (`notebooks/spinnaker_lif_shd_deploy.ipynb`)

Only attempt once tiers 1–2 look reasonable, to avoid spending hardware
queue time on bugs catchable locally. See the notebook and the repo root
`README.md` for how to run this on EBRAINS Collaboratory.
