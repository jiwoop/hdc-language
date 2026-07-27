# Running the HDC Nystrom sweep on Lonestar6

Reproduces everything in `hdc_nystrom.ipynb` as a batch job:
`run_experiments.py` (the compute + plots) driven by `hdc_nystrom.slurm` (the LS6 job).

## Why CPU, not GPU

The bottleneck is `nystrom_encoder.kmer_spectrum` — a pure-Python character loop run over
~100K training lines × 10 languages for every `(D, k)` point. That is CPU-bound Python, and
there is **no GPU code anywhere in this project**; an A100 would sit idle on the slow part.
What *is* exploitable is that the sweep's ~30 experiment points are fully independent, so we
fan them out across the 128 cores of one LS6 `normal`-queue node with a `multiprocessing.Pool`
(one worker per core, BLAS pinned to 1 thread per worker so they don't oversubscribe).

If you later port `kmer_spectrum` to a vectorized/GPU form, switch the queue to `gpu-a100`
and add CuPy — but as the code stands today, the CPU node is strictly the right choice.

## One-time setup (on a login node)

No conda required. The code's only third-party dependencies are `numpy` and `matplotlib`,
both installable straight into TACC's system Python:

```bash
module load python3
pip3 install --user numpy matplotlib   # only needed once; persists across jobs/sessions
```

`hdc_nystrom.slurm` also runs this check automatically at job start (so a fresh account
self-heals on first submit), but running it once yourself on the login node lets you catch
install issues before burning a queued job on it.

Before submitting, you may still need to:
- Set `#SBATCH -A YOUR_ALLOCATION` in `hdc_nystrom.slurm` if your TACC account requires an
  allocation flag (run `sbatch hdc_nystrom.slurm` once — if it errors asking for `-A`, you
  need it; check available allocations with `/usr/local/etc/taccinfo`).
- Optionally uncomment `--mail-user` for email notifications.

## Smoke test first (cheap)

On a login node or a short `development`-queue allocation, validate the pipeline on a reduced
grid before spending a full node:

```bash
module load python3
python3 run_experiments.py --quick --workers 4 --outdir results_smoketest
```

`--quick` runs a single small dimension, k=3 only, and a reduced bit-budget/ensemble grid —
enough to confirm imports, pickling, plotting, and JSON output all work.

## Submit the full run

```bash
sbatch hdc_nystrom.slurm
```

Monitor it:

```bash
squeue -u $USER                    # queue state
tail -f hdc_nystrom.<jobid>.out    # live progress (each point prints as it finishes)
```

## Outputs

Everything lands in `results_<jobid>/`:

- `results.json` — every accuracy number, per-language breakdowns, and the landmark-count
  timing measurements, in one machine-readable file.
- `accuracy_vs_dimension_comparison.png`
- `accuracy_vs_hamming_precision_comparison.png`
- `accuracy_vs_dimension_by_k.png`
- `accuracy_vs_dimension_stratified_comparison.png`
- `accuracy_vs_num_landmark.png` and `time_vs_num_landmark.png`
- `accuracy_vs_bit_budget.png` — bit-budget × ensemble-size sweep: for each total bit budget in
  `{1024, 512, 256, 128}`, accuracy vs. number of independent encoders sharing that budget
  (`{1, 3, 4, 5}`; 1 = single full-width encoder, the baseline for that budget). This
  supersedes the notebook's original single-budget (D=1024-only) small-D-encoder ensembling
  plot with a generalized multi-budget version.
- `accuracy_vs_num_landmark_fine.png` and `time_vs_num_landmark_fine.png` — round-2 finer
  landmark-count sweep (200..2000, step 200) at fixed D=1024, k=3.

These are the same figures the notebook's plotting cells produce (bit-budget sweep
generalized as above), plus the round-2 finer landmark sweep.

## Tuning

- `-t 03:00:00` in the `.slurm` is generous; the full sweep (round 1 + round 2, ~75 points)
  is well under that on 128 cores. Shrink it for faster scheduling once you've seen a real
  run's wall time (printed at the end).
- `--workers` defaults to `SLURM_CPUS_ON_NODE`. Leave it; the script picks the node's core
  count automatically.
- To add/remove experiment points, edit the grid constants at the top of `run_experiments.py`.

---

# Running the UWaveGestureLibrary sweep (permutation baseline vs. Nystrom+GAK)

A second, independent experiment: gesture classification on UWaveGestureLibrary (8 classes,
3-axis accelerometer), comparing a multi-channel permutation HDC baseline
(`permutation_encoder.py`) against a Nystrom + GAK (Global Alignment Kernel) HDC encoder
(`gak_encoder.py`). Driven by `run_gesture_experiments.py` / `hdc_gesture.slurm`, same
Pool-over-one-128-core-node shape as the language sweep above.

**New third-party dependency**: this experiment needs `tslearn` (for GAK) in addition to
`numpy`/`matplotlib` — unlike the language sweep, this repo is no longer numpy+matplotlib-only.
Install it the same way:

```bash
module load python3
pip3 install --user numpy matplotlib tslearn
```

**Data**: `UWaveGestureLibraryAll/UWaveGestureLibraryAll_{TRAIN,TEST}.ts` stores each gesture
as X/Y/Z concatenated into one 945-length univariate series. `uwave_data.py` splits each row
back into genuine `(3 channels, 315 samples)` data before either encoder sees it — confirm this
directory is present (it's untracked; not part of git history) before running.

**Headline result**: accuracy vs. vector dimension `D`, both methods on one plot
(`accuracy_vs_dimension_gesture.png`), swept over
`{64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000}` — the upper end reaches D=10000, the
standard hypervector width used in binary-HDC time-series literature (record-based/
spatiotemporal HDC), so the sweep can show where accuracy actually plateaus.

**Secondary result**: Nystrom+GAK accuracy vs. number of landmarks at fixed D=1024
(`accuracy_vs_num_landmark_gesture.png`) — GAK's own core knob, alongside D.

**Tertiary result**: permutation baseline accuracy vs. `num_levels` (quantization resolution)
at fixed D=1024 (`accuracy_vs_num_levels_gesture.png`) — a free parameter this task introduces
that the language task never had (continuous accelerometer values must be quantized into
discrete levels before the permutation encoder's item-memory lookups apply).

GAK is expensive — empirically ~1.5ms per (example, landmark) pair on this data — and its cost
is set by the *number of examples and landmarks*, not by `D`. `run_gesture_experiments.py`
exploits this: for the dimension sweep it computes the GAK kernel matrices once (at a fixed
landmark count) and reuses them across every `D`, rather than recomputing GAK per `(D,
landmark)` point. Only the num-landmark sweep pays GAK's cost repeatedly, once per landmark
count tested.

Smoke test first, same pattern as above:

```bash
python3 run_gesture_experiments.py --quick --workers 4 --outdir results_gesture_smoketest
```

Submit the full run:

```bash
sbatch hdc_gesture.slurm
```

**Is HDC actually a good fit here?** Worth reading critically, not taking for granted:
- Permutation encoding needs quantizing continuous accelerometer data into discrete levels —
  a lossy step with its own free parameter (`num_levels`) that the language task never had.
- GAK is O(seriesLength²) per pair, and the Nystrom step still needs the full kernel matrix
  computed *before* any HDC benefit kicks in — this experiment tests whether Nystrom+GAK+HDC
  preserves GAK's accuracy with cheap Hamming-distance *inference*, not whether HDC makes
  gesture classification cheap end-to-end the way the language n-gram encoder is cheap to train.
- The Nystrom map's final `sign()` binarization discards magnitude information that a smooth
  kernel like GAK relies on more than the spectrum kernel's discrete bag-of-k-mers counts did —
  if Nystrom+GAK accuracy looks surprisingly low, this binarization step is the first place to
  look, even though no dedicated ablation isolating it is included in this sweep.

---

# Running the UWaveGestureLibrary SNN baseline (snnTorch)

A third point of comparison on the same UWaveGestureLibrary train/test split and accuracy
metric as the two HDC methods above — but a genuinely different model class: a small
feedforward spiking neural network (`snn_encoder.py`), trained end to end with backprop
through time, rather than a bundled hypervector. Driven by `run_gesture_snn_experiments.py` /
`hdc_gesture_snn.slurm`.

**Model**: each channel's raw 315-sample series is delta-modulation encoded into a spike train
(`spikegen.delta`), fed into `fc1 -> lif1 (Leaky) -> fc2 -> lif2 (Leaky)`, `lif2` having 8 output
neurons (one per gesture class). Classification is rate-coded: each class's output spikes are
summed over all 315 timesteps, and the argmax is the prediction. Trained with
`snntorch.functional.ce_rate_loss` + Adam.

**New third-party dependency**: `torch` + `snntorch`, in addition to `numpy`/`matplotlib` — this
is the only experiment in this repo that needs torch (the HDC encoders deliberately avoid it; see
`nystrom_encoder.py`'s docstring). Install it the same way:

```bash
pip install torch snntorch
```

**Compute target**: unlike the two CPU-only HDC sweeps, the SNN training loop is dense-matmul-
bound (fc layers applied every timestep, every epoch), so this is the first workload in the repo
that actually benefits from a GPU — `hdc_gesture_snn.slurm` targets TACC's `gpu-h100` partition.
**No wall-clock comparison is made against the HDC methods anywhere** (GPU vs CPU time isn't a
fair comparison) — only accuracy, which is hardware-independent.

**Headline result**: accuracy vs. hidden layer size (`accuracy_vs_hidden_size_snn.png`), swept
over `{16, 32, 64, 128}` — the SNN's own capacity knob, analogous in spirit to the HDC methods'
vector dimension `D` but not the same quantity, so it's reported on its own plot rather than
merged into `accuracy_vs_dimension_gesture.png`.

Smoke test first, same pattern as above:

```bash
python3 run_gesture_snn_experiments.py --quick --outdir results_gesture_snn_smoketest --device cpu
```

Submit the full run:

```bash
sbatch hdc_gesture_snn.slurm
```
