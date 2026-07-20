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
