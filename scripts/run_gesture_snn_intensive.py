"""More layers, bigger hidden layer size. Less epochs.

Usage:
    python run_gesture_snn_intensive.py [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs a reduced hidden_size grid with fewer epochs, for a smoke test before
committing a full run.
"""

import run_gesture_snn as rgs

# ---------------------------------------------------------------------------
# Experiment grid.
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [128, 256, 512]
NUM_EPOCHS = 15
BATCH_SIZE = 32

# Decay rate knobs -- temporal information using different betas
BETA_DEC_4 = [0.95, 0.8, 0.65, 0.5]
BETA_ALT_4 = [0.9, 0.5, 0.9, 0.5]
BETA_DEC_5 = [0.9, 0.8, 0.7, 0.6, 0.5]
BETA_ALT_5 = [0.9, 0.5, 0.9, 0.5, 0.9]


MODEL_CONFIGS = {
    "MultiBeta4_dec": (se.SNNGestureClassifierMultiBeta4,
                       {"beta_1": BETA_DEC_4[0], "beta_2": BETA_DEC_4[1], "beta_3": BETA_DEC_4[2], "beta_out": BETA_DEC_4[3]}),
    "MultiBeta4_alt": (se.SNNGestureClassifierMultiBeta4,
                       {"beta_1": BETA_ALT_4[0], "beta_2": BETA_ALT_4[1], "beta_3": BETA_ALT_4[2], "beta+out": BETA_ALT_4[3]}),
    "MultiBeta5_dec": (se.SNNGestureClassifierMultiBeta5,
                       {"beta_1": BETA_DEC_5[0], "beta_2": BETA_DEC_5[1], "beta_3": BETA_DEC_5[2], "beta_4": BETA_DEC_5[3], "beta_out": BETA_DEC_5[4]}),
    "MultiBeta5_alt": (se.SNNGestureClassifierMultiBeta5,
                       {"beta_1": BETA_ALT_5[0], "beta_2": BETA_ALT_5[1], "beta_3": BETA_ALT_5[2], "beta_4": BETA_ALT_5[3], "beta_out": BETA_ALT_5[4]}),
}