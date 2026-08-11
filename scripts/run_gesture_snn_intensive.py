"""More layers, bigger hidden layer size, fewer epochs -- a faster/cheaper variant
of run_gesture_snn.py's sweep for quicker iteration on architecture choices.

Reuses run_gesture_snn.py's MODEL_CONFIGS (MultiBeta3/4/5, beta spaced 0.9->0.7
per layer) and HIDDEN_SIZES (starting at 512, halved per hidden layer) as-is --
only NUM_EPOCHS is reduced here.

Usage:
    python run_gesture_snn_intensive.py [--outdir DIR] [--quick] [--device cuda|cpu]

--quick runs a reduced hidden_size grid with fewer epochs, for a smoke test before
committing a full run.
"""
import run_gesture_snn as rgs

NUM_EPOCHS = 15

if __name__ == "__main__":
    rgs.NUM_EPOCHS = NUM_EPOCHS
    rgs.main()
