"""Simple SNN using PyNN"""

import pyNN.nest as sim
sim.setup(timestep=0.1)

# Somewhat arbitrary
cell_type = sim.IF_curr_exp(v_rest=-65, v_thresh=-55, v_reset=-65,
                            tau_refrac=1, tau_m=10, cm=1, i_offset=1.1)