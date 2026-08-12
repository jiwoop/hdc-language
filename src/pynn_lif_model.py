"""
Builds a PyNN network equivalent to a trained sparch LIF SNN, from weights
exported by sparch/export_weights.py (see hdc-language/README.md for the
full pipeline: sparch training -> export_weights.py -> this module).

Pure Python/numpy -- no torch dependency. Parameterized by an already-
imported PyNN backend module, so build_network() works identically whether
called with pyNN.nest (local verification) or pyNN.spiNNaker (real hardware
batch job): this is standard PyNN backend portability.

Architecture (see hdc-language README and the project plan for the full
rationale behind each choice):
  - Every layer, including the input layer, is a genuine PyNN Population.
    SHD input is already binned spikes, so layer 0 is a SpikeSourceArray
    built directly from the dataset's (times, units) pairs -- no host-side
    current-injection workaround needed (that would only be required for
    continuous-valued input like HD's mel-filterbank features).
  - Hidden/readout layers use IF_curr_exp (current-based synapses -- the
    structural match for sparch's Wx, which has no conductance/reversal-
    potential concept).
  - alpha (sparch's discrete-time decay) is converted to tau_m (PyNN's
    continuous-time membrane constant) via tau_m = -dt_ms / ln(alpha),
    since both describe the same exponential decay, just parameterized
    differently (alpha = exp(-dt_ms / tau_m)).
  - sparch's subtractive reset (spike subtracts a fixed 1.0 from ut before
    that step's decay) has no exact IF_curr_exp equivalent (which resets to
    an absolute v_reset). Approximated as v_thresh=1.0, v_reset=0.0,
    v_rest=0.0, tau_refrac at the simulator minimum -- a named
    approximation, expected to diverge most under high firing rates.
  - The readout layer's softmax-accumulation (smooth, non-spiking, requires
    comparing all output neurons' potentials against each other every
    timestep) has no PyNN neuron equivalent either -- no primitive computes
    softmax across a population. Approximated as a spiking IF_curr_exp
    population classified by spike-count argmax (rate coding) instead of
    softmax-argmax. This is the single largest expected source of accuracy
    divergence from the trained sparch model -- see verification/ for the
    agreement-rate check between the two decision rules.
"""
import numpy as np
from scipy.integrate import odeint

DT_MS = 14.0  # sparch SHD: max_time=1.4s / nb_steps=100, exact (not inferred)
TAU_SYN = 1.0  # ms, kept small since sparch has no separate synaptic filter
SIM_TIMESTEP_MS = 0.1  # coarser (e.g. 1.0ms) introduces visible error in the
# weight-scaling correspondence derived below -- verified empirically.


def load_weights(npz_path):
    data = np.load(npz_path)
    nb_layers = int(data["nb_layers"])
    layers = []
    for i in range(nb_layers):
        layers.append(
            {
                "W_eff": data[f"layer{i}_W_eff"],
                "b_eff": data[f"layer{i}_b_eff"],
                "alpha": data[f"layer{i}_alpha"],
                "threshold": float(data[f"layer{i}_threshold"]),
                "is_readout": bool(data[f"layer{i}_is_readout"]),
            }
        )
    return layers


def alpha_to_tau_m(alpha, dt_ms=DT_MS):
    """Per-neuron alpha (sparch discrete-time decay) -> tau_m (PyNN, ms)."""
    return -dt_ms / np.log(alpha)


def unit_weight_impulse_response(tau_m, tau_syn, dt_ms, n_points=5000):
    """
    Numerically integrates dv/dt = -v/tau_m + I, dI/dt = -I/tau_syn, with a
    unit-weight spike at t=0 (I(0)=1), and returns v(dt_ms) -- the response
    of an IF_curr_exp neuron's membrane potential, at the end of one sparch
    discrete step, to a single incoming spike of weight 1. Used to derive
    the weight scaling below; NOT approximated as a delta function, since
    tau_syn=1ms and tau_m up to ~350ms aren't different enough orders of
    magnitude for that approximation to hold (an earlier attempt assuming
    it was ~126x wrong, confirmed by this exact numerical check).
    """

    def deriv(y, t):
        v, I = y
        return [-v / tau_m + I, -I / tau_syn]

    t = np.linspace(0, dt_ms, n_points)
    sol = odeint(deriv, [0.0, 1.0], t)
    return sol[-1, 0]


def bias_to_current(b_eff, tau_m):
    """
    sparch's fixed point (steady-state u absent spiking) is u_inf = b_eff,
    from u = alpha*u + (1-alpha)*b_eff at equilibrium. A continuous IF_curr_exp
    neuron driven by constant current I has fixed point v_inf = I*tau_m.
    Setting I = b_eff/tau_m makes both systems converge to the same
    asymptotic value; since tau_m is shared, the approach dynamics (discrete
    vs continuous, same time constant) end up producing matching firing
    behavior. Verified empirically against the sparch numpy reference:
    exact spike-count match in 3/4 tested (alpha, b) combinations, off by
    one spike in the fourth (see plan file for the full validation).

    NOT (1-alpha)*b_eff and NOT raw b_eff -- both tried and confirmed wrong
    (the former still saturates ~17x too often; the latter ~90x).
    """
    return b_eff / tau_m


def weight_to_scaled_weight(W_eff, alpha, tau_m, tau_syn, dt_ms=DT_MS):
    """
    A single presynaptic spike with raw weight w should raise sparch's u by
    exactly (1-alpha)*w after one dt-window. r = unit_weight_impulse_response
    is the ACTUAL (numerically integrated) response of a continuous
    IF_curr_exp neuron to a unit-weight impulse under its real tau_syn decay
    -- so w_scaled = w * (1-alpha) / r reproduces the same delta-u at t=dt.
    r depends on tau_m (hence alpha), so this must be computed per neuron.

    Verified empirically: matched to ~1e-4 absolute error in v at t=dt
    against the target (1-alpha)*w, with SIM_TIMESTEP_MS=0.1 (see plan file).
    """
    hidden = W_eff.shape[0]
    r = np.array(
        [unit_weight_impulse_response(tau_m[i], tau_syn, dt_ms) for i in range(hidden)]
    )
    scale = (1.0 - alpha) / r  # (hidden,)
    return W_eff * scale[:, None]


def spike_times_to_pynn(times_sec, units, nb_units, dt_ms=DT_MS):
    """
    Converts an SHD example's raw (times_sec, units) arrays -- as read
    directly from the dataset's .h5 file, same format sparch's
    SpikingDataset.__getitem__ consumes -- into a per-neuron list of spike
    times in ms, suitable for pynn.SpikeSourceArray(spike_times=...).

    Returns a list of length nb_units, each entry a sorted list of spike
    times (ms) for that input neuron.
    """
    times_ms = np.asarray(times_sec) * 1000.0
    spike_lists = [[] for _ in range(nb_units)]
    for t, u in zip(times_ms, units):
        spike_lists[int(u)].append(float(t))
    return [sorted(s) for s in spike_lists]


def build_network(pynn, layers, input_spike_times, dt_ms=DT_MS, tau_syn=TAU_SYN):
    """
    pynn: an already-imported PyNN backend module (pyNN.nest, pyNN.spiNNaker, ...)
          with pynn.setup(...) already called by the caller.
    layers: list of layer dicts from load_weights()
    input_spike_times: list of per-neuron spike time lists (ms), length
                        layers[0]["W_eff"].shape[1] (nb_inputs)

    Returns (populations, projections, readout_population) -- caller is
    responsible for pynn.run(duration_ms) and reading spike counts off
    readout_population afterwards.
    """
    nb_inputs = layers[0]["W_eff"].shape[1]
    assert len(input_spike_times) == nb_inputs

    populations = []
    projections = []

    input_pop = pynn.Population(
        nb_inputs, pynn.SpikeSourceArray(spike_times=input_spike_times)
    )
    input_pop.record("spikes")
    populations.append(input_pop)

    prev_pop = input_pop
    readout_pop = None

    for layer in layers:
        hidden = layer["W_eff"].shape[0]
        alpha = layer["alpha"]
        tau_m = alpha_to_tau_m(alpha, dt_ms=dt_ms)

        # See bias_to_current / weight_to_scaled_weight docstrings for the
        # derivation and empirical validation of these two scalings -- both
        # required to make sparch's discrete per-step update correspond to
        # continuous-time IF_curr_exp integration (a naive (1-alpha) flat
        # scale, or raw injection, were both tried and confirmed wrong by
        # 1-2 orders of magnitude; see the plan file for the full record).
        W_scaled = weight_to_scaled_weight(layer["W_eff"], alpha, tau_m, tau_syn, dt_ms)
        b_scaled = bias_to_current(layer["b_eff"], tau_m)

        cell_params = {
            "tau_m": tau_m,
            "tau_syn_E": tau_syn,
            "tau_syn_I": tau_syn,
            "v_thresh": 1.0,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "tau_refrac": 0.1,  # simulator minimum; sparch has no refractory concept
        }
        pop = pynn.Population(hidden, pynn.IF_curr_exp(**cell_params))
        # IF_curr_exp defaults to v=-65.0 at t=0 regardless of v_rest --
        # must set explicitly. sparch draws u0 ~ Uniform[0,1); 0.5 is used
        # here as a deterministic stand-in (exact RNG parity across
        # numpy/PyNN/NEST isn't achievable, and this only affects early
        # transient behavior, not steady-state dynamics).
        pop.initialize(v=0.5)
        pop.record("spikes")
        populations.append(pop)

        # PyNN's AllToAllConnector expects weight shape (pre.size, post.size)
        # i.e. (input, hidden); W_eff is (hidden, input) (sparch's nn.Linear
        # convention), so transpose here. sparch's trained W_eff has no
        # sign constraint (~half the entries are negative in practice), but
        # a single receptor_type in PyNN's current-based IF_curr_exp only
        # accepts one sign: "excitatory" requires weight >= 0, "inhibitory"
        # requires weight <= 0 (its own negative value directly gives
        # negative current -- there is no separate magnitude convention).
        # Split into two connectors, one per sign, zero elsewhere, so the
        # combined injected current matches the original signed weight.
        W = W_scaled.T
        W_exc = np.where(W > 0, W, 0.0)
        W_inh = np.where(W < 0, W, 0.0)

        proj_exc = pynn.Projection(
            prev_pop,
            pop,
            pynn.AllToAllConnector(),
            pynn.StaticSynapse(weight=W_exc),
            receptor_type="excitatory",
        )
        proj_inh = pynn.Projection(
            prev_pop,
            pop,
            pynn.AllToAllConnector(),
            pynn.StaticSynapse(weight=W_inh),
            receptor_type="inhibitory",
        )
        projections.append(proj_exc)
        projections.append(proj_inh)

        # Per-neuron bias has no Projection equivalent -- inject as a
        # constant DCSource per neuron, active for the full run. DCSource
        # only takes a scalar amplitude (not a per-neuron array), so create
        # one source per neuron and inject into that single-neuron view.
        for i in range(hidden):
            bias_source = pynn.DCSource(amplitude=float(b_scaled[i]), start=0.0)
            bias_source.inject_into(pop[i : i + 1])

        prev_pop = pop
        if layer["is_readout"]:
            readout_pop = pop

    return populations, projections, readout_pop


def spike_count_argmax(readout_pop, nb_outputs):
    """
    Reads recorded spikes off the readout population and returns the
    predicted class via spike-count argmax (rate coding) -- the deployable
    substitute for sparch's softmax-accumulation argmax. See module
    docstring for why this substitution is necessary and its expected
    accuracy impact.
    """
    spiketrains = readout_pop.get_data("spikes").segments[0].spiketrains
    counts = np.array([len(st) for st in spiketrains])
    assert len(counts) == nb_outputs
    return int(np.argmax(counts)), counts
