########################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from typing import Optional, Union, List, Tuple
import jax.numpy as jnp
from jaxley_models import PyloricNetwork # type: ignore

# ---------------------------------------------------------------------------
# Type aliases (avoid torch import which was crashing earlier kernels)
# ---------------------------------------------------------------------------

from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from torch import Tensor

## function given with notebook
def plot_pyloric(ts: Union[ndarray, Tensor], v: Union[ndarray, Tensor], axs: Optional[plt.Axes] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot the voltage of the pyloric network for each neuron.

    Args:
        ts: The time points to plot.
        v: The voltage of the pyloric network.
        axs: The axes to plot on. Allows to plot multiple traces in one figure.

    Returns:
        fig: The figure.
        axs: The axes.
    """
    if axs is None:

        fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True, layout='constrained')
    for ax_i, v_i in zip(axs, v): # type: ignore
        ax_i.plot(ts, v_i, **kwargs)
        ax_i.set_ylabel('V (mV)')
    axs[0].set_title(f'AB/PD') # type: ignore
    axs[1].set_title(f'LP') # type: ignore
    axs[2].set_title(f'PY') # type: ignore
    axs[2].set_xlabel('t (ms)') # type: ignore
    return fig, axs # type: ignore


# ---------------------------------------------------------------------------
# Network / simulator helpers
# ---------------------------------------------------------------------------

# Synapse order (matches edge indices 0-6 in PyloricNetwork):
#   0: GlutamatergicSynapse  ab_pd → lp
#   1: CholinergicSynapse     ab_pd → lp
#   2: GlutamatergicSynapse  ab_pd → py
#   3: CholinergicSynapse     ab_pd → py
#   4: GlutamatergicSynapse  lp    → ab_pd
#   5: GlutamatergicSynapse  lp    → py
#   6: GlutamatergicSynapse  py    → lp
SYNAPSE_LABELS = [
    "AB/PD→LP (Glut)",
    "AB/PD→LP (Chol)",
    "AB/PD→PY (Glut)",
    "AB/PD→PY (Chol)",
    "LP→AB/PD (Glut)",
    "LP→PY (Glut)",
    "PY→LP (Glut)",
]

# Starting conductances (µS) verified to produce partial pyloric bursting
# with the default Prinz 2004 membrane parameters.
# AB/PD→LP and AB/PD→PY inhibition drives the rhythm; weaker feedback closes it.
PRINZ_G_INIT_US = np.array([
    0.10,   # ab_pd → lp  glut  (strong forward inhibition)
    0.10,   # ab_pd → lp  chol
    0.05,   # ab_pd → py  glut
    0.05,   # ab_pd → py  chol
    0.05,   # lp → ab_pd  glut  (feedback)
    0.05,   # lp → py     glut
    0.10,   # py → lp     glut
], dtype=np.float64)

LOG10_G_BOUNDS = (-5.0, 1.0)  # log10(µS) prior bounds


def build_network(g_us: np.ndarray):
    """Construct a PyloricNetwork with per-synapse conductances. 
    This function is just initializing the network paramenters and making them trainable.

    Args:
        g_us: INITIAL Array of 7 synaptic conductances in µS.

    Returns:
        (net, params): Jaxley network with trainable params set to g_us.
    """

    net = PyloricNetwork()
    net.record("v") # teh simulator records the voltage

    # Make each edge (synapse) independently trainable
    for i in range(7):
        row = net.edges.iloc[i] # retrieves each synapse (each row)
        param_name = f"{row['type']}_gS" # we define the CONDUCTANCE variable for each synapse
        net.scope("global").edge(i).make_trainable(param_name) # For each synapse, this tells Jaxley: "Do not lock this conductance value at
           # its default. Let my gradient descent algorithm change this number to see if it makes the network behave better."

    # Build parameter list and override with given values

    params = net.get_parameters() # returns A DICTIONARY of the trainable parameters stated above
    for j, p in enumerate(params):
        key = next(iter(p.keys())) # this retrieves the type of synapse (the key)
        params[j] = {key: jnp.array([g_us[j]])} # it updates the value of the 

    return net, params


def simulate(
    g_us: np.ndarray,
    t_max: float = 4000.0,
    dt: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the pyloric network simulator with given synaptic conductances.

    Args:
        g_us: Array of 7 synaptic conductances in µS.
        t_max: Simulation duration in ms.
        dt: Integration time step in ms.

    Returns:
        (t_sim, v_sim): Time array and voltage array (3, T).
    """
    import jaxley as jx
    net, params = build_network(g_us) #initialize and make them trainable
    v = jx.integrate(net, params=params, t_max=t_max, delta_t=dt)
    t = np.arange(v.shape[1]) * dt
    return t, np.array(v)


def simulate_log10(
    log10_g: np.ndarray,
    t_max: float = 4000.0,
    dt: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate from log10(conductance) representation."""
    return simulate(10.0 ** log10_g, t_max=t_max, dt=dt)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def detect_spikes(
    v: np.ndarray, dt: float = 0.025, lockout_ms: float = 5.0, threshold: float = 0
) -> np.ndarray:
    """Detect spikes in a 1D intracellular voltage trace.

    Parameters
    ----------
    v: np.ndarray
        1-D voltage trace from a single neuron.
    dt: float
        Time step of the trace in ms (matches Jaxley output).
    lockout_ms: float
        Minimum inter-spike interval (refractory period) in ms.
    threshold: float
        Voltage threshold for spike detection in mV.

    Returns
    -------
    t: np.ndarray
        Array of spike times in ms.
    """
    
    # Convert lockout time (ms) to samples
    refractory_period_samples = lockout_ms / dt
    
    # Find all spikes
    s, _ = signal.find_peaks(
        v, 
        height=threshold, 
        distance=refractory_period_samples
    )

    # Convert sample indices to times (in ms) and return ONLY this array
    t = s * dt

    return t


'''
def detect_bursts(
    spike_times: np.ndarray,
    max_intra_burst_isi: float = 200.0,
    min_spikes_per_burst: int = 1,
) -> List[np.ndarray]:
    """Group spike times into bursts.

    Args:
        spike_times: Sorted spike times in ms.
        max_intra_burst_isi: Maximum ISI inside a burst (ms).
        min_spikes_per_burst: Minimum number of spikes to count as a burst.

    Returns:
        List of arrays, each containing spike times belonging to one burst.
    """
    if len(spike_times) == 0: 
        return []

    bursts: List[List[float]] = [[spike_times[0]]]
    for t in spike_times[1:]:
        if t - bursts[-1][-1] <= max_intra_burst_isi:
            bursts[-1].append(t)
        else:
            bursts.append([t])

    bursts_filtered = [np.array(b) for b in bursts if len(b) >= min_spikes_per_burst]
    return bursts_filtered
'''

def detect_bursts(spike_times: np.ndarray, ISI_threshold: float = 200.0, min_spikes_per_burst: int = 1):
    """Group spike times into bursts.
    Args:
        spike_times: Sorted spike times in ms.
        ISI_threshold: Maximum ISI inside a burst (ms).
        min_spikes_per_burst: Minimum number of spikes to count as a burst.
    Returns:
        List of arrays, each containing spike times belonging to one burst.
    """
    
    # Catch empty arrays to avoid errors
    if len(spike_times) == 0: 
        return []

    bursts = []
    current_burst = [spike_times[0]] # Start the first burst with the very first spike

    # Loop through the spikes, starting from the second one (index 1)
    for i in range(1, len(spike_times)):
        
        # Calculate ISI (Inter-Spike Interval) using simple subtraction
        current_ISI = spike_times[i] - spike_times[i-1]
        
        if current_ISI < ISI_threshold: 
            # if the spike is right after another one, add it to the burst
            current_burst.append(spike_times[i])
        else: 
            # It was too slow. Save the finished burst, and start a new one
            bursts.append(current_burst) 
            current_burst = [spike_times[i]]
            
    # Don't forget to append the very last burst when the loop finishes!
    bursts.append(current_burst)
    
    # Filter out bursts that don't have enough spikes, and convert them to arrays
    bursts_filtered = [np.array(b) for b in bursts if len(b) >= min_spikes_per_burst]
    
    return bursts_filtered



def burst_stats_single(
    v: np.ndarray,
    dt: float = 0.025,
    threshold: float = 0.0,
    max_intra_burst_isi: float = 200.0,
    burn_in_ms: float = 500.0,
) -> dict:
    """Compute burst statistics for a single neuron trace.

    Returns dict with: period_ms, duty_cycle, onset_times, n_bursts (no. oof spikes / burst).
    NaN for all stats if fewer than 2 bursts detected after burn-in.
    """
    spike_times = detect_spikes(v, dt=dt, threshold=threshold)
    # discard transient
    #spike_times = spike_times[spike_times >= burn_in_ms] #removed it cause it led to artifacts (first AB/PD burst not detected, except single spike, skewing all stats)
    bursts = detect_bursts(spike_times, ISI_threshold=max_intra_burst_isi)

    if len(bursts) < 2:
        return {"period_ms": np.nan, "duty_cycle": np.nan, "onset_times": np.array([]), "n_bursts": len(bursts), "n_spikes": np.array([np.nan])}

    onset_times = np.array([b[0] for b in bursts])
    offset_times = np.array([b[-1] for b in bursts])

    ibis = np.diff(onset_times) # inter-burst intervals, does NOT take all pairwise diffs, just consec.
    period = float(np.mean(ibis))
    durations = offset_times - onset_times
    duration = np.mean(durations)
    duty_cycle = float(duration / period) if period > 0 else np.nan

    buffer_idx = int(10.0 / dt) 
    # 2. Convert to indices and apply the buffer (using max/min to avoid going out of bounds)
    idx_pairs = [(max(0, int(b[0]/dt) - buffer_idx), min(len(v), int(b[-1]/dt) + buffer_idx)) for b in bursts]

    # 3. Calculate the mean peak-to-peak amplitude
    mean_burst_amp = np.mean([np.max(v[start:end]) - np.min(v[start:end]) for start, end in idx_pairs])

    n_spikes = np.array([len(b) for b in bursts])

    return {"period_ms": period, "burst_duration": duration, "duty_cycle": duty_cycle, "mean_p2p_amp": mean_burst_amp, "n_bursts": len(bursts), "n_spikes": n_spikes, "onset_times": onset_times}


def compute_phase_offset(
    ref_onsets: np.ndarray,
    target_onsets: np.ndarray,
    period: float,
) -> float:
    """Compute mean phase offset of target bursts relative to reference.

    Phase is in [0, 1). Uses the nearest reference burst for each target.
    """
    if len(ref_onsets) == 0 or len(target_onsets) == 0 or np.isnan(period) or period <= 0:
        return np.nan

    phases = []
    for t_on in target_onsets:
        # Find nearest reference onset before or at this time
        idx = np.searchsorted(ref_onsets, t_on, side="right") - 1
        if idx < 0:
            continue
        phase = ((t_on - ref_onsets[idx]) % period) / period
        phases.append(phase)

    return float(np.mean(phases)) if phases else np.nan


def summary_statistics(
    v_sim: np.ndarray,
    dt: float = 0.025,
    threshold: float = 0.0,
    burn_in_ms: float = 500.0,
) -> np.ndarray:
    """Extract summary statistics from a simulated voltage trace.

    Statistics (9 values):
      [0] period (ms)
      [1] AB/PD duty cycle
      [2] LP duty cycle
      [3] PY duty cycle
      [4] LP phase offset relative to AB/PD
      [5] PY phase offset relative to AB/PD
      [6] AB/PD mean spikes per burst
      [7] LP mean spikes per burst
      [8] PY mean spikes per burst

    Args:
        v_sim: Simulated voltage, shape (3, T).
        dt: Time step of v_sim in ms.
        threshold: Spike detection threshold (mV).
        burn_in_ms: Time (ms) to discard as transient before computing stats.

    Returns:
        Array of 9 summary statistics (NaN where computation fails).
    """
    stats_per_neuron = [
        burst_stats_single(v_sim[i], dt=dt, burn_in_ms=burn_in_ms, threshold=threshold)
        for i in range(3)
    ]

    result = np.full(9, np.nan)

    # Median period across neurons (all should have the same period)
    result[0] = jnp.mean(np.array([s["period_ms"] for s in stats_per_neuron]))

    # Period and duty cycle per neuron
    for i, s in enumerate(stats_per_neuron):
        result[i + 1] = s["duty_cycle"]

    # Phase offsets relative to AB/PD (index 0)
    ref_onsets = stats_per_neuron[0]["onset_times"]
    ref_period = stats_per_neuron[0]["period_ms"]
    result[4] = compute_phase_offset(ref_onsets, stats_per_neuron[1]["onset_times"], ref_period)
    result[5] = compute_phase_offset(ref_onsets, stats_per_neuron[2]["onset_times"], ref_period)

    # mean spike count calculation moved to burst_stats_single
    # add averaged count per neuron to result[6,7,8]
    mean_spike_counts = [jnp.median(s["n_spikes"]) if len(s["n_spikes"]) > 0 else np.nan for s in stats_per_neuron]
    result[6:9] = mean_spike_counts

    return result


STAT_LABELS = [
    "period (ms)",
    "AB/PD duty cycle",
    "LP duty cycle",
    "PY duty cycle",
    "LP phase",
    "PY phase",
    "AB/PD spikes/burst",
    "LP spikes/burst",
    "PY spikes/burst",
]


def print_stats(stats: np.ndarray, label: str = "") -> None:
    """Pretty-print a summary statistics vector."""
    prefix = f"[{label}] " if label else ""
    for name, val in zip(STAT_LABELS, stats):
        print(f"  {prefix}{name}: {val:.3f}" if not np.isnan(val) else f"  {prefix}{name}: NaN")


# ---------------------------------------------------------------------------
# Gradient descent helpers (differentiable via jaxley + JAX)
# ---------------------------------------------------------------------------

def build_network_for_grad():
    """Create a single shared network for gradient-based optimization.

    Returns (net, param_template) where param_template is the default
    parameter list from which new parameters can be constructed.
    """

    net = PyloricNetwork()
    net.record("v")
    for i in range(7):
        row = net.edges.iloc[i]
        net.scope("global").edge(i).make_trainable(f"{row['type']}_gS")

    param_keys = [next(iter(p.keys())) for p in net.get_parameters()] # gets each type of conductance 
                                                                    # (e.g., GlutamatergicSynapse_gS)
    return net, param_keys


def make_params_from_log10g(param_keys: list, log10_g):
    """Convert log10 conductance array to jaxley params list.

    Args:
        param_keys: List of 7 parameter name strings.
        log10_g: JAX array of shape (7,) in log10(µS).

    Returns:
        List of 7 single-element dicts suitable for jx.integrate.
    """
    import jax.numpy as jnp
    g = 10.0 ** log10_g
    return [{key: jnp.array([g[j]])} for j, key in enumerate(param_keys)]


def mse_voltage_loss(log10_g, net, param_keys, v_obs_jax, subsample: int = 10):
    """Differentiable MSE loss on voltage traces.

    Args:
        log10_g: JAX array of shape (7,), log10 conductances.
        net: Shared jaxley Network (static structure).
        param_keys: Parameter name list from build_network_for_grad.
        v_obs_jax: Observed voltages, shape (3, T_obs), JAX array.
        subsample: Subsampling factor to match observation dt.

    Returns:
        Scalar MSE loss.
    """
    import jaxley as jx
    import jax.numpy as jnp

    params = make_params_from_log10g(param_keys, log10_g)
    t_max = v_obs_jax.shape[1] * subsample * 0.025  # ms
    v_sim = jx.integrate(net, params=params, t_max=t_max, delta_t=0.025)
    # Subsample simulation to match observation sampling rate
    v_sim_sub = v_sim[:, ::subsample]
    n = min(v_sim_sub.shape[1], v_obs_jax.shape[1])
    return jnp.mean((v_sim_sub[:, :n] - v_obs_jax[:, :n]) ** 2)


# ---------------------------------------------------------------------------
# SBI helpers
# ---------------------------------------------------------------------------

def sbi_simulator(params_np: np.ndarray, t_max: float = 4000.0, dt: float = 0.025) -> np.ndarray:
    """Single simulation for SBI: parameters → summary statistics.

    Args:
        params_np: Array of shape (7,) with log10 conductances.
        t_max: Simulation duration in ms.
        dt: Integration time step in ms.

    Returns:
        Summary statistics array of shape (9,), may contain NaN.
    """
    g_us = 10.0 ** params_np
    # clip to valid range
    g_us = np.clip(g_us, 1e-6, 15.0)
    try:
        _, v_sim = simulate(g_us, t_max=t_max, dt=dt)
        return summary_statistics(v_sim, dt=dt)
    except Exception:
        return np.full(9, np.nan)


def check_bursting(v_sim: np.ndarray, dt: float = 0.025, burn_in_ms: float = 500.0) -> bool:
    """Return True if all three neurons show rhythmic bursting."""
    for i in range(3):
        s = burst_stats_single(v_sim[i], dt=dt, burn_in_ms=burn_in_ms)
        if np.isnan(s["period_ms"]) or s["n_bursts"] < 2:
            return False
    return True

# ---------------------------------------------------------------------------
# plot the differntial evolution loss landscape
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_loss_landscape(history_params, history_loss, param_names):
    """
    Visualizes parameter sensitivity with logarithmic color scaling and
    explicit separation of penalized network topologies.
    """
    P = np.array(history_params)
    L = np.array(history_loss)

    n_params = P.shape[1]
    cols = 4
    rows = int(np.ceil(n_params / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 3 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()

    # Define the penalty threshold we established in the loss function
    penalty_threshold = 10.0

    # Create boolean masks to isolate the populations
    valid_mask = L < penalty_threshold
    failed_mask = L >= penalty_threshold

    # Calculate global min and max for the valid loss to lock the colorbar scale
    if np.any(valid_mask):
        vmin_valid = L[valid_mask].min()
        vmax_valid = L[valid_mask].max()

    for i in range(n_params):
        ax = axes[i]

        # 1. Plot the penalized parameter sets (the "stripes")
        if np.any(failed_mask):
            ax.scatter(
                P[failed_mask, i],
                L[failed_mask],
                alpha=0.15,
                s=10,
                color="gray",
                edgecolors="none",
                zorder=1,
            )

        # 2. Plot the valid parameter sets with a logarithmic colormap
        if np.any(valid_mask):
            sc = ax.scatter(
                P[valid_mask, i],
                L[valid_mask],
                alpha=0.7,
                s=20,
                edgecolors="none",
                c=L[valid_mask],
                cmap="viridis",
                norm=LogNorm(vmin=vmin_valid, vmax=vmax_valid),
                zorder=2,
            )

        # Format axes
        ax.set_title(
            param_names[i] if i < len(param_names) else f"Param {i}", fontsize=10
        )
        if i in [3, 4, 5, 6]:
            ax.set_xlabel("log10(g) [µS]")
        ax.set_yscale("log")

        # Draw a delimiter line indicating the penalty threshold
        ax.axhline(
            penalty_threshold, color="red", linestyle="--", alpha=0.3, linewidth=1
        )

        if i % cols == 0:
            ax.set_ylabel("Feature Loss")

        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    # Attach a global colorbar
    if np.any(valid_mask):
        cbar = fig.colorbar(
            sc,
            ax=axes.ravel().tolist(),
            orientation="vertical",
            fraction=0.015,
            pad=0.04,
        )
        cbar.set_label("Feature Loss (Log Scale)")

    plt.show()