"""
Utility functions for the pyloric network parameter inference project.

Covers:
  - Network construction and simulation (jaxley wrapper)
  - Summary statistics (burst period, duty cycle, phase offsets)
  - Plotting helpers
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, List, Tuple

# ---------------------------------------------------------------------------
# Type aliases (avoid torch import which was crashing earlier kernels)
# ---------------------------------------------------------------------------
from numpy import ndarray

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pyloric(
    ts: ndarray,
    v: Union[ndarray, list],
    axs: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot voltage traces for all three pyloric neurons.

    Args:
        ts: Time array (ms).
        v: Voltage traces, shape (3, T) or list of 3 arrays.
        axs: Existing axes to plot on (3-element array).
        **kwargs: Passed to ax.plot.

    Returns:
        fig, axs
    """
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(12, 5), sharex=True, layout="constrained")
    else:
        fig = axs[0].get_figure()
    titles = ["AB/PD", "LP", "PY"]
    for ax, v_i, title in zip(axs, v, titles):
        ax.plot(ts, v_i, **kwargs)
        ax.set_ylabel("V (mV)")
        ax.set_title(title)
    axs[-1].set_xlabel("t (ms)")
    return fig, axs


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

    Args:
        g_us: Array of 7 synaptic conductances in µS.

    Returns:
        (net, params): Jaxley network with trainable params set to g_us.
    """
    import jaxley as jx
    import jax.numpy as jnp
    from jaxley_models import PyloricNetwork

    net = PyloricNetwork()
    net.record("v")

    # Make each edge independently trainable
    for i in range(7):
        row = net.edges.iloc[i]
        param_name = f"{row['type']}_gS"
        net.scope("global").edge(i).make_trainable(param_name)

    # Build parameter list and override with given values
    params = net.get_parameters()
    for j, p in enumerate(params):
        key = next(iter(p.keys()))
        params[j] = {key: jnp.array([g_us[j]])}

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
    net, params = build_network(g_us)
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
    v: np.ndarray,
    threshold: float = -20.0,
    min_isi_ms: float = 5.0,
    dt: float = 0.025,
) -> np.ndarray:
    """Detect spike times (upward threshold crossings) in a voltage trace.

    Args:
        v: 1-D voltage trace.
        threshold: Spike detection threshold in mV.
        min_isi_ms: Minimum inter-spike interval (ms) for de-bouncing.
        dt: Time step of the trace in ms.

    Returns:
        Array of spike times in ms.
    """
    above = v >= threshold
    crossings = np.nonzero(np.diff(above.astype(np.int8)) > 0)[0]
    spike_times = crossings * dt

    # De-bounce: remove spikes closer than min_isi_ms
    if len(spike_times) < 2:
        return spike_times
    keep = [True] + [
        spike_times[i] - spike_times[i - 1] >= min_isi_ms
        for i in range(1, len(spike_times))
    ]
    return spike_times[np.array(keep)]


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


def _burst_stats_single(
    v: np.ndarray,
    dt: float = 0.025,
    threshold: float = -20.0,
    max_intra_burst_isi: float = 200.0,
    burn_in_ms: float = 500.0,
) -> dict:
    """Compute burst statistics for a single neuron trace.

    Returns dict with: period_ms, duty_cycle, onset_times, n_bursts.
    NaN for all stats if fewer than 2 bursts detected after burn-in.
    """
    spike_times = detect_spikes(v, threshold=threshold, dt=dt)
    # discard transient
    spike_times = spike_times[spike_times >= burn_in_ms]
    bursts = detect_bursts(spike_times, max_intra_burst_isi=max_intra_burst_isi)

    if len(bursts) < 2:
        return {"period_ms": np.nan, "duty_cycle": np.nan, "onset_times": np.array([]), "n_bursts": len(bursts)}

    onset_times = np.array([b[0] for b in bursts])
    offset_times = np.array([b[-1] for b in bursts])

    ibis = np.diff(onset_times)
    period = float(np.mean(ibis))
    durations = offset_times - onset_times
    duty_cycle = float(np.mean(durations) / period) if period > 0 else np.nan

    return {"period_ms": period, "duty_cycle": duty_cycle, "onset_times": onset_times, "n_bursts": len(bursts)}


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
    threshold: float = -20.0,
    burn_in_ms: float = 500.0,
) -> np.ndarray:
    """Extract summary statistics from a simulated voltage trace.

    Statistics (9 values):
      [0] AB/PD burst period (ms)
      [1] AB/PD duty cycle
      [2] LP burst period (ms)
      [3] LP duty cycle
      [4] PY burst period (ms)
      [5] PY duty cycle
      [6] LP phase offset relative to AB/PD
      [7] PY phase offset relative to AB/PD
      [8] mean spike count per burst (averaged over all neurons)

    Args:
        v_sim: Simulated voltage, shape (3, T).
        dt: Time step of v_sim in ms.
        threshold: Spike detection threshold (mV).
        burn_in_ms: Time (ms) to discard as transient before computing stats.

    Returns:
        Array of 9 summary statistics (NaN where computation fails).
    """
    stats_per_neuron = [
        _burst_stats_single(v_sim[i], dt=dt, threshold=threshold, burn_in_ms=burn_in_ms)
        for i in range(3)
    ]

    result = np.full(9, np.nan)

    # Period and duty cycle per neuron
    for i, s in enumerate(stats_per_neuron):
        result[2 * i] = s["period_ms"]
        result[2 * i + 1] = s["duty_cycle"]

    # Phase offsets relative to AB/PD (index 0)
    ref_onsets = stats_per_neuron[0]["onset_times"]
    ref_period = stats_per_neuron[0]["period_ms"]
    result[6] = compute_phase_offset(ref_onsets, stats_per_neuron[1]["onset_times"], ref_period)
    result[7] = compute_phase_offset(ref_onsets, stats_per_neuron[2]["onset_times"], ref_period)

    # Mean spike count per burst averaged across neurons
    spike_counts = []
    for i in range(3):
        v_i = v_sim[i]
        spike_times = detect_spikes(v_i[int(burn_in_ms / dt):], threshold=threshold, dt=dt)
        bursts = detect_bursts(spike_times)
        if bursts:
            spike_counts.append(np.mean([len(b) for b in bursts]))
    result[8] = float(np.mean(spike_counts)) if spike_counts else np.nan

    return result


STAT_LABELS = [
    "AB/PD period (ms)",
    "AB/PD duty cycle",
    "LP period (ms)",
    "LP duty cycle",
    "PY period (ms)",
    "PY duty cycle",
    "LP phase",
    "PY phase",
    "Mean spikes/burst",
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
    import jaxley as jx
    import jax.numpy as jnp
    from jaxley_models import PyloricNetwork

    net = PyloricNetwork()
    net.record("v")
    for i in range(7):
        row = net.edges.iloc[i]
        net.scope("global").edge(i).make_trainable(f"{row['type']}_gS")

    param_keys = [next(iter(p.keys())) for p in net.get_parameters()]
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
        s = _burst_stats_single(v_sim[i], dt=dt, burn_in_ms=burn_in_ms)
        if np.isnan(s["period_ms"]) or s["n_bursts"] < 2:
            return False
    return True
