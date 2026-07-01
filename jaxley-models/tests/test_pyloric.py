# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pandas as pd
import pytest

from jaxley_models.pyloric import PyloricNetwork


def _load_params(data_dir, idx):
    """Load parameter set at row ``idx`` from the test CSV.

    Returns a fresh (unmodified) ``pd.Series`` with two-level column index:
    membrane conductances in mS/cm², synaptic conductances in log(mS).
    """
    return pd.read_csv(
        os.path.join(data_dir, "pyloric_params_20250604.csv"),
        index_col=0,
        dtype=float,
        header=[0, 1],
    ).loc[idx]


def _run_jaxley(params, temp, dt, t_max):
    """Run jaxley PyloricNetwork and return voltage array of shape (3, T) in mV.

    Parameters
    ----------
    params:
        ``pd.Series`` from ``_load_params`` — membrane conductances in mS/cm²,
        synaptic conductances in log(mS).  This Series is copied internally and
        not mutated.
    """
    params = params.copy()

    net = PyloricNetwork()
    net.record("v")
    # Channels share a single "temperature" node parameter; synapses each have
    # their own prefixed temperature parameter.
    net.set("temperature", temp)
    net.set("GlutamatergicSynapse_temperature", temp)
    net.set("CholinergicSynapse_temperature", temp)

    # Membrane conductances: CSV is mS/cm², jaxley channels expect S/cm²
    for neuron in ["AB/PD", "LP", "PY"]:
        params[neuron] = params[neuron] * 1e-3  # mS/cm² → S/cm²

    # Synaptic conductances: CSV is log(mS), jaxley synapses expect µS
    params["Synapses"] = np.exp(params["Synapses"].values) * 1e3  # log(mS) → mS → µS

    for group_key, syn_or_channel_name in params.keys():
        if group_key != "Synapses":
            neuron = net.select(net.nodes[group_key.lower().replace("/", "_")])
            value = params[group_key, syn_or_channel_name].item()
            neuron.set(f"{syn_or_channel_name}_g{syn_or_channel_name}", value)

    for i, val in enumerate(params["Synapses"].values):
        synapse = net.select(edges=i)
        if "glut" in synapse.edges["type"].item().lower():
            synapse.set("GlutamatergicSynapse_gS", val)
        else:
            synapse.set("CholinergicSynapse_gS", val)

    v = jx.integrate(net, t_max=t_max, delta_t=dt)
    return np.asarray(v[:, :-1])  # drop the extra timestep jaxley appends, shape (3, T)


@pytest.mark.parametrize("temp,idx", [(283.0, 0), (283.0, 1), (273.0, 0)])
def test_pyloric(temp, idx):
    """Compare jaxley PyloricNetwork against the mackelab/pyloric Cython reference.

    The pyloric package (https://github.com/mackelab/pyloric) is the original
    Cython implementation that serves as the ground truth.  Install it with::

        uv pip install --no-deps git+https://github.com/mackelab/pyloric.git
        uv pip install cython setuptools tqdm

    The test is skipped automatically if the package is not available.

    Tolerances: MAE < 1 mV.  Small numerical differences cause spike-timing
    drift over long simulations, so errors below 1 mV are a good match.
    """
    pytest.importorskip(
        "pyximport",
        reason="pyximport not installed; cannot compile pyloric Cython simulator",
    )

    # Import here so the test is skipped gracefully if pyloric is absent.
    # Use importlib to load from the same directory as this test file so the
    # import works regardless of how pytest is invoked.
    try:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "pyloric_reference",
            os.path.join(os.path.dirname(__file__), "pyloric_reference.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        simulate_pyloric_reference = _mod.simulate_pyloric_reference
    except Exception as _e:
        pytest.skip(f"pyloric_reference could not be loaded: {_e}")

    tmax = 500.0
    dt = 0.025

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    params = _load_params(data_dir, idx)

    v_jaxley = _run_jaxley(params, temp, dt, tmax)
    v_pyloric = simulate_pyloric_reference(params, temp=temp, dt=dt, t_max=tmax)

    mae = np.mean(np.abs(v_jaxley - v_pyloric))
    assert mae < 1.0, (
        f"MAE between jaxley and pyloric reference is {mae:.3f} mV "
        f"(temp={temp} K, param set {idx})"
    )
