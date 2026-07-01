# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Wrapper around the mackelab/pyloric Cython simulator.

Accepts parameters in the same format as the test CSV produced by the pyloric
prior (membrane conductances in mS/cm², synaptic conductances in log(mS)) and
returns a voltage array with shape (3, T) in mV — matching jaxley output.

The pyloric package is imported without going through its ``__init__.py``
(which requires ``torch`` / ``sbi``) by pre-registering a stub module before
importing the ``simulator`` submodule via pyximport.

Units summary
-------------
Membrane conductances (CSV)  : mS/cm²   →  multiply by AREA (cm²) → mS  for sim_time
Synaptic conductances (CSV)  : log(mS)  →  negate exp to mS             for sim_time
Voltage output               : mV       (same as jaxley)
Time                         : ms       (same as jaxley)
Temperature                  : K        (same as jaxley)
"""

import sys
import types

import numpy as np
import pandas as pd

# Cell membrane area used by the pyloric simulator (hardcoded in simulator.pyx as C=0.628e-3 uF
# and in interface.py membrane_conductances_replaced_with_defaults as factor 0.628e-3 cm²).
_AREA = 0.628e-3  # cm²

# Default Q10 values — must match the values hardcoded in simulator.pyx and used by
# interface.py.  These are also the values jaxley_models/pyloric/channels.py uses.
_Q10_GBAR_MEM = np.ones(8) * 1.5  # g Q10 for all 8 membrane channels
_Q10_GBAR_SYN = np.ones(7) * 1.5  # g Q10 for all 7 synapses
_Q10_TAU_SYN = np.ones(7) * 1.7  # tau Q10 for all 7 synapses
_Q10_TAU_M = np.ones(7) * 2.4  # tau_m Q10 — overrides per channel below
_Q10_TAU_H = np.ones(4) * 2.8  # tau_h Q10
_Q10_TAU_CABUFF = np.ones(1) * 2.0

# Per-channel tau_m Q10 overrides (Caplan 2014), indices match channel order:
# Na(0), CaT(1), CaS(2), A(3), KCa(4), Kd(5), H(6)
_Q10_TAU_M[2] = 2.0  # CaS
_Q10_TAU_M[4] = 1.6  # KCa


def _ensure_simulator_pyx(pkg_dir: str) -> None:
    """Copy simulator.pyx into the installed pyloric package if it is missing.

    The pyloric package is installed without building the Cython extension
    (``--no-deps`` skips the build step).  ``simulator.pyx`` must be present
    in the package directory so that pyximport can compile it at import time.

    We locate the source file by reading the ``direct_url.json`` metadata that
    uv writes when installing from a git URL, then searching the uv git cache
    for the matching commit.
    """
    import glob
    import json
    import shutil
    from pathlib import Path

    pyx_dst = Path(pkg_dir) / "simulator.pyx"
    if pyx_dst.exists():
        return  # already present — nothing to do

    # Read the installed commit ID from PEP 610 direct_url.json
    dist_info = next(Path(pkg_dir).parent.glob("pyloric-*.dist-info"), None)
    if dist_info is None:
        raise FileNotFoundError("pyloric dist-info not found; is pyloric installed?")
    direct_url = dist_info / "direct_url.json"
    if not direct_url.exists():
        raise FileNotFoundError(f"No direct_url.json in {dist_info}")

    info = json.loads(direct_url.read_text())
    commit = info.get("vcs_info", {}).get("commit_id", "")
    if not commit:
        raise ValueError(f"Could not read commit ID from {direct_url}")

    # Search the uv git cache for simulator.pyx at this commit
    short = commit[:7]
    candidates = glob.glob(
        str(
            Path.home()
            / ".cache"
            / "uv"
            / "git-v0"
            / "checkouts"
            / "*"
            / short
            / "pyloric"
            / "simulator.pyx"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"simulator.pyx not found in uv cache for commit {commit}.\n"
            "Make sure pyloric was installed via uv from the git URL."
        )

    shutil.copy(candidates[0], pyx_dst)


def _get_sim_time():
    """Import sim_time from pyloric.simulator via pyximport, bypassing __init__.py."""
    import importlib.util

    import numpy as _np
    import pyximport

    pyximport.install(
        setup_args={"include_dirs": _np.get_include()},
        reload_support=True,
        language_level=3,
    )

    # Register a stub for the pyloric package so Python does not execute
    # pyloric/__init__.py (which unconditionally imports torch and sbi).
    if "pyloric" not in sys.modules:
        spec = importlib.util.find_spec("pyloric")
        if spec is None:
            # If pyloric isn't installed in site-packages, create a lightweight
            # package directory there and copy our vendored simulator.pyx into
            # it. This avoids requiring pip inside the uv-managed venv.
            import sysconfig
            from pathlib import Path
            import shutil

            site_packages = Path(sysconfig.get_paths()["purelib"])
            pkg_dir = site_packages / "pyloric"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal __init__.py that does not import heavy deps.
            init_py = pkg_dir / "__init__.py"
            if not init_py.exists():
                init_py.write_text(
                    "# Minimal pyloric stub installed by tests to compile simulator.pyx\n"
                )

            # Ensure simulator.pyx is present in the package: prefer a vendored
            # copy under tests/pyloric, then fall back to uv cache lookup.
            vend = Path(__file__).parent / "pyloric_reference" / "simulator.pyx"
            if vend.exists():
                shutil.copy(vend, pkg_dir / "simulator.pyx")
            else:
                # Fallback to uv cache lookup using existing helper
                _ensure_simulator_pyx(pkg_dir)

            stub = types.ModuleType("pyloric")
            stub.__path__ = [str(pkg_dir)]
            stub.__package__ = "pyloric"
            sys.modules["pyloric"] = stub
        else:
            pkg_dir = list(spec.submodule_search_locations or [])[0]
            _ensure_simulator_pyx(pkg_dir)

    from pyloric.simulator import sim_time  # compiled on first import by pyximport

    return sim_time


def _build_conns(
    syn_log_ms, e_glut=-70.0, e_chol=-80.0, kminus_glut=40.0, kminus_chol=100.0
):
    """Build the (7, 7) connection matrix expected by sim_time.

    Parameters
    ----------
    syn_log_ms:
        1-D array of 7 synaptic conductances in log(mS), ordered as:
        AB-LP(glut), PD-LP(chol), AB-PY(glut), PD-PY(chol),
        LP-PD(glut), LP-PY(glut), PY-LP(glut).
    """
    # Convert log(mS) → mS, then negate (sim_time convention: negative = inhibitory)
    g = -np.exp(syn_log_ms)

    # Columns: [post, pre, strength(mS), E_syn(mV), kminus(ms⁻¹), Vth(mV), Delta(mV)]
    vth = -35.0
    delta = 5.0
    return np.array(
        [
            [1, 0, g[0], e_glut, kminus_glut, vth, delta],  # AB->LP  glut
            [1, 0, g[1], e_chol, kminus_chol, vth, delta],  # AB->LP  chol  (PD-LP)
            [2, 0, g[2], e_glut, kminus_glut, vth, delta],  # AB->PY  glut
            [2, 0, g[3], e_chol, kminus_chol, vth, delta],  # AB->PY  chol  (PD-PY)
            [0, 1, g[4], e_glut, kminus_glut, vth, delta],  # LP->AB  glut  (LP-PD)
            [2, 1, g[5], e_glut, kminus_glut, vth, delta],  # LP->PY  glut
            [1, 2, g[6], e_glut, kminus_glut, vth, delta],  # PY->LP  glut
        ],
        dtype=float,
    )


def simulate_pyloric_reference(
    params_row: pd.Series,
    temp: float = 283.0,
    dt: float = 0.025,
    t_max: float = 500.0,
) -> np.ndarray:
    """Run the pyloric Cython reference simulator.

    Parameters
    ----------
    params_row:
        A ``pd.Series`` with a two-level MultiIndex matching the CSV produced by
        the pyloric prior.  Membrane conductances under keys
        ``('AB/PD', channel)``, ``('LP', channel)``, ``('PY', channel)`` are in
        **mS/cm²**.  Synaptic conductances under ``('Synapses', name)`` are in
        **log(mS)** (natural log).
    temp:
        Temperature in Kelvin.
    dt:
        Integration timestep in milliseconds.
    t_max:
        Simulation duration in milliseconds.

    Returns
    -------
    np.ndarray
        Voltage array of shape ``(3, T)`` in mV, where T = len(np.arange(0, t_max, dt)).
        Row order: AB/PD (0), LP (1), PY (2).
    """
    sim_time = _get_sim_time()

    t = np.arange(0, t_max, dt)
    n_steps = len(t)

    # Zero-noise input currents
    I = np.zeros((3, n_steps), dtype=float)

    # Build modelx: shape (3, 8), values in mS (mS/cm² × area cm²)
    channel_order = ["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"]
    neuron_order = ["AB/PD", "LP", "PY"]
    modelx = (
        np.array(
            [
                [params_row[neuron, ch] for ch in channel_order]
                for neuron in neuron_order
            ],
            dtype=float,
        )
        * _AREA
    )  # mS/cm² × cm² = mS

    # Build synapse connection matrix
    conns = _build_conns(params_row["Synapses"].values)

    data = sim_time(
        dt,
        t,
        I,
        modelx,
        conns,
        g_q10_conns_gbar=_Q10_GBAR_SYN,
        g_q10_conns_tau=_Q10_TAU_SYN,
        g_q10_memb_gbar=_Q10_GBAR_MEM,
        g_q10_memb_tau_m=_Q10_TAU_M,
        g_q10_memb_tau_h=_Q10_TAU_H,
        g_q10_memb_tau_CaBuff=_Q10_TAU_CABUFF,
        temp=temp,
        num_energy_timesteps=0,
        num_energyscape_timesteps=0,
        verbose=False,
    )

    return data["Vs"]  # shape (3, T), mV
