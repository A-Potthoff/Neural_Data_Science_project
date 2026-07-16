# Neural Data Science Project — Pyloric Circuit Parameter Inference

**Course**: Neural Data Science SoSe 2026, Uni Tübingen  
**Lecturers**: Prof. Dr. Philipp Berens, Dr. Jan Lause  
**Students**: Lucía Grande González, Andre Potthoff, Niclas Collmer

---

## Research question

> What synaptic conductances gave rise to a 4-second voltage recording from the stomatogastric ganglion (STG) pyloric circuit, and is there more than one viable parameter set (degeneracy)?

---

## Overview

We work with a simplified 3-neuron STG model (AB/PD, LP, PY) implemented in [Jaxley](https://jaxley.readthedocs.io/), a JAX-based differentiable neuroscience simulator. The model has **7 synaptic conductances** as free parameters (5 Glutamatergic + 2 Cholinergic).

Three inference approaches are compared:

| Method | Description |
|--------|-------------|
| **Differential evolution** | Gradient-free global optimiser (SciPy), weighted fractional-error loss on 9 burst features, 50 iterations × population size 5 |
| **Gradient descent** (empirical comparison) | Adam + gradient-norm clipping on a differentiable MSE-voltage loss, 3 restarts × 200 steps, log₁₀-conductance space |
| **SNPE** | Sequential Neural Posterior Estimation (sbi / SNPE-C), log-uniform prior, same 9-feature summary statistics, trained on ≥10,000 simulations |

Comparing the DE point estimate with the SNPE posterior directly answers the degeneracy question (Prinz et al. 2004); the gradient-descent run empirically confirms why DE was preferred over a differentiable approach in the first place.

---

## Why differential evolution — and not gradient descent?

We initially attempted gradient descent using JAX's automatic differentiation through the Jaxley simulator. This is technically possible for a raw MSE loss on voltage, but has two critical limitations for the STG problem:

**1. Discrete features break gradients.**  
Spike counts and burst periods are step functions of the conductances — their gradients are zero almost everywhere. A biologically meaningful loss (period, duty cycle, phase, spikes/burst) cannot be differentiated through; only a raw voltage MSE remains, which is a much weaker signal.

**2. Phase sensitivity traps gradient descent.**  
MSE on the raw voltage trace gives large gradients when the simulated rhythm is phase-shifted relative to the observation, even if the period and duty cycle are perfectly matched. This causes gradient descent to get stuck in local minima that look good numerically but are biologically wrong (e.g., the rhythm runs at the right speed but is half a cycle out of phase).

Differential evolution sidesteps both problems:
- It is **gradient-free** and treats the simulator as a black box.
- The **feature loss** (fractional error on period, duty cycles, phases, spike counts, with 3× weight on phase features) is invariant to absolute phase offset.
- As a **population-based global search**, it is far less likely to get stuck in local minima than gradient-based methods.

In practice, DE converged to a feature loss of **0.0073** (roughly 7% fractional error across all features), producing a trace that visually and quantitatively matches the observation.

**Empirical confirmation** (notebook section 5b): we also ran gradient descent for real — Adam (lr=0.02, grad-norm clipping) on a subsampled MSE-voltage loss, 3 restarts from different initialisations, 200 steps each. Final losses: **256.8, 248.3, 322.95 mV²** (vs. the pre-DE Prinz-init loss of ~315-350) — noticeably improved, but each restart converges to a *different* optimum (evidence of a multi-modal loss landscape), and the best restart's full-trace verification still fails to reproduce PY bursting. DE's feature loss (0.0073) reflects an essentially complete fit by comparison.

---

## Repository layout

```
.
├── data/
│   └── pyloric_observation.csv     # 4 s recording, dt=0.25 ms, 3 neurons
├── notebooks/
│   ├── main.ipynb                  # full project notebook (submit this)
│   ├── build_notebook.py           # regenerates main.ipynb from source
│   └── utils.py                    # simulation, burst detection, summary stats
├── jaxley-models/                  # git submodule: PyloricNetwork model
├── environment.yml                 # conda environment spec
├── requirements.txt                # pip-installable dependencies
└── setup.sh                        # one-shot environment + git setup
```

---

## Setup

### 1. Using the virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e jaxley-models/
```

**JAX compatibility note**: Jaxley 0.13 uses `jnp.clip(..., a_max=...)` which was removed in JAX ≥ 0.5. Patch the installed file:

```python
# .venv/lib/pythonX.Y/site-packages/jaxley/solver_gate.py  line ~9
# change:  jnp.clip(x, a_max=max_value)
# to:      jnp.clip(x, max=max_value)
```

### 2. Using conda

```bash
conda env create -f environment.yml
conda activate nds_env
pip install -e jaxley-models/
```

### 3. Register the Jupyter kernel

```bash
.venv/bin/python3 -m ipykernel install --user --name nds_env --display-name "Python 3 (nds_env)"
```

---

## Running the notebook

```bash
source .venv/bin/activate
cd notebooks/
jupyter notebook main.ipynb
```

**Expected runtime** (Apple M-series / modern CPU; measured on a real run):

| Section | Detail | Time |
|---------|--------|------|
| JIT warmup | First `jx.integrate` compiles XLA graph | ~3 min |
| Differential evolution | 50 iter × population 5 × ~2 s/sim (non-JIT'd, single-sample simulator) | ~2 h |
| *(skip DE: load from npz)* | `optimization_results.npz` already in repo | instant |
| Gradient descent (5b) | Adam, 3 restarts × 200 steps × ~2.5 s/step | ~25 min |
| SNPE simulations | JIT + `vmap`-batched simulator (batch 50), sampled until ≥1000 valid | ~14.5 min (10,850 sims, 1,001 valid, measured) |
| SNPE training + verification | Flow training (202 epochs) + 20 batched posterior-predictive samples | ~1-2 min |
| **Total (skipping DE)** | | **~41 min (measured)** |

The DE results are saved to `optimization_results.npz` and a hardcoded fallback is provided, so you can run the full analysis **without** re-running the 2-hour optimisation step (skip that one cell — the notebook is designed for this and warns you above it).

**Why so much faster than before**: the SNPE simulator used to rebuild the whole Jaxley network from scratch on every call and never JIT-compiled `jx.integrate` (~2 s/simulation). `utils.py` now exposes `build_batched_simulator()`, which builds the network once and wraps integration in `jax.jit(jax.vmap(...))` so a whole batch of parameter sets is simulated in a single compiled XLA call — measured at **~0.08-0.1 s/simulation (batch size 50), a ~20× speedup**. This is what makes a ≥10,000-simulation SNPE training budget (the rule-of-thumb minimum for a reliably-trained normalising flow) tractable in minutes instead of hours.

To rebuild `main.ipynb` from source after editing `build_notebook.py`:

```bash
python3 notebooks/build_notebook.py
```

---

## Summary statistics (9 features)

| # | Statistic | Why it matters |
|---|-----------|----------------|
| 0 | Burst period (ms) | primary pacemaker frequency |
| 1 | AB/PD duty cycle | how long AB/PD is active per cycle |
| 2 | LP duty cycle | second-phase timing |
| 3 | PY duty cycle | third-phase timing |
| 4 | LP phase offset (re AB/PD) | triphasic coordination |
| 5 | PY phase offset (re AB/PD) | triphasic coordination |
| 6 | AB/PD spikes / burst | firing intensity |
| 7 | LP spikes / burst | firing intensity |
| 8 | PY spikes / burst | firing intensity |

Phase offsets are weighted 3× in the DE loss because they are the most diagnostically sensitive features of the pyloric rhythm.

---

## LLM disclaimer

Claude (Anthropic) was used to assist with code structure, debug JAX/Jaxley API usage, refine summary statistics, and draft documentation. All modelling decisions and scientific interpretation are our own.
