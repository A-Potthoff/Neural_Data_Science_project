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

Two inference approaches are compared:

| Method | Description |
|--------|-------------|
| **Gradient descent** | Adam optimiser in log₁₀-conductance space, MSE loss on subsampled voltage, 3 random restarts |
| **SNPE** | Sequential Neural Posterior Estimation (sbi / SNPE-C), log-uniform prior, 9 burst-feature summary statistics |

Comparing GD point estimates and SNPE posterior samples directly addresses degeneracy (Prinz et al. 2004).

---

## Repository layout

```
.
├── data/
│   └── pyloric_observation.csv   # 4 s recording, dt=0.25 ms, 3 neurons
├── notebooks/
│   ├── main.ipynb                # full project notebook (submit this)
│   ├── build_notebook.py         # generates main.ipynb from cell definitions
│   └── utils.py                  # simulation, burst detection, summary stats
├── jaxley-models/                # git submodule: PyloricNetwork model
├── environment.yml               # conda environment spec
├── requirements.txt              # pip-installable dependencies
└── setup.sh                      # one-shot environment + git setup
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

### 3. Register the kernel

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

**Expected runtime** (Apple M-series / modern CPU):

| Section | Steps / Sims | Time |
|---------|-------------|------|
| JIT warmup | — | ~3 min |
| Gradient descent (3 restarts × 200 steps) | 600 gradient steps | ~40 min |
| SNPE simulation budget | 500 simulations | ~17 min |
| Verification | 20 posterior samples | ~1 min |
| **Total** | | **~60 min** |

To rebuild `main.ipynb` from source (after editing `build_notebook.py`):

```bash
python3 notebooks/build_notebook.py
```

---

## Key implementation details

### Summary statistics (9 scalars)

| # | Statistic |
|---|-----------|
| 0–1 | AB/PD burst period & duty cycle |
| 2–3 | LP burst period & duty cycle |
| 4–5 | PY burst period & duty cycle |
| 6 | LP phase offset relative to AB/PD |
| 7 | PY phase offset relative to AB/PD |
| 8 | Mean spikes per burst (all neurons) |

### Gradient descent

- Parameterisation: log₁₀(g) ∈ [−5, 1] (log-uniform prior range)
- Loss: MSE on voltage subsampled ×10 (sim at 0.025 ms, obs at 0.25 ms)
- Optimiser: `optax.chain(clip_by_global_norm(5.0), adam(0.02))`
- Memory: `checkpoint_lengths=[400, 200]` (gradient checkpointing)
- 3 restarts: Prinz-inspired init + 2 random inits in [−2.5, −0.5]

### SBI / SNPE

- Prior: `BoxUniform(low=[-5]*7, high=[1]*7)` in log₁₀-space
- ~10 % of random prior samples produce rhythmic bursting; the remainder are discarded before training
- Fallback: if fewer than 20 valid simulations are found, the notebook auto-supplements with 80 near-Prinz perturbations (±0.4 in log₁₀-space) to guarantee sufficient training data

---

## LLM disclaimer

Claude (Anthropic) was used to assist with code structure, debug JAX/Jaxley API usage, and refine summary statistics. All modelling decisions and scientific interpretation are our own.
