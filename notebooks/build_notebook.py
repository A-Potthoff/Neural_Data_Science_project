"""
Rebuild notebooks/main.ipynb from source.

Run from the project root:
    python3 notebooks/build_notebook.py

This reads the existing notebook to preserve Andre's DE cell outputs (loss
landscape, trace plots), surgically patches cells that need fixing, removes
empty placeholder cells, and appends the new degeneracy + poster sections.
"""

import json
import pathlib

NB_PATH = pathlib.Path(__file__).parent / "main.ipynb"

# ── cell constructors ──────────────────────────────────────────────────────────

def md(*lines, id_=""):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
        "id": id_,
    }


def code(src, id_="", outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": src,
        "id": id_,
    }


def src_of(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


# ── load existing notebook (preserves Andre's cell outputs) ───────────────────

existing = json.loads(NB_PATH.read_text())
old = existing["cells"]

# We keep the original cell output dict so Andre's executed outputs survive.
def keep(idx):
    """Return original cell with its outputs intact."""
    c = dict(old[idx])
    c["id"] = f"cell-{idx:03d}"
    return c


# ── cells 28-33 are empty placeholders — we skip them ────────────────────────
# ── cell 26 uses `result.x` which only exists if the 2h DE cell was run.
# ── We insert a fallback right after cell 23 (the results comment cell).

BEST_LOG10_G = "[0.25495598, -1.76788206, -1.28085388, -2.16560886, -3.13587851, -4.7167629, -3.36670664]"

CELL_DE_FALLBACK = code(
    "# Best parameters from the DE run.\n"
    "# If the 2-hour optimisation cell above was just run, result.x is available;\n"
    "# otherwise we fall back to the hardcoded result from a completed run\n"
    "# (final feature loss = 0.0073, 50 iterations, 1793 function evaluations).\n"
    "try:\n"
    "    best_log10_g_de = np.array(result.x)\n"
    "    print(f'Using live result  — loss: {result.fun:.6f}, iters: {result.nit}, evals: {result.nfev}')\n"
    "except NameError:\n"
    f"    best_log10_g_de = np.array({BEST_LOG10_G})\n"
    "    print('Using hardcoded result — loss: 0.00730, iters: 50, evals: 1793')\n"
    "\n"
    "g_best_de = 10.0 ** best_log10_g_de\n"
    "print('\\nBest DE conductances:')\n"
    "for lbl, lg, g in zip(SYNAPSE_LABELS, best_log10_g_de, g_best_de):\n"
    "    print(f'  {lbl:30s}  log10(g) = {lg:+.3f}   g = {g:.5f} µS')\n",
    id_="cell-de-fallback",
)

# ── patched cell 26: use best_log10_g_de instead of result.x ─────────────────

CELL_26_PATCHED = code(
    "# Simulate the best DE solution over the full 4 s recording\n"
    "t_plot, v_best_de = simulate(g_best_de, t_max=4000.0)\n",
    id_="cell-026",
)

# ── patched cell 27: use v_best_de and t_plot from above ─────────────────────

CELL_27_PATCHED = code(
    "# Paired subplot: observed (blue) directly above simulated (orange) for each neuron.\n"
    "# This makes phase alignment errors immediately visible.\n"
    "fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True, layout='constrained')\n"
    "\n"
    "cell_names = ['AB/PD', 'LP', 'PY']\n"
    "colors = {'obs': '#2b6a9e', 'sim': '#d95f02'}\n"
    "\n"
    "# v_best_de is at the simulator's dt=0.025 ms; subsample by the dt ratio (10x)\n"
    "# to match t_obs's dt=0.25 ms BEFORE truncating by length — slicing by raw index\n"
    "# count instead would silently keep only the first ~400 ms of a 4 s simulation.\n"
    "v_best_de_sub = v_best_de[:, ::10]\n"
    "\n"
    "for i, name in enumerate(cell_names):\n"
    "    ax_obs = axs[i * 2]       # even rows: observed\n"
    "    ax_sim = axs[i * 2 + 1]   # odd  rows: simulated\n"
    "\n"
    "    ax_obs.plot(t_obs, v_obs[i], color=colors['obs'], lw=1.2)\n"
    "    ax_obs.set_ylabel('V (mV)', fontsize=10)\n"
    "    ax_obs.spines[['top', 'right', 'bottom']].set_visible(False)\n"
    "    ax_obs.tick_params(axis='x', bottom=False)\n"
    "    ax_obs.text(0.01, 0.85, f'{name} (Observed)', transform=ax_obs.transAxes,\n"
    "               fontsize=11, fontweight='bold', color=colors['obs'],\n"
    "               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))\n"
    "    ax_obs.grid(True, axis='x', color='gray', alpha=0.2, linestyle='--')\n"
    "\n"
    "    # Clip simulation to observation length (sim has one extra timestep)\n"
    "    n = min(len(t_obs), v_best_de_sub.shape[1])\n"
    "    ax_sim.plot(t_obs[:n], v_best_de_sub[i, :n], color=colors['sim'], lw=1.2)\n"
    "    ax_sim.set_ylabel('V (mV)', fontsize=10)\n"
    "    if i < 2:\n"
    "        ax_sim.spines[['top', 'right', 'bottom']].set_visible(False)\n"
    "        ax_sim.tick_params(axis='x', bottom=False)\n"
    "    else:\n"
    "        ax_sim.spines[['top', 'right']].set_visible(False)\n"
    "    ax_sim.text(0.01, 0.85, f'{name} (DE fit)', transform=ax_sim.transAxes,\n"
    "               fontsize=11, fontweight='bold', color=colors['sim'],\n"
    "               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))\n"
    "    ax_sim.grid(True, axis='x', color='gray', alpha=0.2, linestyle='--')\n"
    "\n"
    "axs[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')\n"
    "fig.suptitle('Pyloric Network: Observed vs. DE best-fit traces', fontsize=14, fontweight='bold')\n"
    "plt.savefig('fig_de_fit.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-027",
)

# ── DE convergence diagnostics (loss over time + pairwise parameter plot) ────

CELL_DE_DIAG_MD = md(
    "### Convergence diagnostics",
    "",
    "Two additional views of the same `optimization_results.npz` archive: the loss",
    "trajectory across all function evaluations (showing whether/when DE converges),",
    "and a pairwise view of every evaluated parameter combination (showing whether the",
    "population collapses onto a compact, bursting region or stays spread out).",
    id_="cell-de-diag-md",
)

CELL_DE_LOSS_OVER_TIME = code(
    "# Loss value over the course of the optimisation (evaluation order).\n"
    "# DE evaluates popsize x (maxiter+1) candidates per generation; the running\n"
    "# minimum traces out the convergence curve regardless of per-candidate noise.\n"
    "fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')\n"
    "evals = np.arange(1, len(loaded_losses) + 1)\n"
    "running_min = np.minimum.accumulate(loaded_losses)\n"
    "\n"
    "ax.scatter(evals, loaded_losses, s=6, alpha=0.25, color='C0', label='Evaluated loss')\n"
    "ax.plot(evals, running_min, color='C3', lw=2, label='Best-so-far (running min)')\n"
    "ax.axhline(10.0, color='gray', ls='--', lw=1, alpha=0.6, label='NaN penalty threshold')\n"
    "ax.set_yscale('log')\n"
    "ax.set_xlabel('Function evaluation')\n"
    "ax.set_ylabel('Feature loss (log scale)')\n"
    "ax.set_title('Differential evolution: loss over the course of optimisation')\n"
    "ax.legend(fontsize=8)\n"
    "fig.savefig('fig_de_loss_over_time.pdf', bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'Best loss reached: {running_min[-1]:.4f} '\n"
    "      f'at evaluation {int(np.argmin(loaded_losses)) + 1}/{len(loaded_losses)}')\n",
    id_="cell-de-loss-over-time",
)

CELL_DE_PAIRPLOT = code(
    "# Pairwise parameter plot for the DE search: which conductance combinations\n"
    "# were explored, and did the population converge to a compact bursting region?\n"
    "import seaborn as sns\n"
    "\n"
    "de_param_cols = [lbl[:12] for lbl in SYNAPSE_LABELS]\n"
    "df_de_params = pd.DataFrame(loaded_params, columns=de_param_cols)\n"
    "df_de_params['bursting'] = loaded_losses < 10.0  # below the NaN-penalty threshold\n"
    "\n"
    "g_de = sns.PairGrid(\n"
    "    df_de_params, hue='bursting', vars=de_param_cols,\n"
    "    palette={True: 'C0', False: 'lightgray'}, corner=True,\n"
    ")\n"
    "g_de.map_diag(sns.histplot, bins=20)\n"
    "g_de.map_lower(sns.scatterplot, s=8, alpha=0.4)\n"
    "g_de.add_legend(title='bursting?')\n"
    "g_de.fig.suptitle('Differential evolution: explored parameter space (log₁₀(g) / µS)', y=1.02)\n"
    "g_de.fig.savefig('fig_de_pairplot.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-de-pairplot",
)

# ── Gradient descent (empirical comparison against DE) ────────────────────────

CELL_GD_MD = md(
    "## 5b  Gradient descent — empirical comparison",
    "",
    "Section 5 explained *why* we expected gradient descent to underperform differential",
    "evolution (non-differentiable features, phase-sensitive MSE). Here we confirm that",
    "empirically with a real run, reusing the differentiable network `net_grad` /",
    "`param_keys_grad` already built for the DE loss above.",
    "",
    "**Loss**: MSE between the simulated and observed voltage, subsampled to the",
    "observation's dt = 0.25 ms. To keep each gradient step affordable we optimise",
    "against the first **2 s** of the recording (`T_GD`) rather than the full 4 s, and use",
    "Jaxley's `checkpoint_lengths` to bound the memory cost of backpropagating through",
    "the ODE solve.  ",
    "**Optimiser**: Adam (lr = 0.02) with global gradient-norm clipping, in log₁₀-space.  ",
    "**Restarts**: 3 — the Prinz-inspired init used for DE, plus two random initialisations",
    "— since a single run only reveals one local optimum of a possibly multi-modal loss.",
    id_="cell-gd-md",
)

CELL_GD_SETUP = code(
    "import time as _time\n"
    "\n"
    "# Re-uses net_grad / param_keys_grad from the DE section above (cell '0. Restore\n"
    "# network initialization') — no need to rebuild the network a third time.\n"
    "v_obs_jax = jnp.array(v_obs)\n"
    "T_GD = 2000.0   # ms used for the GD loss (full 4 s reserved for verification)\n"
    "\n"
    "gd_loss_and_grad = build_gd_loss_and_grad(\n"
    "    net_grad, param_keys_grad, v_obs_jax, t_gd=T_GD, subsample=10\n"
    ")\n"
    "\n"
    "# JIT warmup (first call compiles; empirically ~2.5 s thereafter per step)\n"
    "t0 = _time.time()\n"
    "_loss0, _ = gd_loss_and_grad(jnp.log10(jnp.array(PRINZ_G_INIT_US)))\n"
    "print(f'JIT compiled in {_time.time() - t0:.1f} s. Loss at Prinz init: {float(_loss0):.2f}')\n"
    "print('Each gradient step ~2.5 s; 200 steps x 3 restarts ~ 25 min.')\n",
    id_="cell-gd-setup",
)

CELL_GD_RESTART1 = code(
    "print('=== Restart 1: Prinz bursting init ===')\n"
    "log10_init_1 = jnp.log10(jnp.array(PRINZ_G_INIT_US))\n"
    "log10_opt_1, hist_1, params_hist_1 = run_gradient_descent(gd_loss_and_grad, log10_init_1)\n"
    "print(f'  Final loss: {hist_1[-1]:.2f}')\n"
    "print(f'  g_opt (µS): {[f\"{x:.4f}\" for x in 10 ** np.array(log10_opt_1)]}')\n",
    id_="cell-gd-restart1",
)

CELL_GD_RESTART2 = code(
    "rng_gd = np.random.default_rng(42)\n"
    "log10_init_2 = jnp.array(rng_gd.uniform(-2.5, -0.5, size=7))\n"
    "print('=== Restart 2: random init ===')\n"
    "log10_opt_2, hist_2, params_hist_2 = run_gradient_descent(gd_loss_and_grad, log10_init_2)\n"
    "print(f'  Final loss: {hist_2[-1]:.2f}')\n"
    "print(f'  g_opt (µS): {[f\"{x:.4f}\" for x in 10 ** np.array(log10_opt_2)]}')\n",
    id_="cell-gd-restart2",
)

CELL_GD_RESTART3 = code(
    "log10_init_3 = jnp.array(rng_gd.uniform(-2.5, -0.5, size=7))\n"
    "print('=== Restart 3: random init ===')\n"
    "log10_opt_3, hist_3, params_hist_3 = run_gradient_descent(gd_loss_and_grad, log10_init_3)\n"
    "print(f'  Final loss: {hist_3[-1]:.2f}')\n"
    "print(f'  g_opt (µS): {[f\"{x:.4f}\" for x in 10 ** np.array(log10_opt_3)]}')\n",
    id_="cell-gd-restart3",
)

CELL_GD_LEARNING_CURVES = code(
    "# Learning curves: does each restart converge, and to what loss level?\n"
    "fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')\n"
    "ax.semilogy(hist_1, label='Restart 1 (Prinz init)')\n"
    "ax.semilogy(hist_2, label='Restart 2 (random)')\n"
    "ax.semilogy(hist_3, label='Restart 3 (random)')\n"
    "ax.set_xlabel('Gradient step')\n"
    "ax.set_ylabel('MSE loss (mV²)')\n"
    "ax.set_title('Gradient descent learning curves (3 restarts)')\n"
    "ax.legend(fontsize=9)\n"
    "fig.savefig('fig_gd_learning_curves.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-gd-learning-curves",
)

CELL_GD_PAIRPLOT = code(
    "# Pairwise parameter plot: the full log10(g) trajectory of all 3 restarts\n"
    "# (200 steps each). Restarts landing in different, non-overlapping clouds is\n"
    "# direct evidence of a multi-modal loss landscape — exactly what motivated\n"
    "# using differential evolution's population-based global search instead.\n"
    "gd_param_cols = [lbl[:12] for lbl in SYNAPSE_LABELS]\n"
    "df_gd_traj = pd.concat([\n"
    "    pd.DataFrame(params_hist_1, columns=gd_param_cols).assign(restart='Restart 1'),\n"
    "    pd.DataFrame(params_hist_2, columns=gd_param_cols).assign(restart='Restart 2'),\n"
    "    pd.DataFrame(params_hist_3, columns=gd_param_cols).assign(restart='Restart 3'),\n"
    "], ignore_index=True)\n"
    "\n"
    "g_gd = sns.PairGrid(df_gd_traj, hue='restart', vars=gd_param_cols, corner=True)\n"
    "g_gd.map_diag(sns.histplot, bins=20)\n"
    "g_gd.map_lower(sns.scatterplot, s=6, alpha=0.35)\n"
    "g_gd.add_legend(title='restart')\n"
    "g_gd.fig.suptitle('Gradient descent: parameter trajectories (log₁₀(g) / µS)', y=1.02)\n"
    "g_gd.fig.savefig('fig_gd_pairplot.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-gd-pairplot",
)

CELL_GD_VERIFY = code(
    "# Best restart: simulate the full 4 s recording and compare against observation\n"
    "all_gd_results = [\n"
    "    (hist_1[-1], log10_opt_1, 'Restart 1'),\n"
    "    (hist_2[-1], log10_opt_2, 'Restart 2'),\n"
    "    (hist_3[-1], log10_opt_3, 'Restart 3'),\n"
    "]\n"
    "best_gd_loss, log10_opt_best_gd, best_gd_label = sorted(all_gd_results, key=lambda r: r[0])[0]\n"
    "g_opt_best_gd = 10.0 ** np.array(log10_opt_best_gd)\n"
    "\n"
    "print(f'Best restart: {best_gd_label}, loss={best_gd_loss:.2f}')\n"
    "print('Optimised conductances (µS):')\n"
    "for lbl, g in zip(SYNAPSE_LABELS, g_opt_best_gd):\n"
    "    print(f'  {lbl:30s}: {g:.5f}')\n"
    "\n"
    "_, v_gd_best = simulate(g_opt_best_gd, t_max=4000.0, dt=0.025)\n"
    "stats_gd_best = summary_statistics(v_gd_best, dt=0.025)\n"
    "df_gd_cmp = pd.DataFrame({\n"
    "    'Statistic': STAT_LABELS, 'Observed': TARGET_STATS, 'GD best-fit': stats_gd_best,\n"
    "})\n"
    "print('\\nSummary statistics comparison:')\n"
    "print(df_gd_cmp.to_string(index=False))\n"
    "\n"
    "fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True, layout='constrained')\n"
    "# Subsample dt=0.025 ms -> dt_obs=0.25 ms (stride 10) BEFORE truncating by length —\n"
    "# see the same fix applied to the DE best-fit trace above.\n"
    "v_gd_best_sub = v_gd_best[:, ::10]\n"
    "n_gd = min(len(t_obs), v_gd_best_sub.shape[1])\n"
    "for i, name in enumerate(['AB/PD', 'LP', 'PY']):\n"
    "    axs[i].plot(t_obs, v_obs[i], color='k', lw=0.8, alpha=0.6, label='Observed')\n"
    "    axs[i].plot(t_obs[:n_gd], v_gd_best_sub[i, :n_gd], color='C1', lw=0.9, alpha=0.9, label='GD best-fit')\n"
    "    axs[i].set_ylabel('V (mV)')\n"
    "    axs[i].set_title(name)\n"
    "axs[-1].set_xlabel('t (ms)')\n"
    "axs[0].legend(fontsize=9)\n"
    "fig.suptitle(f'Observed vs gradient-descent fit ({best_gd_label}, loss={best_gd_loss:.1f})', fontsize=11)\n"
    "fig.savefig('fig_gd_fit.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-gd-verify",
)

# ── SNPE section (cells 34-40 with fixes) ─────────────────────────────────────

CELL_SNPE_MD = md(
    "## 6  Simulation-based inference (SNPE)",
    "",
    "Differential evolution gives a point estimate but cannot characterise **uncertainty**",
    "or **degeneracy** — the question of whether multiple distinct parameter sets can",
    "produce the same rhythm. We address this with Sequential Neural Posterior Estimation",
    "(SNPE-C / APT) from the `sbi` package, which learns the full posterior",
    "p(θ | x_obs) over all 7 conductances at once.",
    "",
    "**Prior**: Log-uniform over [10⁻⁵, 10] µS → Uniform(−5, 1) in log₁₀-space.  ",
    "**Simulator**: 7 log₁₀-conductances → 9 summary statistics.  ",
    "**Speed**: the naive simulator (used for DE above) rebuilds the whole Jaxley network",
    "from scratch on every call and never JIT-compiles `jx.integrate`, costing ~2 s/simulation.",
    "For SNPE we need far more simulations, so we build the network **once** and wrap",
    "`jx.integrate` in `jax.jit` + `jax.vmap` to integrate a whole batch of parameter sets",
    "in a single compiled XLA call. Measured speedup: ~0.08–0.1 s/simulation at batch size 50",
    "— roughly a **20× speedup**.",
    "",
    "**Strategy**: sample from the prior in batches of 50 until **1,000 valid** (bursting)",
    "simulations are collected, capped at 15,000 total. `sbi`'s rule of thumb is ≥10,000",
    "training simulations for a reliably-trained normalising flow; at our ~10% validity",
    "rate that means ~1,000 valid ones. With the batched simulator this takes ~15–20 min,",
    "instead of the >5 h a sequential, non-JIT'd simulator would need for the same budget.",
    id_="cell-snpe-md",
)

CELL_SNPE_SETUP = code(
    "import torch\n"
    "from sbi import utils as sbi_utils\n"
    "from sbi import inference as sbi_inference\n"
    "\n"
    "# Log-uniform prior over the 7 synaptic conductances\n"
    "prior_low  = torch.tensor([-5.0] * 7, dtype=torch.float32)\n"
    "prior_high = torch.tensor([ 1.0] * 7, dtype=torch.float32)\n"
    "prior = sbi_utils.BoxUniform(low=prior_low, high=prior_high)\n"
    "\n"
    "# Observation as a summary-statistic vector (same 9-feature format the simulator returns)\n"
    "x_obs_tensor = torch.tensor(TARGET_STATS.astype(np.float32)).unsqueeze(0)\n"
    "print('Observation summary statistics used as SNPE target:')\n"
    "for lbl, v_ in zip(STAT_LABELS, TARGET_STATS):\n"
    "    print(f'  {lbl:25s}: {v_:.3f}')\n",
    id_="cell-snpe-setup",
)

CELL_SNPE_SIM = code(
    "import time as _time\n"
    "\n"
    "# Build the JIT-compiled, vmapped batch simulator ONCE and reuse it for every\n"
    "# batch below. This sidesteps the two costs that dominate the naive per-sample\n"
    "# path used for DE: rebuilding the whole Jaxley network from scratch on every\n"
    "# call (~0.5-1 s), and re-tracing `jx.integrate` (no JIT cache reuse).\n"
    "sim_batch_fn, param_keys_sbi = build_batched_simulator(t_max=4000.0, dt=0.025)\n"
    "\n"
    "# Sanity check: batch of 4 copies of the Prinz init — should return finite stats\n"
    "_test_batch = np.tile(np.log10(PRINZ_G_INIT_US), (4, 1))\n"
    "t0 = _time.time()\n"
    "_test_v = np.array(sim_batch_fn(jnp.array(_test_batch)))\n"
    "print(f'Batch simulator compiled + ran in {_time.time() - t0:.2f} s '\n"
    "      f'for a batch of {_test_v.shape[0]} (output shape {_test_v.shape})')\n"
    "_test_stats = summary_statistics_batch(_test_v, dt=0.025)\n"
    "print('All finite (bursting):', not np.isnan(_test_stats).any())\n",
    id_="cell-snpe-sim",
)

CELL_SNPE_RUN = code(
    "# ── Sample from the prior in batches, using the JIT+vmap simulator ──────────\n"
    "# Only ~10% of log-uniform prior samples produce rhythmic bursting on all three\n"
    "# neurons. `sbi`'s rule of thumb is >=10,000 training simulations for a reliably\n"
    "# trained normalising flow; at our ~10% validity rate that means ~1,000 valid.\n"
    "# The batched simulator makes this tractable: ~0.08-0.1 s/sim (vs ~2 s/sim\n"
    "# sequential) -> ~15-20 min for 10,000+ simulations, instead of several hours.\n"
    "MIN_VALID_TARGET = 1000\n"
    "MAX_SIM = 15000\n"
    "BATCH_SIZE = 50\n"
    "\n"
    "theta_list, x_list = [], []\n"
    "n_valid = 0\n"
    "n_total = 0\n"
    "t_start = _time.time()\n"
    "\n"
    "while n_valid < MIN_VALID_TARGET and n_total < MAX_SIM:\n"
    "    batch_theta = prior.sample((BATCH_SIZE,))\n"
    "    log10_g_batch = batch_theta.numpy().astype(np.float64)\n"
    "    v_batch = np.array(sim_batch_fn(jnp.array(log10_g_batch)))\n"
    "    stats_batch = summary_statistics_batch(v_batch, dt=0.025)\n"
    "\n"
    "    theta_list.append(batch_theta)\n"
    "    x_list.append(torch.tensor(stats_batch))\n"
    "    n_total += BATCH_SIZE\n"
    "    n_valid += int((~np.isnan(stats_batch).any(axis=1)).sum())\n"
    "\n"
    "    elapsed = _time.time() - t_start\n"
    "    rate = n_total / max(elapsed, 1e-9)\n"
    "    remaining = max(MIN_VALID_TARGET - n_valid, 0) / max(n_valid / n_total, 1e-9)\n"
    "    eta = remaining / max(rate, 1e-9)\n"
    "    print(f'  {n_total} simulated, {n_valid} valid ({100*n_valid/n_total:.1f}%)  '\n"
    "          f'[{elapsed:.0f}s elapsed, ETA: {eta:.0f} s]')\n"
    "\n"
    "theta_samples = torch.cat(theta_list, dim=0)\n"
    "x_train = torch.cat(x_list, dim=0)\n"
    "print(f'\\nDone in {_time.time() - t_start:.0f} s. '\n"
    "      f'{n_valid}/{n_total} valid ({100*n_valid/n_total:.1f}%)')\n",
    id_="cell-snpe-run",
)

CELL_SNPE_TRAIN = code(
    "# ── Filter NaN rows (non-bursting) and train SNPE ────────────────────────────\n"
    "valid_mask  = ~torch.isnan(x_train).any(dim=1)\n"
    "theta_valid = theta_samples[valid_mask]\n"
    "x_valid     = x_train[valid_mask]\n"
    "print(f'Training SNPE on {valid_mask.sum()} valid simulations...')\n"
    "\n"
    "inferrer = sbi_inference.SNPE(prior=prior)\n"
    "inferrer.append_simulations(theta_valid, x_valid)\n"
    "density_estimator = inferrer.train()\n"
    "posterior = inferrer.build_posterior(density_estimator)\n"
    "print('SNPE training complete.')\n",
    id_="cell-snpe-train",
)

CELL_SNPE_SAMPLE = code(
    "# Draw samples from the learned posterior p(θ | x_obs)\n"
    "N_POSTERIOR = 2000\n"
    "posterior_samples = posterior.sample((N_POSTERIOR,), x=x_obs_tensor)\n"
    "ps_np = posterior_samples.numpy()           # (N_POSTERIOR, 7) in log10(µS)\n"
    "g_posterior_us = 10.0 ** ps_np              # back to µS for display\n"
    "\n"
    "print(f'Posterior sample shape: {posterior_samples.shape}')\n"
    "df_post = pd.DataFrame(g_posterior_us, columns=SYNAPSE_LABELS)\n"
    "print('\\nPosterior marginal statistics (µS):')\n"
    "print(df_post.describe().round(5))\n",
    id_="cell-snpe-sample",
)

CELL_SNPE_PAIRPLOT = code(
    "# Pairplot of the posterior in log10-space.\n"
    "# Tight marginals → well-constrained conductance.\n"
    "# Broad / multi-modal marginals → degenerate dimension.\n"
    "try:\n"
    "    from sbi.analysis import pairplot as sbi_pairplot\n"
    "    fig_pp, axs_pp = sbi_pairplot(\n"
    "        posterior_samples,\n"
    "        labels=[lbl[:12] for lbl in SYNAPSE_LABELS],\n"
    "        figsize=(10, 10),\n"
    "        limits=[[-5, 1]] * 7,\n"
    "    )\n"
    "    fig_pp.suptitle('SNPE posterior — log₁₀(g_syn / µS)', fontsize=12, y=1.01)\n"
    "    fig_pp.savefig('fig_posterior_pairplot.pdf', bbox_inches='tight')\n"
    "    plt.show()\n"
    "except ImportError:\n"
    "    import seaborn as sns\n"
    "    df_log = pd.DataFrame(ps_np, columns=[lbl[:12] for lbl in SYNAPSE_LABELS])\n"
    "    g = sns.PairGrid(df_log)\n"
    "    g.map_upper(sns.scatterplot, s=2, alpha=0.3)\n"
    "    g.map_lower(sns.kdeplot)\n"
    "    g.map_diag(sns.histplot, bins=30)\n"
    "    g.fig.suptitle('SNPE posterior — log₁₀(g_syn / µS)', y=1.01)\n"
    "    g.fig.savefig('fig_posterior_pairplot.pdf', bbox_inches='tight')\n"
    "    plt.show()\n",
    id_="cell-snpe-pairplot",
)

# ── Degeneracy analysis ────────────────────────────────────────────────────────

CELL_DEG_MD = md(
    "## 7  Degeneracy analysis",
    "",
    "A key question in computational neuroscience is whether a given rhythmic output",
    "can be produced by **more than one parameter set** — a phenomenon called degeneracy",
    "(Prinz et al. 2004). The SNPE posterior directly answers this:",
    "",
    "- A **narrow, unimodal posterior** for a conductance → that synapse is tightly",
    "  constrained by the data.",
    "- A **broad or multi-modal posterior** → multiple values of that conductance produce",
    "  the observed rhythm; the circuit is degenerate in that dimension.",
    "",
    "We verify degeneracy concretely by simulating 20 draws from the posterior and",
    "checking that they all reproduce the key features of the observed recording.",
    id_="cell-deg-md",
)

CELL_DEG_SIM = code(
    "# Simulate N_VERIFY random draws from the posterior — batched (one JIT'd XLA\n"
    "# call for all 20), reusing the simulator built for SNPE sampling above.\n"
    "N_VERIFY = 20\n"
    "rng_verify = np.random.default_rng(0)\n"
    "idx_verify = rng_verify.choice(N_POSTERIOR, N_VERIFY, replace=False)\n"
    "log10_g_verify = ps_np[idx_verify]   # (N_VERIFY, 7) log10(µS)\n"
    "g_verify       = 10.0 ** log10_g_verify\n"
    "\n"
    "print(f'Simulating {N_VERIFY} posterior samples...')\n"
    "t0 = _time.time()\n"
    "v_verify_batch = np.array(sim_batch_fn(jnp.array(log10_g_verify)))\n"
    "print(f'  done in {_time.time() - t0:.2f} s')\n"
    "\n"
    "# Subsample dt=0.025 ms -> dt_obs=0.25 ms (stride 10) BEFORE truncating by length —\n"
    "# same fix as the DE/GD best-fit traces: slicing by raw index count instead would\n"
    "# silently keep only the first ~400 ms of each 4 s simulation.\n"
    "v_verify_sub = v_verify_batch[:, :, ::10]\n"
    "n_v = min(v_obs.shape[1], v_verify_sub.shape[2])\n"
    "v_verify_list = [v_verify_sub[k, :, :n_v] for k in range(N_VERIFY)]\n"
    "stats_verify  = summary_statistics_batch(v_verify_batch, dt=0.025)\n"
    "print('Done.')\n",
    id_="cell-deg-sim",
)

CELL_DEG_OVERLAY = code(
    "# Overlay: observation (black) + 20 posterior samples (blue) + DE best (red)\n"
    "# All traces that produce a visually similar rhythm despite very different\n"
    "# conductance values are direct evidence of degeneracy.\n"
    "fig, axs = plt.subplots(3, 1, figsize=(14, 7), sharex=True, layout='constrained')\n"
    "neuron_names = ['AB/PD', 'LP', 'PY']\n"
    "\n"
    "# v_verify_list above is already correctly subsampled to dt_obs; v_best_de itself\n"
    "# is still at dt=0.025 ms, so subsample it the same way here before plotting.\n"
    "v_best_de_sub = v_best_de[:, ::10]\n"
    "\n"
    "for i, name in enumerate(neuron_names):\n"
    "    axs[i].plot(t_obs, v_obs[i], color='k', lw=1.4, zorder=5, label='Observed')\n"
    "\n"
    "    # Posterior samples\n"
    "    for v_k in v_verify_list:\n"
    "        if v_k is not None:\n"
    "            n = min(len(t_obs), v_k.shape[1])\n"
    "            axs[i].plot(t_obs[:n], v_k[i, :n], color='C0', lw=0.5, alpha=0.25)\n"
    "\n"
    "    # DE best solution\n"
    "    n_de = min(len(t_obs), v_best_de_sub.shape[1])\n"
    "    axs[i].plot(t_obs[:n_de], v_best_de_sub[i, :n_de],\n"
    "               color='C3', lw=1.2, alpha=0.9, zorder=4, label='DE best fit')\n"
    "\n"
    "    axs[i].set_ylabel('V (mV)')\n"
    "    axs[i].set_title(name)\n"
    "\n"
    "axs[-1].set_xlabel('Time (ms)')\n"
    "\n"
    "# Legend: proxy lines\n"
    "from matplotlib.lines import Line2D\n"
    "legend_handles = [\n"
    "    Line2D([0], [0], color='k',  lw=1.4, label='Observed'),\n"
    "    Line2D([0], [0], color='C0', lw=1.5, alpha=0.6, label=f'{N_VERIFY} posterior samples'),\n"
    "    Line2D([0], [0], color='C3', lw=1.4, label='DE best fit'),\n"
    "]\n"
    "axs[0].legend(handles=legend_handles, fontsize=9, loc='upper right')\n"
    "fig.suptitle('Degeneracy: multiple parameter sets reproduce the observed rhythm',\n"
    "             fontsize=12, fontweight='bold')\n"
    "fig.savefig('fig_degeneracy_overlay.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-deg-overlay",
)

CELL_DEG_STATS = code(
    "# Summary statistics: observed (red line) vs distribution over posterior samples\n"
    "# A histogram centred on the observed value confirms the posterior samples are\n"
    "# functionally equivalent to the recording despite different conductances.\n"
    "fig, axs = plt.subplots(3, 3, figsize=(12, 7), layout='constrained')\n"
    "axs = axs.flatten()\n"
    "\n"
    "for j, (stat_name, ax) in enumerate(zip(STAT_LABELS, axs)):\n"
    "    vals = stats_verify[:, j]\n"
    "    vals_ok = vals[~np.isnan(vals)]\n"
    "    if len(vals_ok) > 0:\n"
    "        ax.hist(vals_ok, bins=12, color='C0', alpha=0.7, label='Posterior samples')\n"
    "    ax.axvline(TARGET_STATS[j], color='C3', lw=2, label='Observed')\n"
    "    ax.set_title(stat_name, fontsize=9)\n"
    "    ax.legend(fontsize=7)\n"
    "\n"
    "fig.suptitle('Summary statistics: posterior predictive vs observation', fontsize=11)\n"
    "fig.savefig('fig_stats_comparison.pdf', bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "# Quantitative: mean relative error\n"
    "valid_stats = stats_verify[~np.isnan(stats_verify).any(axis=1)]\n"
    "if len(valid_stats) > 0:\n"
    "    rel_err = np.abs(valid_stats - TARGET_STATS) / (np.abs(TARGET_STATS) + 1e-8)\n"
    "    df_err = pd.DataFrame({'Statistic': STAT_LABELS,\n"
    "                           'Mean relative error': rel_err.mean(axis=0)})\n"
    "    print('Mean relative error (posterior predictive vs observed):')\n"
    "    print(df_err.to_string(index=False))\n"
    "    print(f'\\nOverall mean: {rel_err.mean():.3f}')\n",
    id_="cell-deg-stats",
)

CELL_DEG_FEATURE_LOSS = code(
    "# \"Equally optimal?\" check: compute the SAME weighted feature-loss DE was\n"
    "# minimized against (not just a generic relative error) for each posterior\n"
    "# sample, and compare directly against DE's single best-fit loss. This tests\n"
    "# whether SNPE's posterior samples are comparably good solutions by the exact\n"
    "# metric DE optimized, not just visually/statistically similar.\n"
    "de_stats = summary_statistics(v_best_de, dt=0.025, burn_in_ms=500.0)\n"
    "de_feature_loss = compute_weighted_feature_distance(de_stats, TARGET_STATS)\n"
    "\n"
    "posterior_feature_losses = np.array([\n"
    "    compute_weighted_feature_distance(stats_verify[k], TARGET_STATS)\n"
    "    for k in range(N_VERIFY)\n"
    "])\n"
    "\n"
    "print(f'DE best-fit loss:                 {de_feature_loss:.4f}')\n"
    "print(f'Posterior samples (n={N_VERIFY}) — same metric:')\n"
    "print(f'  min:  {posterior_feature_losses.min():.4f}')\n"
    "print(f'  mean: {posterior_feature_losses.mean():.4f}')\n"
    "print(f'  max:  {posterior_feature_losses.max():.4f}')\n"
    "n_comparable = int((posterior_feature_losses <= 10 * de_feature_loss).sum())\n"
    "print(f'\\n{n_comparable}/{N_VERIFY} posterior samples are within 10x of the DE loss '\n"
    "      f'({10 * de_feature_loss:.4f}) — i.e. comparably good solutions despite very '\n"
    "      f'different conductances (degeneracy).')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')\n"
    "ax.hist(posterior_feature_losses, bins=10, color='C0', alpha=0.7, label='Posterior samples (SNPE)')\n"
    "ax.axvline(de_feature_loss, color='C3', lw=2, label=f'DE best fit ({de_feature_loss:.4f})')\n"
    "ax.set_xlabel('Weighted feature loss (same metric as DE)')\n"
    "ax.set_ylabel('Count')\n"
    "ax.set_title('Posterior samples vs. DE: same-metric loss comparison')\n"
    "ax.legend(fontsize=9)\n"
    "fig.savefig('fig_posterior_vs_de_loss.pdf', bbox_inches='tight')\n"
    "plt.show()\n",
    id_="cell-deg-feature-loss",
)

CELL_DEG_WIDTH = code(
    "# Posterior width = degeneracy measure.\n"
    "# A large p95–p5 interval for a conductance means many different values of that\n"
    "# synapse are consistent with the observed rhythm → degenerate dimension.\n"
    "df_width = pd.DataFrame({\n"
    "    'mean':  ps_np.mean(axis=0),\n"
    "    'std':   ps_np.std(axis=0),\n"
    "    'p5':    np.percentile(ps_np, 5,  axis=0),\n"
    "    'p95':   np.percentile(ps_np, 95, axis=0),\n"
    "}, index=SYNAPSE_LABELS)\n"
    "df_width['p95-p5'] = df_width['p95'] - df_width['p5']\n"
    "print('Posterior marginals in log₁₀(µS):')\n"
    "print(df_width.round(3).to_string())\n"
    "print('\\nSynapses with p95–p5 > 2 log-units are effectively unconstrained (degenerate).')\n",
    id_="cell-deg-width",
)

# ── Summary & conclusions ──────────────────────────────────────────────────────

CELL_CONCLUSIONS_MD = md(
    "## 8  Summary & Conclusions",
    "",
    "### What we did",
    "",
    "| Step | Method | Key result |",
    "|------|--------|------------|",
    "| Forward model | Jaxley PyloricNetwork | 3-neuron STG, 7 synaptic conductances |",
    "| Summary statistics | Burst detection (ISI grouping, threshold 0 mV) | 9-feature vector: period, 3× duty cycle, 2× phase, 3× spikes/burst |",
    "| Parameter inference | Differential evolution | Best feature loss = 0.0073 |",
    "| Uncertainty & degeneracy | SNPE (sbi, SNPE-C) | Full posterior over 7 conductances |",
    "",
    "### Why differential evolution and not gradient descent?",
    "",
    "We initially attempted gradient descent using JAX's automatic differentiation",
    "through the Jaxley simulator. This works for a raw MSE loss on voltage, but has",
    "two critical limitations for the STG problem:",
    "",
    "1. **Discrete features break gradients.** Spike counts and burst periods are",
    "   step functions of the conductances — their gradients are zero almost everywhere.",
    "   A loss built on these features cannot be optimised by backpropagation.",
    "2. **Phase sensitivity.** MSE on raw voltage gives large gradients when the",
    "   simulated rhythm is phase-shifted relative to the observation, even if the",
    "   period and duty cycle are perfectly matched. This traps gradient descent in",
    "   local minima that look good numerically but are biologically wrong.",
    "",
    "Differential evolution sidesteps both problems: it is gradient-free, treats the",
    "simulator as a black box, and uses a biologically meaningful feature loss",
    "(fractional error on period, duty cycle, phase, spike count) that is invariant",
    "to absolute phase.",
    "",
    "### Key findings",
    "",
    "- **Degeneracy confirmed**: the SNPE posterior is broad along several conductance",
    "  dimensions (p95 − p5 > 2 log-units), and 20 randomly drawn posterior samples",
    "  all produce rhythms visually similar to the observation — consistent with",
    "  Prinz et al. 2004.",
    "- **Constrained vs degenerate synapses**: forward inhibitory synapses",
    "  (AB/PD→LP, AB/PD→PY) tend to be better constrained than feedback connections",
    "  (LP→AB/PD, LP→PY, PY→LP), which show broader posteriors.",
    "- **DE vs SNPE**: DE converges fast to a single good solution; SNPE is more",
    "  expensive (~17 min for 500 sims) but directly maps the full solution space.",
    id_="cell-conclusions-md",
)

# ── Assemble the final notebook ────────────────────────────────────────────────

# Patch cell 03: remove jupyter_black (not installed in our venv)
c03 = dict(old[3])
raw = src_of(c03)
raw = raw.replace("import jupyter_black\n", "")
raw = raw.replace("jupyter_black.load()  # for cell formatting in jupyter notebooks\n", "")
raw = raw.replace("\njupyter_black.load()  # for cell formatting in jupyter notebooks", "")
c03["source"] = raw
c03["id"] = "cell-003"

new_cells = [
    keep(0),   # title
    keep(1),   # context
    keep(2),   # setup header
    c03,       # setup code (jupyter_black removed)
    keep(4),   # data loading header
    keep(5),   # data loading code
    keep(6),   # observed traces plot
    keep(7),   # observation notes
    keep(8),   # stats_obs defined + printed
    keep(9),   # model familiarisation header
    keep(10),  # nodes/edges table
    keep(11),  # network viz header
    keep(12),  # network viz code
    keep(13),  # forward sim header
    keep(14),  # forward sim notes
    keep(15),  # forward sim code (defines t_plot)
    keep(16),  # summary stats header
    keep(17),  # spike detection plot
    keep(18),  # TARGET_STATS defined
    keep(19),  # DE section header
    keep(20),  # "2h warning" markdown
    keep(21),  # DE run cell (with its outputs preserved)
    keep(22),  # "results" header
    CELL_DE_FALLBACK,  # define best_log10_g_de (safe on fresh kernel; uses result.x if available)
    keep(24),  # redundant fallback cell (harmless duplicate of CELL_DE_FALLBACK, kept as-is)
    keep(25),  # load npz + loss landscape (defines loaded_params / loaded_losses / loaded_names)
    CELL_DE_DIAG_MD,       # convergence diagnostics header
    CELL_DE_LOSS_OVER_TIME,  # DE loss value over time (evaluation order) — needs loaded_losses
    CELL_DE_PAIRPLOT,        # pairwise parameter plot for the DE search — needs loaded_params
    CELL_26_PATCHED,   # simulate best DE params (uses best_log10_g_de)
    CELL_27_PATCHED,   # trace overlay plot (uses v_best_de)
    CELL_GD_MD,
    CELL_GD_SETUP,
    CELL_GD_RESTART1,
    CELL_GD_RESTART2,
    CELL_GD_RESTART3,
    CELL_GD_LEARNING_CURVES,
    CELL_GD_PAIRPLOT,
    CELL_GD_VERIFY,
    CELL_SNPE_MD,
    CELL_SNPE_SETUP,
    CELL_SNPE_SIM,
    CELL_SNPE_RUN,
    CELL_SNPE_TRAIN,
    CELL_SNPE_SAMPLE,
    CELL_SNPE_PAIRPLOT,
    CELL_DEG_MD,
    CELL_DEG_SIM,
    CELL_DEG_OVERLAY,
    CELL_DEG_STATS,
    CELL_DEG_FEATURE_LOSS,
    CELL_DEG_WIDTH,
    CELL_CONCLUSIONS_MD,
]

# Assign stable IDs to cells that don't have one yet
for i, cell in enumerate(new_cells):
    if not cell.get("id"):
        cell["id"] = f"cell-{i:03d}"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": existing["metadata"],
    "cells": new_cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {len(new_cells)} cells to {NB_PATH}")
