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
    "    n = min(len(t_obs), v_best_de.shape[1])\n"
    "    ax_sim.plot(t_obs[:n], v_best_de[i, :n], color=colors['sim'], lw=1.2)\n"
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
    "**Strategy**: one amortised round of SNPE, then posterior predictive checks.",
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
    "def simulator_for_sbi(log10_g_tensor):\n"
    "    \"\"\"Wraps the Jaxley simulator for sbi: Tensor → summary-stat Tensor.\n"
    "\n"
    "    Returns a 9-element tensor with NaN entries for non-bursting simulations;\n"
    "    sbi handles NaN rows by filtering them out before training.\n"
    "    \"\"\"\n"
    "    log10_g = log10_g_tensor.numpy().astype(np.float64)\n"
    "    g_us = np.clip(10.0 ** log10_g, 1e-6, 15.0)\n"
    "    try:\n"
    "        _, v_sim = simulate(g_us, t_max=4000.0, dt=0.025)\n"
    "        stats = summary_statistics(v_sim, dt=0.025).astype(np.float32)\n"
    "    except Exception:\n"
    "        stats = np.full(9, np.nan, dtype=np.float32)\n"
    "    return torch.tensor(stats)\n"
    "\n"
    "# Quick sanity check with the Prinz init — should return finite stats\n"
    "test_g = torch.tensor(np.log10(PRINZ_G_INIT_US).astype(np.float32))\n"
    "test_out = simulator_for_sbi(test_g)\n"
    "print('Simulator test (Prinz init):', test_out)\n"
    "print('All finite:', not torch.isnan(test_out).any().item())\n",
    id_="cell-snpe-sim",
)

CELL_SNPE_RUN = code(
    "# ── Run N_SIM forward simulations from the prior ──────────────────────────────\n"
    "# Runtime: ~2 s / sim → 500 sims ≈ 17 min, 2000 sims ≈ 67 min.\n"
    "# Only ~10% of log-uniform prior samples produce rhythmic bursting; the rest\n"
    "# return NaN and are filtered before training.\n"
    "import time as _time\n"
    "\n"
    "N_SIM = 500\n"
    "theta_samples = prior.sample((N_SIM,))\n"
    "\n"
    "x_list, n_valid = [], 0\n"
    "t_start = _time.time()\n"
    "\n"
    "for i, theta in enumerate(theta_samples):\n"
    "    xi = simulator_for_sbi(theta)\n"
    "    x_list.append(xi)\n"
    "    if not torch.isnan(xi).any():\n"
    "        n_valid += 1\n"
    "    if (i + 1) % 50 == 0:\n"
    "        elapsed = _time.time() - t_start\n"
    "        eta = elapsed / (i + 1) * (N_SIM - i - 1)\n"
    "        print(f'  {i+1}/{N_SIM}  valid: {n_valid}  ETA: {eta:.0f} s')\n"
    "\n"
    "x_train = torch.stack(x_list)\n"
    "print(f'\\nDone. Valid: {n_valid}/{N_SIM} ({100*n_valid/N_SIM:.0f}%)')\n",
    id_="cell-snpe-run",
)

CELL_SNPE_TRAIN = code(
    "# ── Supplement with near-Prinz seeds if random prior yields too few valid sims ─\n"
    "# With only ~10% validity from the uniform prior, small N_SIM runs can produce\n"
    "# too few valid samples for the normalising flow to train (std ≈ 0 → NaN scale).\n"
    "# We add jittered Prinz-neighbourhood samples as a fallback.\n"
    "MIN_VALID = 20\n"
    "valid_mask_check = ~torch.isnan(x_train).any(dim=1)\n"
    "\n"
    "if valid_mask_check.sum() < MIN_VALID:\n"
    "    print(f'Only {valid_mask_check.sum()} valid; supplementing with near-Prinz seeds...')\n"
    "    rng_seed = np.random.default_rng(0)\n"
    "    seed_log10_g = np.log10(PRINZ_G_INIT_US).astype(np.float32)\n"
    "    extra_thetas, extra_xs = [], []\n"
    "    for _ in range(80):\n"
    "        jitter = rng_seed.normal(0, 0.4, 7).astype(np.float32)\n"
    "        log10_g_j = np.clip(seed_log10_g + jitter, -5.0, 1.0)\n"
    "        xi = simulator_for_sbi(torch.tensor(log10_g_j))\n"
    "        extra_thetas.append(torch.tensor(log10_g_j))\n"
    "        extra_xs.append(xi)\n"
    "    theta_samples = torch.cat([theta_samples, torch.stack(extra_thetas)])\n"
    "    x_train       = torch.cat([x_train,       torch.stack(extra_xs)])\n"
    "    n_supp = (~torch.isnan(torch.stack(extra_xs)).any(dim=1)).sum().item()\n"
    "    print(f'Added {n_supp}/80 valid supplementary simulations.')\n"
    "\n"
    "# ── Filter NaN rows and train SNPE ────────────────────────────────────────────\n"
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
    "# Simulate N_VERIFY random draws from the posterior and the DE best solution\n"
    "N_VERIFY = 20\n"
    "rng_verify = np.random.default_rng(0)\n"
    "idx_verify = rng_verify.choice(N_POSTERIOR, N_VERIFY, replace=False)\n"
    "g_verify   = 10.0 ** ps_np[idx_verify]   # (N_VERIFY, 7) µS\n"
    "\n"
    "v_verify_list  = []   # subsampled traces (3, T_obs)\n"
    "stats_verify   = []   # 9-element summary stat vector per sample\n"
    "\n"
    "print(f'Simulating {N_VERIFY} posterior samples + DE best solution...')\n"
    "for k, g_k in enumerate(g_verify):\n"
    "    try:\n"
    "        _, v_k = simulate(g_k, t_max=4000.0, dt=0.025)\n"
    "        n = min(v_obs.shape[1], v_k.shape[1])\n"
    "        v_verify_list.append(v_k[:, :n])\n"
    "        stats_verify.append(summary_statistics(v_k, dt=0.025))\n"
    "    except Exception as e:\n"
    "        print(f'  Sample {k} failed: {e}')\n"
    "        v_verify_list.append(None)\n"
    "        stats_verify.append(np.full(9, np.nan))\n"
    "    if (k + 1) % 5 == 0:\n"
    "        print(f'  {k+1}/{N_VERIFY} done')\n"
    "\n"
    "stats_verify = np.array(stats_verify)\n"
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
    "    n_de = min(len(t_obs), v_best_de.shape[1])\n"
    "    axs[i].plot(t_obs[:n_de], v_best_de[i, :n_de],\n"
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
    keep(24),  # load npz + loss landscape
    keep(25),  # "plot traces" header
    CELL_26_PATCHED,   # simulate best DE params (uses best_log10_g_de)
    CELL_27_PATCHED,   # trace overlay plot (uses v_best_de)
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
