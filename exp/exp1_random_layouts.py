"""
Experiment 1: Bias and RMSE vs. Signal-to-Noise Ratio (σ_ζ) for Random Layouts
– Independent Returns.

Covers Figures 1–6 of the paper. Monte Carlo study with M=500 simulations
per (layout, σ_ζ, k, n) combination.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from src.estimators import apply_all_estimators, ESTIMATOR_NAMES
from src.simulation import draw_population_snr, generate_iid_returns, compute_sample_sharpe, sort_by_sample_sharpe

# ---------------------------------------------------------------------------
# Colour / style scheme (consistent across all experiments)
# ---------------------------------------------------------------------------
ESTIMATOR_COLORS = {
    'Biased':     '#e41a1c',
    'GrandMean':  '#377eb8',
    'JamesStein': '#4daf4a',
    'GMLEB':      '#984ea3',
    'SURE':       '#ff7f00',
    'EBayes':     '#a65628',
    'PolyMedian': '#f781bf',
    'PolyMLE':    '#999999',
    'ExpMax':     '#8B0000',
}
ESTIMATOR_LS = {
    'Biased': '-', 'GrandMean': '--', 'JamesStein': '-',
    'GMLEB': '-', 'SURE': '-.', 'EBayes': ':',
    'PolyMedian': '--', 'PolyMLE': ':', 'ExpMax': '-',
}

# Estimators shown in bias/RMSE line plots (exclude polyhedral)
PLOT_ESTIMATORS = ['Biased', 'GrandMean', 'JamesStein', 'GMLEB', 'SURE', 'EBayes', 'ExpMax']

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
SIGMA_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_GRID      = [10, 100, 1000]
N_GRID      = [504, 1008, 2016]
LAYOUTS     = ['gaussian', 'uniform', 'bimodal']
M           = int(os.environ.get('MC_M_OVERRIDE', 500))
MASTER_SEED = 42

# Figure 5 additional grid
N_FIG5      = [504, 756, 1008, 1260, 1512, 1764, 2016]
SZ_FIG5     = [0.333, 0.667, 1.0]
K_FIG5      = 100


# ---------------------------------------------------------------------------
# Worker: run M simulations for one (layout, sigma_zeta, k, n) setting
# ---------------------------------------------------------------------------
def run_setting(layout: str, sigma_zeta: float, k: int, n: int,
                M: int, seed: int) -> np.ndarray:
    """
    Run M Monte-Carlo simulations.

    Returns ndarray of shape (M, 10): columns = [zeta_k, est_1…est_9].
    """
    rng = np.random.default_rng(seed)
    records = np.empty((M, 10))
    for m in range(M):
        zeta = draw_population_snr(layout, sigma_zeta, k, rng)
        zhat, zeta_s = sort_by_sample_sharpe(
            compute_sample_sharpe(generate_iid_returns(n, k, zeta, rng)),
            zeta
        )
        ests = apply_all_estimators(zhat, n)
        records[m, 0] = zeta_s[-1]
        records[m, 1:] = ests
    return records


# ---------------------------------------------------------------------------
# Aggregate: bias and RMSE from (M, 10) records
# ---------------------------------------------------------------------------
def agg(records: np.ndarray) -> dict:
    zk = records[:, 0]
    out = {}
    for i, name in enumerate(ESTIMATOR_NAMES):
        e = records[:, i + 1]
        out[f'bias_{name}'] = float(np.nanmean(e - zk))
        out[f'rmse_{name}'] = float(np.sqrt(np.nanmean((e - zk) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_grid(df_plot: pd.DataFrame, metric: str, row_var: str, row_vals: list,
              col_vals: list, title: str, ylabel: str, figname: str,
              results_dir: str) -> None:
    """3×3 grid plot of metric vs σ_ζ."""
    fig, axes = plt.subplots(len(row_vals), len(col_vals),
                             figsize=(14, 4 * len(row_vals)), sharex=True)
    axes = np.array(axes).reshape(len(row_vals), len(col_vals))
    for ri, rv in enumerate(row_vals):
        for ci, layout in enumerate(col_vals):
            ax = axes[ri, ci]
            sub = df_plot[(df_plot[row_var] == rv) & (df_plot['layout'] == layout)]
            sub = sub.sort_values('sigma_zeta')
            for name in PLOT_ESTIMATORS:
                col = f'{metric}_{name}'
                if col in sub.columns:
                    ax.plot(sub['sigma_zeta'], sub[col],
                            color=ESTIMATOR_COLORS[name],
                            ls=ESTIMATOR_LS[name], lw=1.5,
                            label=name, marker='o', ms=3)
            if metric == 'bias':
                ax.axhline(0, color='black', ls=':', lw=0.8)
            ax.set_title(f'{layout.capitalize()},  {row_var}={rv}', fontsize=9)
            ax.set_xlabel('σ_ζ', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, figname), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {figname}')


# ---------------------------------------------------------------------------
# Figure 5: RMSE vs n (years)
# ---------------------------------------------------------------------------
def plot_fig5(df5: pd.DataFrame, results_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ci, sz in enumerate(SZ_FIG5):
        ax = axes[ci]
        sub = df5[np.isclose(df5['sigma_zeta'], sz)]
        sub = sub.sort_values('n')
        for name in PLOT_ESTIMATORS:
            col = f'rmse_{name}'
            if col in sub.columns:
                ax.plot(sub['n'] / 252, sub[col],
                        color=ESTIMATOR_COLORS[name],
                        ls=ESTIMATOR_LS[name], lw=1.5,
                        label=name, marker='o', ms=3)
        ax.set_xscale('log')
        ax.set_xlabel('Backtest length (years)', fontsize=9)
        ax.set_ylabel('RMSE (yr$^{-1/2}$)', fontsize=9)
        ax.set_title(f'σ_ζ = {sz:.3f}', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Figure 5: RMSE vs Backtest Length (k=100)', fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig5_rmse_vs_n.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig5_rmse_vs_n.png')


# ---------------------------------------------------------------------------
# Figure 6: Regret CDF
# ---------------------------------------------------------------------------
def plot_fig6(df_main: pd.DataFrame, results_dir: str) -> None:
    """Empirical CDF of regret ratio RMSE_e / min_e(RMSE), faceted by layout."""
    all_names = ESTIMATOR_NAMES
    rmse_cols = [f'rmse_{n}' for n in all_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ci, layout in enumerate(LAYOUTS):
        ax = axes[ci]
        sub = df_main[df_main['layout'] == layout][rmse_cols].copy()
        # Compute regret per setting
        min_rmse = sub.min(axis=1)
        min_rmse = min_rmse.replace(0, np.nan)
        regret = sub.div(min_rmse, axis=0)
        for name in all_names:
            col = f'rmse_{name}'
            vals = regret[col].dropna().values
            vals = np.clip(vals, 1.0, 5.0)
            vals_sorted = np.sort(vals)
            cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
            ax.plot(vals_sorted, cdf,
                    color=ESTIMATOR_COLORS[name],
                    ls=ESTIMATOR_LS[name], lw=1.5, label=name)
        ax.set_xscale('log')
        ax.set_xlabel('Regret ratio (RMSE / min RMSE)', fontsize=9)
        ax.set_ylabel('Empirical CDF', fontsize=9)
        ax.set_title(layout.capitalize(), fontsize=10)
        ax.axvline(1.0, color='black', ls=':', lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Figure 6: Performance / Regret CDF', fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig6_regret_cdf.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig6_regret_cdf.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # Build list of all settings
    settings = []
    rng_seed = np.random.default_rng(MASTER_SEED)
    for layout in LAYOUTS:
        for sz in SIGMA_GRID:
            for k in K_GRID:
                for n in N_GRID:
                    seed = int(rng_seed.integers(0, 2 ** 31))
                    settings.append((layout, sz, k, n, M, seed))

    print(f'Experiment 1: {len(settings)} settings × {M} sims = '
          f'{len(settings) * M:,} total simulations')

    # Run in parallel
    results_raw = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_setting)(l, sz, k, n, m, s)
        for l, sz, k, n, m, s in tqdm(settings, desc='Exp1 settings')
    )

    # Collect into DataFrame
    rows = []
    for (layout, sz, k, n, m, s), rec in zip(settings, results_raw):
        row = {'layout': layout, 'sigma_zeta': sz, 'k': k, 'n': n}
        row.update(agg(rec))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'exp1_results.csv'), index=False)
    print(f'  Saved exp1_results.csv  ({len(df)} rows)')

    # ---- Figure 1: Bias vs σ_ζ, n=1008 ----
    print('  Generating Figure 1...')
    plot_grid(df[df['n'] == 1008], 'bias', 'k', K_GRID, LAYOUTS,
              'Figure 1: Bias vs σ_ζ (n=1008)',
              'Bias (yr$^{-1/2}$)', 'fig1_bias_random.png', results_dir)

    # ---- Figure 2: RMSE vs σ_ζ, n=1008 ----
    print('  Generating Figure 2...')
    plot_grid(df[df['n'] == 1008], 'rmse', 'k', K_GRID, LAYOUTS,
              'Figure 2: RMSE vs σ_ζ (n=1008)',
              'RMSE (yr$^{-1/2}$)', 'fig2_rmse_random.png', results_dir)

    # ---- Figure 3: Bias vs σ_ζ, k=100 ----
    print('  Generating Figure 3...')
    plot_grid(df[df['k'] == 100], 'bias', 'n', N_GRID, LAYOUTS,
              'Figure 3: Bias vs σ_ζ (k=100)',
              'Bias (yr$^{-1/2}$)', 'fig3_bias_k100.png', results_dir)

    # ---- Figure 4: RMSE vs σ_ζ, k=100 ----
    print('  Generating Figure 4...')
    plot_grid(df[df['k'] == 100], 'rmse', 'n', N_GRID, LAYOUTS,
              'Figure 4: RMSE vs σ_ζ (k=100)',
              'RMSE (yr$^{-1/2}$)', 'fig4_rmse_k100.png', results_dir)

    # ---- Figure 5: RMSE vs n (additional simulations) ----
    print('  Running Figure 5 simulations (k=100, extra n values)...')
    rng_fig5 = np.random.default_rng(MASTER_SEED + 1000)
    fig5_settings = []
    for layout in LAYOUTS:
        for sz in SZ_FIG5:
            for n in N_FIG5:
                seed = int(rng_fig5.integers(0, 2 ** 31))
                fig5_settings.append((layout, sz, K_FIG5, n, M, seed))

    fig5_raw = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_setting)(l, sz, k, n, m, s)
        for l, sz, k, n, m, s in tqdm(fig5_settings, desc='Fig5 settings')
    )
    rows5 = []
    for (layout, sz, k, n, m, s), rec in zip(fig5_settings, fig5_raw):
        row = {'layout': layout, 'sigma_zeta': sz, 'k': k, 'n': n}
        row.update(agg(rec))
        rows5.append(row)
    df5 = pd.DataFrame(rows5)
    # Average over layouts for Figure 5
    df5_avg = df5.groupby(['sigma_zeta', 'n']).mean(numeric_only=True).reset_index()
    plot_fig5(df5_avg, results_dir)

    # ---- Figure 6: Regret CDF ----
    print('  Generating Figure 6...')
    plot_fig6(df, results_dir)

    # Print summary table for n=1008, k=100, σ_ζ=0.5
    summary = df[(df['n'] == 1008) & (df['k'] == 100) & (df['sigma_zeta'] == 0.5)]
    print('\nSummary (n=1008, k=100, σ_ζ=0.5):')
    for name in ESTIMATOR_NAMES:
        row_g = summary[summary['layout'] == 'gaussian']
        if not row_g.empty:
            b = row_g[f'bias_{name}'].values[0]
            r = row_g[f'rmse_{name}'].values[0]
            print(f'  {name:12s}  bias={b:+.3f}  rmse={r:.3f}  [Gaussian]')

    print('\nExperiment 1 complete.')
