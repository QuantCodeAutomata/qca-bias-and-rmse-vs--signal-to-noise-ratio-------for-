"""
Experiment 3: Bias and RMSE Under Equicorrelated Returns – Sensitivity to ρ.
Corresponds to Figures 12–13.
Estimators applied WITHOUT modification for correlation.
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
from src.simulation import (draw_population_snr, generate_correlated_returns,
                             compute_sample_sharpe, sort_by_sample_sharpe)
from exp.exp1_random_layouts import ESTIMATOR_COLORS, ESTIMATOR_LS, PLOT_ESTIMATORS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA_ZETA = 0.5
N = 1008
K = 100
M = int(os.environ.get('MC_M_OVERRIDE', 500))
MASTER_SEED = 3000

LAYOUTS = ['gaussian', 'uniform', 'bimodal']
RHO_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_corr_setting(layout: str, rho: float, sigma_zeta: float,
                     k: int, n: int, M: int, seed: int) -> np.ndarray:
    """M simulations with equicorrelated returns (ρ = rho)."""
    rng = np.random.default_rng(seed)
    records = np.empty((M, 10))
    for m in range(M):
        zeta = draw_population_snr(layout, sigma_zeta, k, rng)
        X = generate_correlated_returns(n, k, zeta, rho, rng)
        zhat = compute_sample_sharpe(X)
        zhat_s, zeta_s = sort_by_sample_sharpe(zhat, zeta)
        ests = apply_all_estimators(zhat_s, n)
        records[m, 0] = zeta_s[-1]
        records[m, 1:] = ests
    return records


def agg(records: np.ndarray) -> dict:
    zk = records[:, 0]
    out = {}
    for i, name in enumerate(ESTIMATOR_NAMES):
        e = records[:, i + 1]
        out[f'bias_{name}'] = float(np.nanmean(e - zk))
        out[f'rmse_{name}'] = float(np.sqrt(np.nanmean((e - zk) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    settings = []
    rng_seed = np.random.default_rng(MASTER_SEED)
    for layout in LAYOUTS:
        for rho in RHO_GRID:
            seed = int(rng_seed.integers(0, 2 ** 31))
            settings.append((layout, rho, SIGMA_ZETA, K, N, M, seed))

    print(f'Experiment 3: {len(settings)} settings × {M} sims = '
          f'{len(settings) * M:,} simulations')

    results_raw = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_corr_setting)(l, rho, sz, k, n, m, s)
        for l, rho, sz, k, n, m, s in tqdm(settings, desc='Exp3 settings')
    )

    rows = []
    for (layout, rho, sz, k, n, m, s), rec in zip(settings, results_raw):
        row = {'layout': layout, 'rho': rho, 'sigma_zeta': sz, 'k': k, 'n': n}
        row.update(agg(rec))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'exp3_results.csv'), index=False)
    print(f'  Saved exp3_results.csv  ({len(df)} rows)')

    # ---- Figure 12: Bias vs ρ ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ci, layout in enumerate(LAYOUTS):
        ax = axes[ci]
        sub = df[df['layout'] == layout].sort_values('rho')
        for name in PLOT_ESTIMATORS:
            col = f'bias_{name}'
            ax.plot(sub['rho'], sub[col],
                    color=ESTIMATOR_COLORS[name], ls=ESTIMATOR_LS[name],
                    lw=1.5, label=name, marker='o', ms=3)
        ax.axhline(0, color='black', ls=':', lw=0.8)
        ax.set_xlabel('ρ (equicorrelation)', fontsize=9)
        ax.set_ylabel('Bias (yr$^{-1/2}$)', fontsize=9)
        ax.set_title(layout.capitalize(), fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle('Figure 12: Bias vs ρ (σ_ζ=0.5, n=1008, k=100)', fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig12_bias_corr.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig12_bias_corr.png')

    # ---- Figure 13: RMSE vs ρ ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ci, layout in enumerate(LAYOUTS):
        ax = axes[ci]
        sub = df[df['layout'] == layout].sort_values('rho')
        for name in PLOT_ESTIMATORS:
            col = f'rmse_{name}'
            ax.plot(sub['rho'], sub[col],
                    color=ESTIMATOR_COLORS[name], ls=ESTIMATOR_LS[name],
                    lw=1.5, label=name, marker='o', ms=3)
        ax.set_xlabel('ρ (equicorrelation)', fontsize=9)
        ax.set_ylabel('RMSE (yr$^{-1/2}$)', fontsize=9)
        ax.set_title(layout.capitalize(), fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle('Figure 13: RMSE vs ρ (σ_ζ=0.5, n=1008, k=100)', fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig13_rmse_corr.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig13_rmse_corr.png')

    print('\nExperiment 3 complete.')
