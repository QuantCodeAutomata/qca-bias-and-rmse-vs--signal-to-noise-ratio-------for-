"""
Experiment 2: Bias and RMSE for Fixed Population Layouts (all-good, one-good, two-good)
– Independent Returns. Corresponds to Figures 7–8.
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
from src.simulation import generate_iid_returns, compute_sample_sharpe, sort_by_sample_sharpe
from exp.exp1_random_layouts import ESTIMATOR_COLORS, ESTIMATOR_LS, PLOT_ESTIMATORS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 1008
K = 100
M = int(os.environ.get('MC_M_OVERRIDE', 500))
MASTER_SEED = 2024

LAYOUTS_FIXED = ['all_good', 'one_good', 'two_good']
ZETA0_ALL_GOOD = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ZETA0_REST     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

ZETA0_GRID = {
    'all_good': ZETA0_ALL_GOOD,
    'one_good': ZETA0_REST,
    'two_good': ZETA0_REST,
}


# ---------------------------------------------------------------------------
# Build fixed population vector
# ---------------------------------------------------------------------------
def build_fixed_population(layout: str, zeta0: float, k: int) -> np.ndarray:
    """Construct the fixed population Sharpe-ratio vector."""
    if layout == 'all_good':
        return np.full(k, zeta0)
    elif layout == 'one_good':
        v = np.full(k, -zeta0)
        v[0] = zeta0
        return v
    elif layout == 'two_good':
        v = np.full(k, -zeta0)
        v[0] = zeta0
        v[1] = zeta0
        return v
    else:
        raise ValueError(f'Unknown layout: {layout!r}')


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_fixed_setting(layout: str, zeta0: float, k: int, n: int,
                      M: int, seed: int) -> np.ndarray:
    """
    M simulations with a FIXED population vector.
    Returns array (M, 10): [zeta_k, est_1…est_9].
    """
    rng = np.random.default_rng(seed)
    zeta_annual = build_fixed_population(layout, zeta0, k)
    records = np.empty((M, 10))
    for m in range(M):
        X = generate_iid_returns(n, k, zeta_annual, rng)
        zhat = compute_sample_sharpe(X)
        zhat_s, zeta_s = sort_by_sample_sharpe(zhat, zeta_annual)
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
    for layout in LAYOUTS_FIXED:
        for zeta0 in ZETA0_GRID[layout]:
            seed = int(rng_seed.integers(0, 2 ** 31))
            settings.append((layout, zeta0, K, N, M, seed))

    total_sims = len(settings) * M
    print(f'Experiment 2: {len(settings)} settings × {M} sims = {total_sims:,} simulations')

    results_raw = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_fixed_setting)(l, z0, k, n, m, s)
        for l, z0, k, n, m, s in tqdm(settings, desc='Exp2 settings')
    )

    rows = []
    for (layout, zeta0, k, n, m, s), rec in zip(settings, results_raw):
        row = {'layout': layout, 'zeta0': zeta0, 'k': k, 'n': n}
        row.update(agg(rec))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'exp2_results.csv'), index=False)
    print(f'  Saved exp2_results.csv  ({len(df)} rows)')

    # ---- Figure 7: Bias vs ζ₀ ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ci, layout in enumerate(LAYOUTS_FIXED):
        ax = axes[ci]
        sub = df[df['layout'] == layout].sort_values('zeta0')
        for name in PLOT_ESTIMATORS:
            col = f'bias_{name}'
            if col in sub.columns:
                ax.plot(sub['zeta0'], sub[col],
                        color=ESTIMATOR_COLORS[name], ls=ESTIMATOR_LS[name],
                        lw=1.5, label=name, marker='o', ms=3)
        ax.axhline(0, color='black', ls=':', lw=0.8)
        ax.set_xlabel('ζ₀ (yr$^{-1/2}$)', fontsize=9)
        ax.set_ylabel('Bias (yr$^{-1/2}$)', fontsize=9)
        ax.set_title(layout.replace('_', '-').capitalize(), fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle('Figure 7: Bias vs ζ₀ – Fixed Layouts (n=1008, k=100)', fontsize=11,
                 y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig7_bias_fixed.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig7_bias_fixed.png')

    # ---- Figure 8: RMSE vs ζ₀ ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ci, layout in enumerate(LAYOUTS_FIXED):
        ax = axes[ci]
        sub = df[df['layout'] == layout].sort_values('zeta0')
        for name in PLOT_ESTIMATORS:
            col = f'rmse_{name}'
            if col in sub.columns:
                ax.plot(sub['zeta0'], sub[col],
                        color=ESTIMATOR_COLORS[name], ls=ESTIMATOR_LS[name],
                        lw=1.5, label=name, marker='o', ms=3)
        ax.set_xlabel('ζ₀ (yr$^{-1/2}$)', fontsize=9)
        ax.set_ylabel('RMSE (yr$^{-1/2}$)', fontsize=9)
        ax.set_title(layout.replace('_', '-').capitalize(), fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(PLOT_ESTIMATORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle('Figure 8: RMSE vs ζ₀ – Fixed Layouts (n=1008, k=100)', fontsize=11,
                 y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig8_rmse_fixed.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig8_rmse_fixed.png')

    # Print summary
    print('\nSummary (all-good, ζ₀=0.5):')
    sub = df[(df['layout'] == 'all_good') & (df['zeta0'] == 0.5)]
    for name in ESTIMATOR_NAMES:
        b = sub[f'bias_{name}'].values[0]
        r = sub[f'rmse_{name}'].values[0]
        print(f'  {name:12s}  bias={b:+.3f}  rmse={r:.3f}')

    print('\nExperiment 2 complete.')
