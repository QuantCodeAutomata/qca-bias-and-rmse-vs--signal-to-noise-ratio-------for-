"""
Experiment 4: Ranking Study – Kendall's τ and Spearman's ρ for Independent Returns.
500 corner parameters, 5 backtest lengths, 3 layouts, 8 reps = 60,000 simulations.
Produces Tables 1–2 and Figures 9–11.
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
from scipy import stats

from src.estimators import apply_all_estimators, ESTIMATOR_NAMES
from src.simulation import (draw_population_snr, generate_iid_returns,
                             compute_sample_sharpe, sort_by_sample_sharpe)
from exp.exp1_random_layouts import ESTIMATOR_COLORS, ESTIMATOR_LS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
_QUICK       = os.environ.get('MC_M_OVERRIDE', '') != ''
N_CORNERS    = 20  if _QUICK else 500
CORNER_SEED  = 12345
N_GRID       = [126, 252, 504, 1008, 2016]
LAYOUTS      = ['gaussian', 'uniform', 'bimodal']
REPS         = 2  if _QUICK else 8
MASTER_SEED  = 4000

EST_COLS = [f'est_{n}' for n in ESTIMATOR_NAMES]
K_BINS   = [(10, 30), (31, 100), (101, 300), (301, 900), (901, 2500)]
SZ_BINS  = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


# ---------------------------------------------------------------------------
# Generate corner parameters (once, reproducible)
# ---------------------------------------------------------------------------
def generate_corners(n_corners: int, seed: int):
    rng = np.random.default_rng(seed)
    sigma_zeta = rng.uniform(0.0, 1.0, n_corners)
    log_k = rng.uniform(np.log(10), np.log(2500), n_corners)
    k_vals = np.clip(np.round(np.exp(log_k)).astype(int), 10, 2500)
    return sigma_zeta, k_vals


# ---------------------------------------------------------------------------
# Worker: run 8 reps for (corner_id, n, layout)
# ---------------------------------------------------------------------------
def run_task(corner_id: int, n: int, layout: str,
             sigma_zeta: float, k: int, reps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(reps):
        zeta = draw_population_snr(layout, sigma_zeta, k, rng)
        X = generate_iid_returns(n, k, zeta, rng)
        zhat = compute_sample_sharpe(X)
        zhat_s, zeta_s = sort_by_sample_sharpe(zhat, zeta)
        ests = apply_all_estimators(zhat_s, n)
        rec = {
            'corner_id': corner_id, 'n': n, 'layout': layout, 'rep': rep,
            'sigma_zeta': sigma_zeta, 'k': k, 'zeta_k': float(zeta_s[-1]),
        }
        for i, name in enumerate(ESTIMATOR_NAMES):
            rec[f'est_{name}'] = float(ests[i]) if np.isfinite(ests[i]) else np.nan
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Rank correlation helper
# ---------------------------------------------------------------------------
def rank_corrs(df: pd.DataFrame, est_cols: list,
               target_col: str = 'zeta_k') -> pd.DataFrame:
    rows = []
    for col in est_cols:
        valid = df[col].notna() & df[target_col].notna()
        v = df.loc[valid, col].values
        t = df.loc[valid, target_col].values
        n_valid = len(v)
        if n_valid < 10:
            tau, rho = np.nan, np.nan
        else:
            tau, _ = stats.kendalltau(v, t)
            rho, _ = stats.spearmanr(v, t)
        name = col.replace('est_', '')
        rows.append({'estimator': name, 'kendall_tau': tau,
                     'spearman_rho': rho, 'n_valid': n_valid})
    return pd.DataFrame(rows).set_index('estimator')


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_rank_by_group(group_vals, group_taus: dict, group_label: str,
                       title: str, figname: str, results_dir: str,
                       x_label: str, log_x: bool = False) -> None:
    """Line plot: x = group values, y = Kendall τ per estimator."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ESTIMATOR_NAMES:
        vals = [group_taus[gv].loc[name, 'kendall_tau']
                if name in group_taus[gv].index else np.nan
                for gv in group_vals]
        ax.plot(group_vals, vals,
                color=ESTIMATOR_COLORS[name], ls=ESTIMATOR_LS[name],
                lw=1.5, label=name, marker='o', ms=4)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Kendall τ", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.axhline(0, color='black', ls=':', lw=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, figname), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {figname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # Generate corner parameters
    sz_corners, k_corners = generate_corners(N_CORNERS, CORNER_SEED)
    corners_df = pd.DataFrame({'sigma_zeta': sz_corners, 'k': k_corners})
    corners_df.to_csv(os.path.join(results_dir, 'exp4_corners.csv'), index=False)

    # Build task list
    tasks = []
    rng_seed = np.random.default_rng(MASTER_SEED)
    for cid in range(N_CORNERS):
        sz = float(sz_corners[cid])
        k  = int(k_corners[cid])
        for n in N_GRID:
            for layout in LAYOUTS:
                seed = int(rng_seed.integers(0, 2 ** 31))
                tasks.append((cid, n, layout, sz, k, REPS, seed))

    total = len(tasks) * REPS
    print(f'Experiment 4: {len(tasks)} tasks × {REPS} reps = {total:,} simulations')

    # Parallel execution
    all_records_nested = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_task)(cid, n, layout, sz, k, reps, seed)
        for cid, n, layout, sz, k, reps, seed in tqdm(tasks, desc='Exp4 tasks')
    )

    # Flatten into DataFrame
    all_records = [r for sublist in all_records_nested for r in sublist]
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(results_dir, 'exp4_records.csv'), index=False)
    print(f'  Saved exp4_records.csv  ({len(df):,} rows)')

    # ---- Table 1: Overall rank correlations ----
    print('  Computing rank correlations...')
    table1 = rank_corrs(df, EST_COLS)
    table1_sorted = table1.sort_values('kendall_tau', ascending=False)
    table1_sorted.to_csv(os.path.join(results_dir, 'table1_overall.csv'))
    print('\nTable 1 – Overall Rank Correlations (60k records):')
    print(table1_sorted[['kendall_tau', 'spearman_rho']].to_string(float_format='{:.4f}'.format))

    # ---- Table 2: By layout ----
    layout_results = {}
    for layout in LAYOUTS:
        sub = df[df['layout'] == layout]
        layout_results[layout] = rank_corrs(sub, EST_COLS)
    table2_frames = []
    for layout, rc in layout_results.items():
        frame = rc[['kendall_tau', 'spearman_rho']].copy()
        frame.columns = [f'{layout}_tau', f'{layout}_rho']
        table2_frames.append(frame)
    table2 = pd.concat(table2_frames, axis=1)
    table2.to_csv(os.path.join(results_dir, 'table2_by_layout.csv'))
    print('\nTable 2 – Rank Correlations by Layout:')
    print(table2.to_string(float_format='{:.4f}'.format))

    # ---- Figure 9: Rank correlations by n ----
    n_taus = {}
    for n in N_GRID:
        sub = df[df['n'] == n]
        n_taus[n] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(N_GRID, n_taus, 'n',
                       'Figure 9: Kendall τ by Backtest Length (Exp4)',
                       'fig9_rank_by_n.png', results_dir,
                       'Backtest length (days)', log_x=False)

    # ---- Figure 10: By k bins ----
    k_bin_centers = []
    k_bin_taus = {}
    for lo, hi in K_BINS:
        sub = df[(df['k'] >= lo) & (df['k'] <= hi)]
        center = int((lo + hi) / 2)
        k_bin_centers.append(center)
        k_bin_taus[center] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(k_bin_centers, k_bin_taus, 'k',
                       'Figure 10: Kendall τ by Number of Strategies (Exp4)',
                       'fig10_rank_by_k.png', results_dir,
                       'Number of strategies k', log_x=True)

    # ---- Figure 11: By σ_ζ bins ----
    sz_bin_centers = []
    sz_bin_taus = {}
    for lo, hi in SZ_BINS:
        sub = df[(df['sigma_zeta'] >= lo) & (df['sigma_zeta'] < hi)]
        center = round((lo + hi) / 2, 3)
        sz_bin_centers.append(center)
        sz_bin_taus[center] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(sz_bin_centers, sz_bin_taus, 'sigma_zeta',
                       'Figure 11: Kendall τ by Signal Spread σ_ζ (Exp4)',
                       'fig11_rank_by_sz.png', results_dir,
                       'σ_ζ bin midpoint', log_x=False)

    print('\nExperiment 4 complete.')
