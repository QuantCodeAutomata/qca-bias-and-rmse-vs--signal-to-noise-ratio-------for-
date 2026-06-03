"""
Experiment 5: Ranking Study – Kendall's τ and Spearman's ρ Under Equicorrelated Returns.
800 corner parameters (σ_ζ, k, ρ), 5 backtest lengths, 3 layouts, 8 reps = 96,000 sims.
Produces Tables 3–5 and Figures 14–17.
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
from src.simulation import (draw_population_snr, generate_correlated_returns,
                             compute_sample_sharpe, sort_by_sample_sharpe)
from exp.exp1_random_layouts import ESTIMATOR_COLORS, ESTIMATOR_LS
from exp.exp4_ranking_independent import rank_corrs, plot_rank_by_group

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
_QUICK      = os.environ.get('MC_M_OVERRIDE', '') != ''
N_CORNERS   = 20  if _QUICK else 800
CORNER_SEED = 99999
N_GRID      = [126, 252, 504, 1008, 2016]
LAYOUTS     = ['gaussian', 'uniform', 'bimodal']
REPS        = 2   if _QUICK else 8
MASTER_SEED = 5000

EST_COLS  = [f'est_{n}' for n in ESTIMATOR_NAMES]
K_BINS    = [(10, 30), (31, 100), (101, 300), (301, 900), (901, 2500)]
SZ_BINS   = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
RHO_BINS  = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9)]


# ---------------------------------------------------------------------------
# Generate corner parameters (once, reproducible)
# ---------------------------------------------------------------------------
def generate_corners_corr(n_corners: int, seed: int):
    rng = np.random.default_rng(seed)
    sigma_zeta = rng.uniform(0.0, 1.0, n_corners)
    log_k = rng.uniform(np.log(10), np.log(2500), n_corners)
    k_vals = np.clip(np.round(np.exp(log_k)).astype(int), 10, 2500)
    rho_vals = rng.uniform(0.0, 0.9, n_corners)
    return sigma_zeta, k_vals, rho_vals


# ---------------------------------------------------------------------------
# Worker: run 8 reps for (corner_id, n, layout) with correlation ρ
# ---------------------------------------------------------------------------
def run_task_corr(corner_id: int, n: int, layout: str,
                  sigma_zeta: float, k: int, rho: float,
                  reps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(reps):
        zeta = draw_population_snr(layout, sigma_zeta, k, rng)
        X = generate_correlated_returns(n, k, zeta, rho, rng)
        zhat = compute_sample_sharpe(X)
        zhat_s, zeta_s = sort_by_sample_sharpe(zhat, zeta)
        ests = apply_all_estimators(zhat_s, n)
        rec = {
            'corner_id': corner_id, 'n': n, 'layout': layout, 'rep': rep,
            'sigma_zeta': sigma_zeta, 'k': k, 'rho': rho,
            'zeta_k': float(zeta_s[-1]),
        }
        for i, name in enumerate(ESTIMATOR_NAMES):
            rec[f'est_{name}'] = float(ests[i]) if np.isfinite(ests[i]) else np.nan
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # Generate corners
    sz_corners, k_corners, rho_corners = generate_corners_corr(N_CORNERS, CORNER_SEED)
    corners_df = pd.DataFrame({'sigma_zeta': sz_corners, 'k': k_corners, 'rho': rho_corners})
    corners_df.to_csv(os.path.join(results_dir, 'exp5_corners.csv'), index=False)

    # Build task list
    tasks = []
    rng_seed = np.random.default_rng(MASTER_SEED)
    for cid in range(N_CORNERS):
        sz  = float(sz_corners[cid])
        k   = int(k_corners[cid])
        rho = float(rho_corners[cid])
        for n in N_GRID:
            for layout in LAYOUTS:
                seed = int(rng_seed.integers(0, 2 ** 31))
                tasks.append((cid, n, layout, sz, k, rho, REPS, seed))

    total = len(tasks) * REPS
    print(f'Experiment 5: {len(tasks)} tasks × {REPS} reps = {total:,} simulations')

    # Parallel execution
    all_records_nested = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(run_task_corr)(cid, n, layout, sz, k, rho, reps, seed)
        for cid, n, layout, sz, k, rho, reps, seed in tqdm(tasks, desc='Exp5 tasks')
    )

    all_records = [r for sublist in all_records_nested for r in sublist]
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(results_dir, 'exp5_records.csv'), index=False)
    print(f'  Saved exp5_records.csv  ({len(df):,} rows)')

    # ---- Table 3: Overall rank correlations ----
    print('  Computing rank correlations...')
    table3 = rank_corrs(df, EST_COLS)
    table3_sorted = table3.sort_values('kendall_tau', ascending=False)
    table3_sorted.to_csv(os.path.join(results_dir, 'table3_overall_corr.csv'))
    print('\nTable 3 – Overall Rank Correlations (96k records, correlated):')
    print(table3_sorted[['kendall_tau', 'spearman_rho']].to_string(float_format='{:.4f}'.format))

    # ---- Table 4: By layout ----
    layout_results = {}
    for layout in LAYOUTS:
        sub = df[df['layout'] == layout]
        layout_results[layout] = rank_corrs(sub, EST_COLS)
    frames4 = []
    for layout, rc in layout_results.items():
        frame = rc[['kendall_tau', 'spearman_rho']].copy()
        frame.columns = [f'{layout}_tau', f'{layout}_rho']
        frames4.append(frame)
    table4 = pd.concat(frames4, axis=1)
    table4.to_csv(os.path.join(results_dir, 'table4_by_layout_corr.csv'))
    print('\nTable 4 – Rank Correlations by Layout (correlated):')
    print(table4.to_string(float_format='{:.4f}'.format))

    # ---- Figure 14: By n ----
    n_taus = {}
    for n in N_GRID:
        sub = df[df['n'] == n]
        n_taus[n] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(N_GRID, n_taus, 'n',
                       'Figure 14: Kendall τ by Backtest Length (Exp5, correlated)',
                       'fig14_rank_by_n_corr.png', results_dir,
                       'Backtest length (days)', log_x=False)

    # ---- Figure 15: By k bins ----
    k_bin_centers = []
    k_bin_taus = {}
    for lo, hi in K_BINS:
        sub = df[(df['k'] >= lo) & (df['k'] <= hi)]
        center = int((lo + hi) / 2)
        k_bin_centers.append(center)
        k_bin_taus[center] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(k_bin_centers, k_bin_taus, 'k',
                       'Figure 15: Kendall τ by Number of Strategies (Exp5, correlated)',
                       'fig15_rank_by_k_corr.png', results_dir,
                       'Number of strategies k', log_x=True)

    # ---- Figure 16: By σ_ζ bins ----
    sz_bin_centers = []
    sz_bin_taus = {}
    for lo, hi in SZ_BINS:
        sub = df[(df['sigma_zeta'] >= lo) & (df['sigma_zeta'] < hi)]
        center = round((lo + hi) / 2, 3)
        sz_bin_centers.append(center)
        sz_bin_taus[center] = rank_corrs(sub, EST_COLS)
    plot_rank_by_group(sz_bin_centers, sz_bin_taus, 'sigma_zeta',
                       'Figure 16: Kendall τ by Signal Spread σ_ζ (Exp5, correlated)',
                       'fig16_rank_by_sz_corr.png', results_dir,
                       'σ_ζ bin midpoint', log_x=False)

    # ---- Table 5 & Figure 17: By ρ bins ----
    rho_bin_centers = []
    rho_bin_taus = {}
    table5_frames = []
    for lo, hi in RHO_BINS:
        sub = df[(df['rho'] >= lo) & (df['rho'] < hi)]
        center = round((lo + hi) / 2, 3)
        rho_bin_centers.append(center)
        rc = rank_corrs(sub, EST_COLS)
        rho_bin_taus[center] = rc
        frame = rc[['kendall_tau', 'spearman_rho']].copy()
        frame.columns = [f'rho_{lo}_{hi}_tau', f'rho_{lo}_{hi}_rho']
        table5_frames.append(frame)
    table5 = pd.concat(table5_frames, axis=1)
    table5.to_csv(os.path.join(results_dir, 'table5_by_rho.csv'))
    print('\nTable 5 – Rank Correlations by ρ bin:')
    print(table5[[c for c in table5.columns if 'tau' in c]].to_string(float_format='{:.4f}'.format))

    plot_rank_by_group(rho_bin_centers, rho_bin_taus, 'rho',
                       'Figure 17: Kendall τ by Correlation ρ (Exp5)',
                       'fig17_rank_by_rho.png', results_dir,
                       'Equicorrelation ρ bin midpoint', log_x=False)

    print('\nExperiment 5 complete.')
