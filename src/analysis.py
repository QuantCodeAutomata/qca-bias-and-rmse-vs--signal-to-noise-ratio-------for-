"""
Analysis utilities: bias/RMSE aggregation and rank correlation computation.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats

from src.estimators import ESTIMATOR_NAMES


def compute_bias(estimates: np.ndarray, true_values: np.ndarray) -> float:
    """Mean bias = mean(est - true), NaN-safe."""
    return float(np.nanmean(estimates - true_values))


def compute_rmse(estimates: np.ndarray, true_values: np.ndarray) -> float:
    """Root-mean-square error, NaN-safe."""
    return float(np.sqrt(np.nanmean((estimates - true_values) ** 2)))


def aggregate_simulation_results(
    records: np.ndarray,
    estimator_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate a (M, 10) simulation result array into per-estimator bias/RMSE.

    Parameters
    ----------
    records : ndarray, shape (M, 10)
        Column 0 = zeta_k (true target); columns 1-9 = estimates E1..E9.
    estimator_names : list of str, optional
        Defaults to ESTIMATOR_NAMES.

    Returns
    -------
    DataFrame with columns ['estimator', 'bias', 'rmse', 'nan_frac'].
    """
    if estimator_names is None:
        estimator_names = ESTIMATOR_NAMES
    zeta_k = records[:, 0]
    rows = []
    for i, name in enumerate(estimator_names):
        ests = records[:, i + 1]
        valid = ~np.isnan(ests)
        if valid.sum() == 0:
            rows.append({'estimator': name, 'bias': np.nan, 'rmse': np.nan,
                         'nan_frac': 1.0})
        else:
            rows.append({
                'estimator': name,
                'bias': compute_bias(ests[valid], zeta_k[valid]),
                'rmse': compute_rmse(ests[valid], zeta_k[valid]),
                'nan_frac': float((~valid).mean()),
            })
    return pd.DataFrame(rows)


def compute_rank_correlations(
    df: pd.DataFrame,
    estimator_cols: List[str],
    target_col: str = 'zeta_k'
) -> pd.DataFrame:
    """
    Kendall τ and Spearman ρ between each estimator column and the target.

    Parameters
    ----------
    df : DataFrame containing estimator and target columns.
    estimator_cols : list of column names.
    target_col : str.

    Returns
    -------
    DataFrame indexed by estimator with columns
    ['kendall_tau', 'spearman_rho', 'n_valid'].
    """
    rows = []
    for col in estimator_cols:
        valid = df[col].notna() & df[target_col].notna()
        vals = df.loc[valid, col].values
        tgt = df.loc[valid, target_col].values
        n_valid = len(vals)
        if n_valid < 10:
            tau, rho = np.nan, np.nan
        else:
            tau, _ = stats.kendalltau(vals, tgt)
            rho, _ = stats.spearmanr(vals, tgt)
        rows.append({'estimator': col,
                     'kendall_tau': float(tau) if not np.isnan(tau) else np.nan,
                     'spearman_rho': float(rho) if not np.isnan(rho) else np.nan,
                     'n_valid': n_valid})
    return pd.DataFrame(rows).set_index('estimator')


def compute_regret_ratios(rmse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Regret ratio = RMSE_e / min_e(RMSE) per parameter setting (row).

    Parameters
    ----------
    rmse_df : DataFrame with rows = settings, columns = estimators.

    Returns
    -------
    DataFrame of same shape with regret ratios.
    """
    min_rmse = rmse_df.min(axis=1)
    min_rmse = min_rmse.replace(0, np.nan)   # avoid division by zero
    return rmse_df.div(min_rmse, axis=0)
