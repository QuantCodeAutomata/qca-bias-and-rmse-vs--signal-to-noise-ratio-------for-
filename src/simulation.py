"""
Simulation utilities for the Monte Carlo post-selection Sharpe ratio study.
Handles population SNR generation, return generation (i.i.d. and equicorrelated),
sample Sharpe computation, and single-simulation runners.
"""
import numpy as np
from typing import Tuple

_SQRT252 = np.sqrt(252.0)


def draw_population_snr(
    layout: str,
    sigma_zeta: float,
    k: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw k population Sharpe ratios (annual) from the specified layout.

    Parameters
    ----------
    layout : str  –  'gaussian', 'uniform', or 'bimodal'
    sigma_zeta : float  –  spread parameter yr^(-1/2)
    k : int  –  number of strategies
    rng : np.random.Generator

    Returns
    -------
    zeta_annual : ndarray, shape (k,)
    """
    if layout == 'gaussian':
        return rng.normal(0.0, max(sigma_zeta, 0.0), k)
    elif layout == 'uniform':
        half = np.sqrt(3.0) * sigma_zeta
        return rng.uniform(-half, half, k)
    elif layout == 'bimodal':
        if sigma_zeta == 0.0:
            return np.zeros(k)
        return rng.choice([-sigma_zeta, sigma_zeta], size=k)
    else:
        raise ValueError(f'Unknown layout: {layout!r}')


def generate_iid_returns(
    n: int,
    k: int,
    zeta_annual: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate i.i.d. returns X = Z + zeta_daily^T, shape (n, k).

    Each column j has mean zeta_annual[j]/sqrt(252) and unit variance.
    """
    zeta_daily = zeta_annual / _SQRT252
    Z = rng.standard_normal((n, k))
    return Z + zeta_daily[None, :]


def generate_correlated_returns(
    n: int,
    k: int,
    zeta_annual: np.ndarray,
    rho: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate equicorrelated returns via common-factor model.

    X_{tj} = zeta_j_daily + sqrt(rho)*W_t + sqrt(1-rho)*eps_{tj}
    Corr(X_j, X_l) = rho for j≠l, Var(X_j) = 1.

    Parameters
    ----------
    rho : float  –  equicorrelation in (0, 1)
    """
    zeta_daily = zeta_annual / _SQRT252
    W = rng.standard_normal(n)           # common factor
    eps = rng.standard_normal((n, k))    # idiosyncratic noise
    return (zeta_daily[None, :]
            + np.sqrt(rho) * W[:, None]
            + np.sqrt(1.0 - rho) * eps)


def compute_sample_sharpe(X: np.ndarray) -> np.ndarray:
    """
    Annualized sample Sharpe ratios from daily returns matrix X (n x k).

    Returns
    -------
    zhat_annual : ndarray, shape (k,)
    """
    mu = X.mean(axis=0)
    sig = X.std(axis=0, ddof=1)
    # Guard against zero std (degenerate columns)
    sig = np.where(sig < 1e-12, 1e-12, sig)
    return (mu / sig) * _SQRT252


def sort_by_sample_sharpe(
    zhat_annual: np.ndarray,
    zeta_annual: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort both arrays ascending by sample Sharpe ratio.

    Returns (zhat_sorted, zeta_sorted) of shape (k,).
    """
    pi = np.argsort(zhat_annual)
    return zhat_annual[pi], zeta_annual[pi]


def run_single_simulation(
    zeta_annual: np.ndarray,
    n: int,
    rng: np.random.Generator,
    rho: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full single simulation: generate returns → sample Sharpe → sort ascending.

    Returns (zhat_sorted, zeta_sorted), both of shape (k,).
    """
    k = len(zeta_annual)
    if rho == 0.0:
        X = generate_iid_returns(n, k, zeta_annual, rng)
    else:
        X = generate_correlated_returns(n, k, zeta_annual, rho, rng)
    zhat = compute_sample_sharpe(X)
    return sort_by_sample_sharpe(zhat, zeta_annual)
