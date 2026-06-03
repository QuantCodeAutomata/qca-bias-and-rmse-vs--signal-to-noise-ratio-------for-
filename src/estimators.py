"""
Post-selection estimators for the true Sharpe ratio of the best-performing strategy.
All estimators receive the sorted (ascending) annual sample Sharpe ratios and return
an estimate in annual yr^(-1/2) units.

References: Bias-corrected estimation after selection in multiple testing (2024).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from typing import Optional

# ---------------------------------------------------------------------------
# Estimator constants
# ---------------------------------------------------------------------------
ESTIMATOR_NAMES = [
    'Biased', 'GrandMean', 'JamesStein', 'GMLEB', 'SURE',
    'EBayes', 'PolyMedian', 'PolyMLE', 'ExpMax'
]

_SQRT252 = np.sqrt(252.0)
_GAMMA = 0.5772157   # Euler-Mascheroni constant


# ---------------------------------------------------------------------------
# E1: Biased
# ---------------------------------------------------------------------------
def est_biased(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E1: Biased – raw sample Sharpe of the best strategy."""
    return float(zhat_annual_sorted[-1])


# ---------------------------------------------------------------------------
# E2: Grand Mean
# ---------------------------------------------------------------------------
def est_grand_mean(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E2: Grand Mean of all sample Sharpe ratios."""
    return float(np.mean(zhat_annual_sorted))


# ---------------------------------------------------------------------------
# E3: James-Stein
# ---------------------------------------------------------------------------
def est_james_stein(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E3: James-Stein shrinkage toward grand mean (computed in daily units)."""
    k = len(zhat_annual_sorted)
    zhat_d = zhat_annual_sorted / _SQRT252
    zbar_d = np.mean(zhat_d)
    y_d = zhat_d - zbar_d
    D = float(np.dot(y_d, y_d))
    if D < 1e-15 or k <= 2:
        sv = 0.0
    else:
        sv = max(0.0, 1.0 - (k - 2) * (1.0 / n) / D)
    return float((zbar_d + sv * (zhat_d[-1] - zbar_d)) * _SQRT252)


# ---------------------------------------------------------------------------
# E4: GMLEB (Jiang-Zhang NPMLE via EM algorithm)
# Custom – Context7 found no library equivalent (paper Eq. E4).
# ---------------------------------------------------------------------------
def _npmle_em(y: np.ndarray, sigma2: float, n_support: int = 60,
              max_iter: int = 60, tol: float = 1e-4) -> np.ndarray:
    """
    NPMLE for Gaussian location mixture via EM (Kiefer-Wolfowitz).
    Model: y_i = theta_i + eps_i, eps_i ~ N(0, sigma2).
    Returns posterior means E[theta_i | y_i].
    """
    k = len(y)
    sigma = np.sqrt(sigma2)
    # Support grid spanning data range
    m = np.linspace(y.min() - 3.0 * sigma, y.max() + 3.0 * sigma, n_support)
    # Likelihood matrix L[i,j], shape (k, n_support) — computed in log for stability
    diff = (y[:, None] - m[None, :]) / sigma
    L = np.exp(np.clip(-0.5 * diff * diff, -500, 0)) / (np.sqrt(2.0 * np.pi) * sigma)
    L = np.maximum(L, 1e-300)
    # Initialize weights uniformly
    w = np.ones(n_support) / n_support
    for _ in range(max_iter):
        f = L @ w                    # shape (k,)
        f = np.maximum(f, 1e-300)
        r = (L * w[None, :]) / f[:, None]   # responsibilities, shape (k, n_support)
        w_new = r.mean(axis=0)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    # Posterior means
    f_final = np.maximum(L @ w, 1e-300)
    return (L * w[None, :]) @ m / f_final


def est_gmleb(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E4: GMLEB (Jiang-Zhang) empirical Bayes estimator via NPMLE EM algorithm."""
    zhat_d = zhat_annual_sorted / _SQRT252
    zbar_d = float(np.mean(zhat_d))
    y_d = zhat_d - zbar_d
    sigma2 = 1.0 / n
    k = len(y_d)
    n_support = min(k, 60)
    if n_support < 3:
        n_support = 3
    post_means = _npmle_em(y_d, sigma2, n_support)
    return float((zbar_d + post_means[-1]) * _SQRT252)


# ---------------------------------------------------------------------------
# E5: SURE (Stein Unbiased Risk Estimate – soft threshold)
# Custom – Context7 found no library equivalent (paper Eq. E5).
# ---------------------------------------------------------------------------
def est_sure(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """
    E5: SURE-optimal soft-threshold estimator.
    Vectorised O(k log k) implementation: threshold selected from sorted |y_i|.
    """
    zhat_d = zhat_annual_sorted / _SQRT252
    zbar_d = float(np.mean(zhat_d))
    y_d = zhat_d - zbar_d
    sigma2 = 1.0 / n
    k = len(y_d)
    y_abs_sorted = np.sort(np.abs(y_d))   # ascending
    y_sq = y_abs_sorted ** 2
    cumsum_sq = np.cumsum(y_sq)
    j_arr = np.arange(k)
    # SURE(t_j) = -k*sigma2 + cumsum_sq[j] + (k-j-1)*y_sq[j] + 2*sigma2*(k-j-1)
    count_gt = k - j_arr - 1
    sure_vals = -k * sigma2 + cumsum_sq + count_gt * y_sq + 2.0 * sigma2 * count_gt
    # SURE at t=0: -k*sigma2 + 0 + 2*sigma2*k = k*sigma2
    sure_t0 = float(k * sigma2)
    best_j = int(np.argmin(sure_vals))
    if sure_t0 <= sure_vals[best_j]:
        best_t = 0.0
    else:
        best_t = float(y_abs_sorted[best_j])
    yk = y_d[-1]
    shrunk = np.sign(yk) * max(0.0, abs(yk) - best_t)
    return float((zbar_d + shrunk) * _SQRT252)


# ---------------------------------------------------------------------------
# E6: Empirical Bayes (Johnstone-Silverman EbayesThresh approximation)
# Custom – Context7 found no library equivalent (paper Eq. E6 / J-S 2005).
# ---------------------------------------------------------------------------
def _laplace_normal_marginal(y: np.ndarray, a: float, sigma: float) -> np.ndarray:
    """
    PDF of Laplace(0,a) convolved with N(0,sigma^2).
    m(y; a, sigma) = (a/2)*exp(a^2*sigma^2/2) *
                     [exp(-a*y)*Phi((y-a*sigma^2)/sigma) + exp(a*y)*Phi((-y-a*sigma^2)/sigma)]
    """
    v = a * sigma
    half_v2 = 0.5 * v * v
    # Clip for numerical safety
    ay = a * np.abs(y)
    log_scale = half_v2 - ay
    # Use log-domain for very negative exponents
    safe = log_scale > -500
    result = np.zeros(len(y))
    if np.any(safe):
        scale = np.exp(np.clip(log_scale[safe], -500, 0))
        arg1 = (np.abs(y[safe]) - v * sigma) / sigma   # (|y|-a*sig^2)/sig = (|y|/sig - v)
        result[safe] = a * scale * norm.cdf(arg1)
    result = np.maximum(result, 1e-300)
    return result


def est_ebayes(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """
    E6: Empirical Bayes (Johnstone-Silverman EbayesThresh).
    Sparse mixture prior pi*Laplace(0,a) + (1-pi)*delta_0.
    2-D grid search over (pi, a) to maximize marginal log-likelihood,
    then apply posterior-median soft-threshold shrinkage.
    """
    zhat_d = zhat_annual_sorted / _SQRT252
    zbar_d = float(np.mean(zhat_d))
    y_d = zhat_d - zbar_d
    sigma = 1.0 / np.sqrt(n)
    k = len(y_d)

    phi0 = norm.pdf(y_d, scale=sigma)   # null component, shape (k,)

    # Grid over a: from sigma^{-1}/4 to 4*sigma^{-1}  (log-spaced)
    a_grid = np.exp(np.linspace(np.log(0.25 / sigma), np.log(4.0 / sigma), 20))
    pi_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

    best_ll = -np.inf
    best_a = 1.0 / sigma
    best_pi = 0.5

    for a_ in a_grid:
        lap = _laplace_normal_marginal(y_d, a_, sigma)  # shape (k,)
        for pi_ in pi_grid:
            mix = (1.0 - pi_) * phi0 + pi_ * lap
            ll = float(np.sum(np.log(np.maximum(mix, 1e-300))))
            if ll > best_ll:
                best_ll = ll
                best_a = a_
                best_pi = pi_

    a_opt, pi_opt = best_a, best_pi
    # Posterior-median shrinkage
    lap_opt = _laplace_normal_marginal(y_d, a_opt, sigma)
    mix_opt = np.maximum((1.0 - pi_opt) * phi0 + pi_opt * lap_opt, 1e-300)
    p_nonzero = (pi_opt * lap_opt) / mix_opt        # shape (k,)
    thresh = (sigma ** 2) * a_opt                    # Laplace posterior median threshold
    shrunk = np.where(
        p_nonzero > 0.5,
        np.sign(y_d) * np.maximum(0.0, np.abs(y_d) - thresh),
        0.0
    )
    return float((zbar_d + shrunk[-1]) * _SQRT252)


# ---------------------------------------------------------------------------
# E7: Polyhedral Median
# Custom – Context7 found no library equivalent (paper Eq. E7).
# ---------------------------------------------------------------------------
def est_poly_median(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E7: Polyhedral Median – truncated-normal conditional median."""
    ak = zhat_annual_sorted[-1] / _SQRT252
    akm1 = zhat_annual_sorted[-2] / _SQRT252
    sq_n = np.sqrt(n)
    if abs(ak - akm1) < 1e-6 / sq_n:
        return np.nan

    def f(z: float) -> float:
        num = norm.cdf(sq_n * (ak - z)) - norm.cdf(sq_n * (akm1 - z))
        den = 1.0 - norm.cdf(sq_n * (akm1 - z))
        if den < 1e-300:
            return -0.5
        return num / den - 0.5

    lo = akm1 - 100.0 / sq_n
    hi = ak - 1e-10
    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return np.nan
        root = brentq(f, lo, hi, xtol=1e-10, maxiter=200)
        return float(root * _SQRT252)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# E8: Polyhedral MLE
# Custom – Context7 found no library equivalent (paper Eq. E8).
# ---------------------------------------------------------------------------
def est_poly_mle(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E8: Polyhedral MLE – score-function root-finding for truncated normal."""
    ak = zhat_annual_sorted[-1] / _SQRT252
    akm1 = zhat_annual_sorted[-2] / _SQRT252
    sq_n = np.sqrt(n)
    if abs(ak - akm1) < 1e-6 / sq_n:
        return np.nan

    def score(z: float) -> float:
        x = sq_n * (akm1 - z)
        sf = norm.sf(x)
        if sf < 1e-300:
            sf = 1e-300
        return n * (ak - z) + sq_n * norm.pdf(x) / sf

    lo = ak - 200.0 / sq_n
    hi = ak - 1e-12
    try:
        slo, shi = score(lo), score(hi)
        if slo * shi > 0:
            return np.nan
        root = brentq(score, lo, hi, xtol=1e-10, maxiter=200)
        return float(root * _SQRT252)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# E9: Expected Max
# ---------------------------------------------------------------------------
def est_expected_max(zhat_annual_sorted: np.ndarray, n: int) -> float:
    """E9: Expected Max correction – subtract bias of maximum of k Gaussians."""
    k = len(zhat_annual_sorted)
    sigma_n = np.sqrt(252.0 / n)  # annual noise std
    arg1 = np.clip(1.0 - 1.0 / k, 1e-12, 1.0 - 1e-12)
    arg2 = np.clip(1.0 - 1.0 / (k * np.e), 1e-12, 1.0 - 1e-12)
    C = sigma_n * ((1.0 - _GAMMA) * norm.ppf(arg1) + _GAMMA * norm.ppf(arg2))
    return float(zhat_annual_sorted[-1] - C)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
ESTIMATOR_FUNCS = [
    est_biased, est_grand_mean, est_james_stein, est_gmleb, est_sure,
    est_ebayes, est_poly_median, est_poly_mle, est_expected_max
]


def apply_all_estimators(zhat_annual_sorted: np.ndarray, n: int) -> np.ndarray:
    """Apply all 9 estimators; return array of length 9 (NaN on failure)."""
    results = np.empty(9)
    for i, func in enumerate(ESTIMATOR_FUNCS):
        try:
            val = func(zhat_annual_sorted, n)
            results[i] = np.nan if (val is None or not np.isfinite(val)) else float(val)
        except Exception:
            results[i] = np.nan
    return results
