# Post-Selection Sharpe Ratio Estimators: Monte Carlo Study

Reproducible Monte Carlo evaluation of nine post-selection estimators for the
true Sharpe ratio of the best-ranked strategy, corresponding to Figures 1–17
and Tables 1–8 of the reference paper.

## Overview

When a practitioner selects the strategy with the **highest sample Sharpe ratio**
from a pool of *k* candidates, the sample maximum is a biased estimate of the
true Sharpe ratio of the chosen strategy (winner's curse / selection bias).
This project measures bias and RMSE of nine correction methods across a
comprehensive Monte Carlo grid of parameters.

### Nine Estimators

| ID | Name | Method |
|----|------|--------|
| E1 | Biased | Raw sample Sharpe (no correction) |
| E2 | Grand Mean | Mean sample Sharpe across all strategies |
| E3 | James-Stein | JS shrinkage toward grand mean (daily units) |
| E4 | GMLEB | Jiang-Zhang NPMLE empirical Bayes (EM posterior means) |
| E5 | SURE | Soft-threshold with SURE-optimal threshold |
| E6 | EBayes | Johnstone-Silverman sparse mixture EB thresholding |
| E7 | PolyMedian | Polyhedral-constraint conditional median |
| E8 | PolyMLE | Polyhedral-constraint truncated-normal MLE |
| E9 | ExpMax | Expected-maximum bias correction |

## Repository Structure

```
.
├── src/
│   ├── estimators.py     # All 9 estimator implementations
│   ├── simulation.py     # Return generation, population draws
│   └── analysis.py       # Bias, RMSE, rank-correlation utilities
├── exp/
│   ├── exp1_random_layouts.py      # Figs 1–6: Bias/RMSE vs σ_ζ
│   ├── exp2_fixed_layouts.py       # Figs 7–8: fixed-layout study
│   ├── exp3_correlated.py          # Figs 12–13: equicorrelated returns
│   ├── exp4_ranking_independent.py # Tables 1,2,6,7,8 + Figs 9–11: ranking
│   └── exp5_ranking_correlated.py  # Tables 3,4,5 + Figs 14–17: ranking+ρ
├── tests/
│   └── test_estimators.py          # pytest: ~40 tests
├── data/                            # Reserved for any external data
├── results/                         # Figures (PNG) + metrics (CSV) + RESULTS.md
├── run_experiments.py               # Main orchestration script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

### Full run (M = 500 simulations per setting)

```bash
python run_experiments.py
```

### Quick validation (M = 50 / 20 corners)

```bash
python run_experiments.py --quick
```

### Individual experiments

```bash
python exp/exp1_random_layouts.py
python exp/exp2_fixed_layouts.py
python exp/exp3_correlated.py
python exp/exp4_ranking_independent.py
python exp/exp5_ranking_correlated.py
```

### Tests

```bash
pytest tests/ -v
```

## Parameter Grids

| Experiment | σ_ζ | k | n | ρ | M | Total sims |
|------------|-----|---|---|---|---|-----------|
| 1 (random layouts) | {0.0,…,1.0} × 11 | {10,100,1000} | {504,1008,2016} | 0 | 500 | 297 × 500 = 148,500 |
| 2 (fixed layouts) | {0,…,2} × 11 | 100 | 1008 | 0 | 500 | 33 × 500 = 16,500 |
| 3 (equicorrelated) | 0.5 | 100 | 1008 | {0.05,…,0.95} × 11 | 500 | 33 × 500 = 16,500 |
| 4 (ranking, indep) | random | random | {126,…,2016} | 0 | — | 60,000 |
| 5 (ranking, corr) | random | random | {126,…,2016} | random | — | 96,000 |

## Key Findings

See `results/RESULTS.md` after running `run_experiments.py`.

**Short summary:**
- James-Stein (E3) achieves the lowest RMSE and highest Kendall τ under Gaussian layouts.
- GMLEB (E4) is second-best; robust to non-Gaussian layouts.
- Expected Max (E9) is the simplest closed-form correction; competitive at large k.
- Grand Mean (E2) and Biased (E1) show the worst RMSE in most settings.
- All estimators degrade under high equicorrelation ρ; relative ranking is preserved at moderate ρ.

## License

MIT
