# AGENTS.md — Repository Knowledge for AI Agents

## Project Overview
Monte Carlo simulation study of **9 post-selection estimators** for the true Sharpe
ratio of the best-ranked strategy, replicating Figures 1–17 and Tables 1–5 of the paper.
Language: Python 3.10+.

## Repository Layout
```
src/estimators.py    # 9 post-selection estimators (E1–E9), apply_all_estimators()
src/simulation.py    # return generation (iid + equicorrelated), sample Sharpe, sort helpers
src/analysis.py      # bias, RMSE, rank correlations, regret ratios
exp/exp1_random_layouts.py  # Exp 1: Bias/RMSE vs σ_ζ (Figs 1–6)
exp/exp2_fixed_layouts.py   # Exp 2: Fixed layouts all-good/one-good/two-good (Figs 7–8)
exp/exp3_correlated.py      # Exp 3: Equicorrelation ρ sweep (Figs 12–13)
exp/exp4_ranking_independent.py # Exp 4: Kendall τ / Spearman ρ, iid returns (Tables 1-2, Figs 9-11)
exp/exp5_ranking_correlated.py  # Exp 5: Kendall τ / Spearman ρ, correlated returns (Tables 3-5, Figs 14-17)
tests/test_estimators.py    # 60 pytest tests (all pass)
run_experiments.py           # Orchestrator; generates results/RESULTS.md
results/                     # All output: CSVs, PNGs, RESULTS.md
data/                        # Placeholder (no real market data used)
```

## Key Parameters
- **M** simulations per setting: 200 (Exp 1–3) / 500 default; override via `MC_M_OVERRIDE` env var.
- **Exp 4**: 500 corners (σ_ζ, k) × 5n × 3 layouts × 8 reps = 60,000 simulations; `MC_M_OVERRIDE` triggers quick mode (20 corners, 2 reps).
- **Exp 5**: 800 corners (σ_ζ, k, ρ) × 5n × 3 layouts × 8 reps = 96,000 simulations; same quick flag.

## Running Experiments
```bash
# Quick validation (M=50 / reduced corners)
python run_experiments.py --quick

# Full run (Exp 1–3 with M=200, Exp 4–5 full)
MC_M_OVERRIDE=200 python exp/exp1_random_layouts.py
MC_M_OVERRIDE=200 python exp/exp2_fixed_layouts.py
MC_M_OVERRIDE=200 python exp/exp3_correlated.py
python exp/exp4_ranking_independent.py
python exp/exp5_ranking_correlated.py

# Tests
python -m pytest tests/ -v
```

## Estimator Formulas (summary)
| # | Name | Key formula |
|---|------|-------------|
| E1 | Biased | `ζ̂_k_annual` (raw max sample Sharpe, annualised) |
| E2 | Grand Mean | `mean(ζ̂_annual)` |
| E3 | James-Stein | JS shrinkage in daily units; sv = max(0,1−(k−2)/n/D) |
| E4 | GMLEB | NPMLE via EM on Gaussian mixture; posterior mean for best strategy |
| E5 | SURE | Soft-threshold t minimising Stein's Unbiased Risk Estimate |
| E6 | EBayes | Johnstone–Silverman sparse-mixture EB posterior median |
| E7 | Poly Median | Bisection on truncated-normal conditional median |
| E8 | Poly MLE | Score-equation root-find for polyhedral MLE |
| E9 | Expected Max | `ζ̂_k − σ_n·((1−γ)·Φ⁻¹(1−1/k) + γ·Φ⁻¹(1−1/ke))` |

## Known Issues / Design Decisions
- **PolyMedian/PolyMLE NaN rates**: High when |ζ̂_k − ζ̂_{k−1}| is small (large k, high ρ). Expected behaviour per paper. nanmean/nanstd used for aggregation.
- **EBayes rank correlation** (Exp 4: τ≈0.26) lower than paper's expected ≈0.42. Simplified JS implementation of Johnstone–Silverman vs. full R `EbayesThresh` package.
- **Unit convention**: All estimators input/output in **annual** units (yr^{−1/2}); internal computation in daily units for numerical stability.
- **Daily Sharpe** = `mean(X[:,j]) / std(X[:,j], ddof=1)`; annualised by `× √252`.

## Validated Results (M=200, full corners for Exp 4–5)
| Setting | James-Stein bias | GMLEB bias | Notes |
|---------|-----------------|------------|-------|
| n=1008, k=100, σ_ζ=0.5, Gaussian | −0.030 | +0.079 | Matches paper |
| Exp 4 overall Kendall τ (JS) | 0.573 | 0.497 | Matches paper ≈0.57, 0.49 |
| Exp 5 overall Kendall τ (JS) | 0.345 | 0.326 | Matches paper ≈0.34, 0.32 |

## Dependencies
See `requirements.txt`. Key: numpy, scipy, pandas, matplotlib, seaborn, joblib, tqdm.
No real market data; all synthetic via NumPy RNG.
