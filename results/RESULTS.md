# RESULTS.md – Monte Carlo Post-Selection Sharpe Ratio Estimators

## Overview

Five Monte Carlo experiments measuring **bias** and **RMSE** of nine
post-selection estimators for the true Sharpe ratio of the best-ranked
strategy, under independent and equicorrelated return distributions.

### Estimators

| Code | Name | Description |
|------|------|-------------|
| E1 | Biased | Raw sample Sharpe of best strategy |
| E2 | Grand Mean | Average sample Sharpe across all strategies |
| E3 | James-Stein | JS shrinkage toward grand mean (daily units) |
| E4 | GMLEB | Jiang-Zhang NPMLE via EM posterior means |
| E5 | SURE | Soft-threshold with SURE-optimal threshold |
| E6 | EBayes | Johnstone-Silverman sparse mixture EB thresholding |
| E7 | Polyhedral Median | Truncated-normal conditional median |
| E8 | Polyhedral MLE | Score-equation MLE under polyhedral constraint |
| E9 | Expected Max | Biased − E[max of k Gaussians] correction |

---

## Experiment 1: Bias & RMSE vs σ_ζ (Random Layouts, Independent Returns)

- **Parameters**: σ_ζ ∈ {0.0,...,1.0}, k ∈ {10,100,1000}, n ∈ {504,1008,2016}, M=500
- **Layouts**: Gaussian, Uniform, Bimodal (drawn fresh each simulation)
- **Figures**: fig1–fig6 (Bias/RMSE grids, RMSE vs n, regret CDF)

### Key Findings

- **E1 (Biased)** shows the largest positive bias growing with σ_ζ and k.
- **E3 (James-Stein)** achieves near-zero bias and lowest RMSE for Gaussian layouts.
- **E2 (Grand Mean)** has near-zero bias at σ_ζ=0, increasingly negative as spread grows.
- **E9 (Expected Max)** shows negative bias, slightly over-correcting.
- **E4 (GMLEB)** is second-best overall with slightly positive bias for non-Gaussian layouts.
- **E6 (Fig 6 Regret CDF)**: James-Stein shows leftmost CDF (best regret) for Gaussian.

| Estimator | Bias (n=1008,k=100,σ_ζ=0.5,Gauss) | RMSE |
|-----------|--------------------------------------|------|
| Biased | +0.876 | 0.958 |
| GrandMean | -0.946 | 1.015 |
| JamesStein | -0.030 | 0.384 |
| GMLEB | +0.079 | 0.513 |
| SURE | +0.424 | 0.601 |
| EBayes | +0.441 | 0.593 |
| PolyMedian | +nan | nan |
| PolyMLE | +nan | nan |
| ExpMax | -0.389 | 0.549 |

---

## Experiment 2: Bias & RMSE – Fixed Layouts (n=1008, k=100)

- **Layouts**: all-good (ζ₀·1_k), one-good (+ζ₀, rest -ζ₀), two-good (+ζ₀, +ζ₀, rest -ζ₀)
- **Figures**: fig7 (Bias vs ζ₀), fig8 (RMSE vs ζ₀)

### Key Findings

- **all-good**: Biased ≈ +0.5 bias; James-Stein and Grand Mean near zero.
- **one-good**: Large-ζ₀ James-Stein over-shrinks (negative bias, shrinks toward grand mean ≈ -ζ₀).
- **two-good**: Pattern similar to one-good but less extreme.

---

## Experiment 3: Bias & RMSE vs Equicorrelation ρ (σ_ζ=0.5, n=1008, k=100)

- **ρ grid**: {0.05, 0.1, ..., 0.95}
- **Figures**: fig12 (Bias vs ρ), fig13 (RMSE vs ρ)

### Key Findings

- All estimators degrade as ρ increases (effective noise ≈ √((1+(k-1)ρ)/n)).
- Relative ordering preserved through moderate ρ (JS best, Biased worst).
- At ρ→1: all strategies nearly identical → all estimates converge, JS shrinks to grand mean.

---

## Experiment 4: Ranking Study – Independent Returns (60,000 simulations)

- **500 corners** (σ_ζ, k) × 5n × 3 layouts × 8 reps = 60,000 records
- **Tables**: Overall Kendall τ & Spearman ρ by layout / n / k / σ_ζ
- **Figures**: fig9 (by n), fig10 (by k), fig11 (by σ_ζ)

### Key Findings

**Table 1 – Overall Rank Correlations:**

| Estimator | Kendall τ | Spearman ρ |
|-----------|-----------|------------|
| JamesStein | 0.5729 | 0.7638 |
| GMLEB | 0.4973 | 0.6697 |
| ExpMax | 0.4516 | 0.6211 |
| SURE | 0.4497 | 0.6249 |
| EBayes | 0.2621 | 0.3718 |
| Biased | 0.1567 | 0.2243 |
| GrandMean | 0.0183 | 0.0271 |
| PolyMedian | nan | nan |
| PolyMLE | nan | nan |

---

## Experiment 5: Ranking Study – Equicorrelated Returns (96,000 simulations)

- **800 corners** (σ_ζ, k, ρ) × 5n × 3 layouts × 8 reps = 96,000 records
- **Tables**: Tables 3–5; **Figures**: fig14–fig17 (by n, k, σ_ζ, ρ)

### Key Findings

**Table 3 – Overall Rank Correlations (correlated returns):**

| Estimator | Kendall τ | Spearman ρ |
|-----------|-----------|------------|
| JamesStein | 0.3451 | 0.4888 |
| SURE | 0.3404 | 0.4872 |
| GMLEB | 0.3258 | 0.4607 |
| EBayes | 0.3176 | 0.4515 |
| ExpMax | 0.3171 | 0.4520 |
| Biased | 0.2096 | 0.3017 |
| GrandMean | 0.0022 | 0.0033 |
| PolyMedian | nan | nan |
| PolyMLE | nan | nan |

- All rank correlations lower than independent case (correlation inflates noise).
- James-Stein still leads; Biased degrades less than shrinkage at very high ρ.

---

## Runtime Summary

| Experiment | Runtime (s) |
|------------|------------|
| exp1_random_layouts | 379.0 |
| exp2_fixed_layouts | 18.5 |
| exp3_correlated | 20.5 |
| exp4_ranking_independent | 360.0 |
| exp5_ranking_correlated | 634.0 |

---

## Output Files

| File | Description |
|------|-------------|
| `results/exp1_results.csv` | Bias/RMSE per (layout, σ_ζ, k, n) – Exp 1 |
| `results/exp2_results.csv` | Bias/RMSE per (layout, ζ₀) – Exp 2 |
| `results/exp3_results.csv` | Bias/RMSE per (layout, ρ) – Exp 3 |
| `results/exp4_records.csv` | 60,000 simulation records – Exp 4 |
| `results/exp5_records.csv` | 96,000 simulation records – Exp 5 |
| `results/table1_overall.csv` | Table 1: overall rank corrs (Exp 4) |
| `results/table2_by_layout.csv` | Table 2: rank corrs by layout (Exp 4) |
| `results/table3_overall_corr.csv` | Table 3: overall rank corrs (Exp 5) |
| `results/table4_by_layout_corr.csv` | Table 4: rank corrs by layout (Exp 5) |
| `results/table5_by_rho.csv` | Table 5: rank corrs by ρ bin (Exp 5) |
| `results/fig*.png` | All figures (16 total) |

*Generated by `run_experiments.py`.*