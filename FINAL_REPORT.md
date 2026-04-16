# FINAL_REPORT — Social Media vs Stock Prices

## Research objective
Evaluate whether social media engagement metrics (Facebook and YouTube) are associated with Indonesian stock performance, both for inference (econometrics) and prediction (machine learning).

## Data and scope
- Dataset: `C:\Tugas Akhir\Full Data.csv`
- Coverage: 10,154 rows, 20 companies, 2021-01-04 to 2023-12-29
- Core variables:
  - Financial: `return`, `residual`, `vol`, `turnover`
  - Social: `composite_index_fb`, `pca_index_yt` (+ underlying social metrics)

## Phase 0 summary (audit/EDA)
Source: `C:\Tugas Akhir\DATA_AUDIT.md`

Key findings:
- Strong zero-inflation in several YouTube indicators:
  - `pca_index_yt` 86.02% zeros
  - `views_yt` 85.39% zeros
  - `comments_yt` 90.68% zeros
- FB/YT multicollinearity is low:
  - Corr(`composite_index_fb`, `pca_index_yt`) = 0.025
- ADF stationarity checks (aggregated daily means) indicate stationarity for:
  - `return`, `residual`, `composite_index_fb`, `pca_index_yt` (all p < 0.05)

## Phase 1 summary (Panel Fixed Effects)
Source: `C:\Tugas Akhir\research\PHASE1_RESULTS.md`

Model:
- Panel index: `id_company` x `tanggal_stata`
- Entity fixed effects (company FE), clustered SE by entity
- Regressors: `composite_index_fb`, `pca_index_yt`
- Control: `vol`

Primary outcome (`residual` as DV):
- `composite_index_fb`: coef -0.000019, p = 0.516 (not significant)
- `pca_index_yt`: coef -0.000009, p = 0.775 (not significant)
- Within R² = 0.000238

Robustness (`return` as DV):
- `composite_index_fb`: p = 0.408 (not significant)
- `pca_index_yt`: p = 0.157 (not significant)
- `vol` significant; social indices remain not significant

Conclusion:
- No statistically significant **contemporaneous** effect of the two social composite indices on returns/abnormal returns in FE specification.

## Phase 2 summary (VAR + Granger + Event Study)
Source: `C:\Tugas Akhir\research\PHASE2_RESULTS.md`

Setup:
- Daily aggregated panel mean series
- VAR variables: `return`, `composite_index_fb`, `pca_index_yt`
- Optimal lag selected by BIC: 1

Granger causality:
- `composite_index_fb` -> `return`: p = 0.232956 (no evidence)
- `return` -> `composite_index_fb`: p = 0.114921 (no evidence)

Event study (top 1% spikes in `views_fb`/`views_yt`, CAR[-1,+1] using `residual`):
- 16 event dates
- Mean CAR = -0.000051
- Median CAR = 0.002341

Conclusion:
- No strong directional predictability between FB composite index and return under this lag structure.
- Spike events show mixed CAR signs, with near-zero average effect.

## Phase 3 summary (Predictive ML)
Source: `C:\Tugas Akhir\research\PHASE3_RESULTS.md`

Task:
- Forecast next-day return and direction using social features at t with lagged features (t-1, t-2)
- Strict `TimeSeriesSplit` (5 folds) to avoid leakage

Models and performance:
- Random Forest Regressor (target: next-day return)
  - Mean RMSE = 0.009359
- XGBoost Classifier (target: next-day up/down)
  - Mean Accuracy = 0.520661
  - Mean F1 = 0.532175

Feature importance highlights:
- Repeatedly important: lagged Facebook activity terms and lagged composite FB index
- YT variables appear among top predictors in some folds, but signal remains modest

Conclusion:
- Predictive performance is only modestly above random baseline for direction classification, suggesting weak but non-zero exploitable signal from social data in this setup.

## Verification checklist
- [x] Phase 0 checks stationarity before time-series modeling (ADF, autolag=AIC).
- [x] Phase 1 uses entity Fixed Effects for company-specific baseline control.
- [x] Phase 2 Granger uses lag length selected via information criteria (BIC chosen from VAR order selection).
- [x] Phase 3 uses strictly time-ordered splits (no random K-fold), preventing leakage.

## Overall conclusion
Across FE econometrics, VAR/Granger causality, and time-series ML:
- Composite social indices do **not** show robust contemporaneous or Granger-causal effects on returns in this sample.
- Social metrics contribute limited predictive information; models achieve only moderate directional accuracy.
- Data sparsity (especially YT zero inflation) is a key limitation and likely attenuates signal strength.
