# PREPROCESSING_AUDIT

## 1) Input and target shift
- Source dataset: `C:\Tugas Akhir\Full Data.csv`
- Rows after basic cleaning: **10154**
- New target variable: `absolute_return = abs(return)` (volatility proxy)
- Alternative targets retained: `vol`, `turnover`

## 2) Zero-inflation handling (YouTube)
- YT source used for spike binning: `pca_index_yt`
- Zero share in `pca_index_yt`: **86.02%**
- Top-5% (non-zero) threshold: **25.142500**
- Event variable: `yt_spike = 1 if YT source >= threshold and >0, else 0`
- Spike event rate: **0.70%**

## 3) Engineered features
- `fb_ma3`: 3-day rolling mean of `composite_index_fb` (within company)
- `fb_ma5`: 5-day rolling mean of `composite_index_fb` (within company)
- `fb_pct_change`: daily pct change of `composite_index_fb` (within company)
- `vol_lag1`: lagged `vol` by 1 day (within company)
- `fb_vol_interaction = composite_index_fb * vol_lag1`

## 4) Basic statistics of new variables
|                    |   count |        mean |          std |   min |          25% |          50% |         75% |            max |
|:-------------------|--------:|------------:|-------------:|------:|-------------:|-------------:|------------:|---------------:|
| absolute_return    |   10154 | 0.013847    |  0.014272    |     0 |  0.004115    |  0.00995     | 0.018868    |    0.25        |
| yt_spike           |   10154 | 0.006992    |  0.083331    |     0 |  0           |  0           | 0           |    1           |
| composite_index_fb |   10154 | 3.94639     |  7.57844     |     0 |  0.36        |  1.405       | 4.12        |  100           |
| fb_ma3             |   10154 | 3.95095     |  5.04632     |     0 |  0.566667    |  2.31        | 5.39        |   57.94        |
| fb_ma5             |   10154 | 3.94409     |  4.39641     |     0 |  0.7385      |  2.802       | 5.392       |   57.94        |
| fb_pct_change      |    9871 | 4.24995     | 44.7169      |    -1 | -0.614531    | -0.019048    | 1.29254     | 2504           |
| vol_lag1           |   10134 | 5.32792e+07 |  7.07029e+07 | 68400 |  1.22509e+07 |  3.55284e+07 | 7.14928e+07 |    1.98321e+09 |
| fb_vol_interaction |   10134 | 2.35545e+08 |  9.85805e+08 |     0 |  7.83694e+06 |  4.19397e+07 | 1.6127e+08  |    5.18468e+10 |

## 5) Output artifacts
- Preprocessed dataset: `C:\Tugas Akhir\research\phase4_preprocessed.csv`
- Audit report: `C:\Tugas Akhir\PREPROCESSING_AUDIT.md`

## 6) Phase 0 stop point
- Phase 0 preprocessing completed.
- **STOP POINT:** Awaiting approval before Phase 1 (Event Study around `yt_spike`).