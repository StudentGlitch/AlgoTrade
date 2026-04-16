# RECTANGULARIZATION_AUDIT

## 1) Phase 0 objective
Repair and rectangularize sparse 2024–2026 market data for sequence-safe predictive modeling.

## 2) Source and output
- Input: `C:\Tugas Akhir\research\openbb_enriched_stock_data_2024_onward.csv`
- Output: `C:\Tugas Akhir\research\phase5_rectangularized_data.csv`
- Rows before: **10760**
- Rows after rectangularization: **10760**

## 3) Price repair with yfinance `repair=True`
- Cross-referenced per symbol and replaced missing/outlier OHLCV values when deviation from repaired feed was large.
- Repaired cell updates applied: **20**

## 4) Temporal imputation (EM-like)
- Method: IterativeImputer + BayesianRidge (EM-style iterative conditional expectation).
- Applied per company, with chronological interpolation seeding.
- Date grid forced to full company x trading-date rectangle.

## 5) Bayesian neutrality baseline for sparse social data
- For zero-news days (`news_count <= 0`), sentiment fields set to neutral baseline **0.0**.
- Prevents sparse zeros from creating unstable gradients in downstream sequence models.

## 6) Missingness comparison (before vs after)
|                     |   before_missing_pct |   after_missing_pct |   delta_pct_points |
|:--------------------|---------------------:|--------------------:|-------------------:|
| news_sentiment_mean |              55      |                   0 |           -55      |
| news_sentiment_std  |              55      |                   0 |           -55      |
| vix_close_ret       |               6.5056 |                   0 |            -6.5056 |
| gold_close_ret      |               5.7621 |                   0 |            -5.7621 |
| oil_close_ret       |               5.7621 |                   0 |            -5.7621 |
| jkse_close_ret      |               5.5762 |                   0 |            -5.5762 |
| mom_20              |               3.9033 |                   0 |            -3.9033 |
| volatility_20d      |               3.9033 |                   0 |            -3.9033 |
| sma_ratio_5_20      |               3.7175 |                   0 |            -3.7175 |
| sma_20              |               3.7175 |                   0 |            -3.7175 |
| volume_z20          |               3.5316 |                   0 |            -3.5316 |
| vix_close           |               2.974  |                   0 |            -2.974  |
| gold_close          |               2.6022 |                   0 |            -2.6022 |
| oil_close           |               2.6022 |                   0 |            -2.6022 |
| rsi_14              |               2.6022 |                   0 |            -2.6022 |
| volatility_5d       |               1.1152 |                   0 |            -1.1152 |
| mom_5               |               1.1152 |                   0 |            -1.1152 |
| sma_5               |               0.9294 |                   0 |            -0.9294 |
| abs_return          |               0.3717 |                   0 |            -0.3717 |
| usd_idr_close_ret   |               0.3717 |                   0 |            -0.3717 |
| return              |               0.3717 |                   0 |            -0.3717 |
| log_return          |               0.3717 |                   0 |            -0.3717 |
| usd_idr_close       |               0.1859 |                   0 |            -0.1859 |
| turnover_proxy      |               0.1859 |                   0 |            -0.1859 |
| jkse_close          |               0.1859 |                   0 |            -0.1859 |
| close               |               0.1859 |                   0 |            -0.1859 |
| range_pct           |               0.1859 |                   0 |            -0.1859 |
| symbol              |               0      |                   0 |             0      |
| company             |               0      |                   0 |             0      |
| date                |               0      |                   0 |             0      |
| volume              |               0      |                   0 |             0      |
| low                 |               0      |                   0 |             0      |
| high                |               0      |                   0 |             0      |
| open                |               0      |                   0 |             0      |
| news_count          |               0      |                   0 |             0      |

## 7) Integrity checks (moments)
| column         |      orig_mean |      rect_mean |       orig_std |       rect_std |
|:---------------|---------------:|---------------:|---------------:|---------------:|
| open           | 6042.31        | 6042.31        | 6737.52        | 6737.52        |
| high           | 6117.11        | 6117.11        | 6802.79        | 6802.79        |
| low            | 5960.95        | 5960.95        | 6667.3         | 6667.3         |
| close          | 6035.97        | 6035.97        | 6736.76        | 6736.76        |
| volume         |    6.07649e+07 |    6.07649e+07 |    8.70622e+07 |    8.70622e+07 |
| return         |    0.00024     |    0.000254    |    0.025468    |    0.025449    |
| abs_return     |    0.017021    |    0.017014    |    0.018945    |    0.018926    |
| volatility_20d |    0.022781    |    0.022616    |    0.011298    |    0.011674    |

## 8) Sequence integrity checks
- Missing `date` values after process: **0**
- Rectangular panel produced for all company-date combinations in the observed period.

## 9) Phase 0 stop point
- Phase 0 completed.
- **STOP POINT:** Awaiting approval before Phase 1 (FinBERT + LPA).