# Phase 3 — Advanced Predictive Modeling

## Forecasting setup
- Objective: use social-media data at time *t* to predict market movement at *t+1*.
- Regression target: `target_return_t1` (next-day return).
- Classification target: `target_up_t1` (1 if next-day return > 0 else 0).
- Features: social metrics at t and lagged social metrics (t-1, t-2).
- Validation: **TimeSeriesSplit (5 folds)** to prevent temporal leakage.

## Performance (out-of-sample)
- Random Forest RMSE (mean): **0.009359**
- XGBoost Accuracy (mean): **0.520661**
- XGBoost F1-score (mean): **0.532175**

### Fold-by-fold metrics
|   fold |   train_n |   test_n |     rmse |   accuracy |       f1 |
|-------:|----------:|---------:|---------:|-----------:|---------:|
|      1 |       124 |      121 | 0.01102  |   0.553719 | 0.564516 |
|      2 |       245 |      121 | 0.012261 |   0.454545 | 0.47619  |
|      3 |       366 |      121 | 0.008104 |   0.528926 | 0.606897 |
|      4 |       487 |      121 | 0.008599 |   0.545455 | 0.513274 |
|      5 |       608 |      121 | 0.00681  |   0.520661 | 0.5      |

## Feature importance
- RF plot: `C:\Tugas Akhir\research\feature_importance_rf.png`
- XGBoost plot: `C:\Tugas Akhir\research\feature_importance_xgb.png`

### Top RF features
| feature                 |   importance |
|:------------------------|-------------:|
| comments_fb_lag2        |     0.066856 |
| likes_fb_lag1           |     0.064771 |
| likes_fb_lag2           |     0.062627 |
| composite_index_fb_lag2 |     0.048787 |
| shares_fb_lag2          |     0.042213 |
| views_yt_lag2           |     0.04021  |
| composite_index_fb      |     0.040081 |
| likes_yt                |     0.039024 |
| views_fb                |     0.038662 |
| views_yt_lag1           |     0.038291 |

### Top XGBoost features
| feature                 |   importance |
|:------------------------|-------------:|
| composite_index_fb_lag2 |     0.048096 |
| likes_yt_lag2           |     0.045549 |
| comments_fb_lag2        |     0.044942 |
| views_yt                |     0.044533 |
| likes_fb_lag2           |     0.04263  |
| likes_yt                |     0.041978 |
| composite_index_fb      |     0.038863 |
| likes_fb_lag1           |     0.038686 |
| pca_index_yt            |     0.038639 |
| comments_yt_lag2        |     0.038126 |

## Leakage control note
- No random K-fold was used. All splits preserve chronological order.