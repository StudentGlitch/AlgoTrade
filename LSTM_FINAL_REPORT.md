# LSTM_FINAL_REPORT — Advanced Deep Learning for Volatility & Social Media

## Objective
Shift from return-direction modeling to volatility/activity modeling, handle YouTube zero-inflation explicitly, and evaluate whether sequence models (LSTM) improve prediction of market activity.

## Data
- Source: `C:\Tugas Akhir\Full Data.csv`
- Preprocessed output: `C:\Tugas Akhir\research\phase4_preprocessed.csv`
- Panel scope: 20 companies, 2021–2023 period

## Phase 0 — Preprocessing summary
See: `C:\Tugas Akhir\PREPROCESSING_AUDIT.md`

Implemented:
1. Target shift to volatility proxy:
   - `absolute_return = abs(return)`
2. Zero-inflation handling for YouTube:
   - Source: `pca_index_yt`
   - Zero share: **86.02%**
   - Top-5% non-zero threshold: **25.1425**
   - Event variable: `yt_spike`
3. Feature engineering:
   - `fb_ma3`, `fb_ma5`
   - `fb_pct_change`
   - `vol_lag1`
   - `fb_vol_interaction = composite_index_fb * vol_lag1`

## Phase 1 — Event study around YouTube spikes
See: `C:\Tugas Akhir\EVENT_STUDY_RESULTS.md`
Plot: `C:\Tugas Akhir\research\event_study_cav_plot.png`

Design:
- Event: `yt_spike == 1`
- Window: **[-2, +5]**
- Metric: **CAV** based on abnormal volatility:
  - `abnormal_volatility = absolute_return - company_mean(absolute_return)`

Findings:
- Valid event windows: **70**
- Average CAV path is small and oscillatory:
  - Starts slightly positive pre-event
  - Turns near zero by tau=+5 (`avg_cav` ≈ -0.000068)
- Interpretation: YouTube spikes are associated with localized volatility perturbations, but not a large persistent cumulative volatility drift in this sample.

## Phase 2 — LSTM volatility prediction
See: `C:\Tugas Akhir\research\PHASE2_LSTM_RESULTS.md`
Plot: `C:\Tugas Akhir\research\lstm_actual_vs_predicted.png`
Script: `C:\Tugas Akhir\research\phase4_lstm_model.py`

Model setup:
- Target: `absolute_return`
- Lookback: **7 timesteps**
- Features include engineered social and market variables (`yt_spike`, FB rolling stats, lagged volume interaction, etc.)
- Sequence generation is strictly per company (no boundary crossing)
- Chronological split:
  - Train: <= 2022
  - Test: 2023
  - Validation: tail of training period
- Training with `shuffle=False` and early stopping

Performance:
- **LSTM RMSE:** 0.012825
- **LSTM MAE:** 0.009270
- **Baseline RMSE (yesterday volatility):** 0.016507
- **Baseline MAE (yesterday volatility):** 0.010965

Conclusion:
- LSTM outperforms a naive persistence baseline on both RMSE and MAE, indicating incremental predictive value from sequential social+market features for volatility proxy forecasting.

## Verification checklist
- [x] Phase 0 transforms target to `absolute_return` and bins zero-inflated YT data using top-5% non-zero threshold.
- [x] Phase 1 computes event-window CAV specifically around `yt_spike` days.
- [x] Phase 2 sequences are created per company without mixing histories.
- [x] Phase 2 train/test split is chronological (no random shuffling / no temporal leakage).

## Deliverables produced
1. `C:\Tugas Akhir\PREPROCESSING_AUDIT.md`
2. `C:\Tugas Akhir\EVENT_STUDY_RESULTS.md`
3. `C:\Tugas Akhir\research\phase4_event_study.py`
4. `C:\Tugas Akhir\research\phase4_lstm_model.py`
5. `C:\Tugas Akhir\research\PHASE2_LSTM_RESULTS.md`
6. `C:\Tugas Akhir\research\event_study_cav_plot.png`
7. `C:\Tugas Akhir\research\lstm_actual_vs_predicted.png`
8. `C:\Tugas Akhir\LSTM_FINAL_REPORT.md`
