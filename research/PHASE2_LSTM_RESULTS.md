# PHASE 2 — LSTM Results

## Data and sequence setup
- Target: `absolute_return` (continuous volatility proxy)
- Lookback window: **7 days**
- Sequences built strictly within each `id_company` (no cross-company sequence mixing).
- Chronological split: train(<=2022), test(2023). Validation is the tail of training period.
- Training uses `shuffle=False` to preserve temporal ordering.

## Evaluation
- LSTM RMSE: **0.012825**
- LSTM MAE: **0.009270**
- Baseline RMSE (yesterday volatility): **0.016507**
- Baseline MAE (yesterday volatility): **0.010965**

## Plot
- Sample company plot (BBRI): `C:\Tugas Akhir\research\lstm_actual_vs_predicted.png`

## Notes
- Epochs run (with early stopping): **30**
- Lower RMSE/MAE indicates better predictive performance.