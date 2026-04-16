# LSTM_FINAL_RESULTS

## 1) Modeling target and features
- Target: `abs_return` (volatility magnitude, not directional return)
- Feature blocks: LPA profile one-hot, FinBERT score, technicals, macro factors (JKSE/VIX/USDIDR/Oil/Gold), interaction terms.
- Lookback window: **10**
- Sequence generation is per-company; no cross-company sequence mixing.

## 2) Temporal split and leakage control
- Train period: <= 2025-06-30
- Test period: > 2025-06-30
- Validation: tail split of training data
- No random shuffling (`shuffle=False`).

## 3) Performance
- LSTM RMSE: **0.018133**
- LSTM MAE: **0.012292**
- LSTM R² (variance explained): **0.100490**

- Persistence RMSE: **0.023367**
- Persistence MAE: **0.015412**
- Persistence R²: **-0.493798**

## 4) Benchmark comparison
- Goal: beat persistence RMSE baseline.
- RMSE improvement (baseline - LSTM): **0.005234**

## 5) Plot
- Actual vs predicted sample plot (BBRI): `C:\Tugas Akhir\research\phase7_lstm_actual_vs_pred.png`

## 6) Training details
- Epochs run (early stopping): **46**
- Number of test sequences: **3840**