# PREFLIGHT_WARMUP_REPORT

## Summary
- Pre-flight fetch completed.
- Synchronous warm-up training completed.
- Required model artifacts validated in production directory.

## Refresh stats
```json
{
  "symbols_attempted": 20,
  "symbols_failed": [],
  "refreshed_rows": 4740,
  "rows_added_net": 0,
  "start_fetch": "2025-04-17",
  "end_fetch": "2026-04-17"
}
```

## Warm-up training stats
```json
{
  "collect": {
    "macro_rows_collected": 33,
    "news_text_rows_collected": 23,
    "news_daily_rows": 23,
    "mirofish_rows_collected": 0
  },
  "refit": {
    "max_date": "2026-04-16",
    "train_cutoff": "2026-03-02",
    "lpa_model_path": "C:\\Tugas Akhir\\production\\models\\lpa_gmm_model.pkl",
    "lstm_model_path": "C:\\Tugas Akhir\\production\\models\\lstm_vol_model.h5",
    "lstm_scaler_path": "C:\\Tugas Akhir\\production\\models\\lstm_feature_scaler.pkl",
    "lpa_stats": {
      "lpa_features": [
        "finbert_score",
        "news_items_day",
        "news_count",
        "volume_z20",
        "volatility_20d",
        "abs_return",
        "vix_close_ret",
        "usd_idr_close_ret"
      ],
      "lpa_train_rows": 6420
    },
    "lstm_stats": {
      "train_sequences": 6220,
      "test_sequences": 540,
      "epochs_run": 6,
      "final_val_loss": 0.0004362364416010678,
      "test_rmse": 0.028959281742572784
    }
  },
  "validated_models": {
    "lstm": "C:\\Tugas Akhir\\production\\models\\lstm_vol_model.h5",
    "gmm": "C:\\Tugas Akhir\\production\\models\\lpa_gmm_model.pkl"
  }
}
```

## Status
`ready_for_live_phase2`