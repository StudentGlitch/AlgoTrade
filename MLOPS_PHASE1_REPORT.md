# MLOPS Phase 1 Report — Continuous Training Pipeline

## Outcome
`train_pipeline.py` was implemented and executed successfully in **run-once** mode.

## Implemented components (decoupled OOP)
1. **Collectors**
   - `MirofishCollector` (API-ready via environment variables)
   - `MacroCollectorYF` (VIX, USD/IDR fallback collector)
   - `TextNewsCollectorYF` (ticker-level news text counts)
2. **Data Repository**
   - `DataRepository` for master dataset load/merge/save
3. **Model Refitters**
   - `LPARefitter` (8-regime GMM refit)
   - `LSTMRefitter` (rolling volatility fine-tuning)
4. **Pipeline Orchestrator**
   - `ContinuousTrainingPipeline`
5. **Scheduler**
   - `TrainingScheduler` with `run_once()` and daemon mode (`interval_hours`)

## Leakage controls
- Strict date cutoff for train/test split in retraining cycle.
- LSTM uses chronological sequences per company and no random shuffle.
- Pipeline never uses future rows to fit current models.

## Validation run result (executed)
- Command:
  - `python C:\Tugas Akhir\research\train_pipeline.py --mode once --epochs 6 --lookback 10`
- Output models:
  - `C:\Tugas Akhir\production\models\lpa_gmm_model.pkl`
  - `C:\Tugas Akhir\production\models\lstm_vol_model.h5`
  - `C:\Tugas Akhir\production\models\lstm_feature_scaler.pkl`
- Log stream:
  - `C:\Tugas Akhir\production\phase1_train_log.jsonl`

Run metrics:
- Train cutoff: `2026-03-01`
- LPA train rows: `6420`
- LSTM train sequences: `6220`
- LSTM test sequences: `540`
- LSTM test RMSE: `0.02464`
- Collected this cycle:
  - Macro rows: `33`
  - News rows: `23`
  - Mirofish rows: `0` (API not configured)

## Deliverable file
- Pipeline script: `C:\Tugas Akhir\research\train_pipeline.py`
