# OPERATIONALIZATION REPORT — ML Pipeline & Live Paper Trading

## Status
Completed across all requested phases (1-5), including strict pre-flight warm-up gate.

## Phase 1 — Pre-flight data fetch & ML warm-up
Implemented:
1. `C:\Tugas Akhir\research\preflight_warmup.py`
   - Fetches freshest market + macro data before live start.
   - Synchronously triggers training warm-up (`collect` + `refit`).
   - Blocks until training finishes.
   - Verifies latest artifacts exist and are newly updated:
     - `lstm_vol_model.h5`
     - `lpa_gmm_model.pkl`
2. `trading_engine.py` live startup gate:
   - `--run-preflight`
   - preflight config flags for master/prod/log/report paths and warm-up epochs

## Phase 2 — Broker integration & paper trading
Updated:
- `C:\Tugas Akhir\research\trading_engine.py`
- `C:\Tugas Akhir\research\notifications.py`

Implemented:
1. Live broker integration (Backtrader IBStore):
   - `run_live_ib(...)`
   - CLI flags: `--mode live --broker ib --paper --symbol`
2. Paper mode support:
   - `--paper` routes to IB paper port (`IB_PAPER_PORT`, default 7497)
3. Live feature enrichment:
   - `LiveFeatureService` fetches latest `finbert_score`, `pred_volatility`, `lpa_profile_id`
   - Sources: `LIVE_FEATURES_API` or `LIVE_FEATURES_PATH`
4. Graceful reconnect:
   - exponential backoff loop on live disconnect

## Phase 3 — MLOps scheduler daemon
Updated:
- `C:\Tugas Akhir\research\train_pipeline.py`

Implemented:
1. Task scheduling (`schedule` package):
   - Daily data collection + FinBERT scoring at `DAILY_COLLECT_TIME` (default 16:10)
   - Weekly model refit on Saturday at `WEEKLY_REFIT_TIME` (default 02:00)
2. Operational modes:
   - `--mode collect | refit | once | scheduler`
3. Error handling:
   - task-level try/except + JSONL logs
   - on failure, previous model artifacts remain active (no crash path)

## Phase 4 — Live monitoring & alerts
Implemented webhook alerts in strategy:
1. LPA regime change detected
2. Parent order initiation (includes stop distance / child scheduling context)
3. Order fill completion
4. Stop-loss execution
5. Live reconnect events

## Phase 5 — Dockerization and shared model volume
Added:
- `C:\Tugas Akhir\research\Dockerfile` (multi-stage with TA-Lib + ML stack)
- `C:\Tugas Akhir\docker-compose.yml`
- `C:\Tugas Akhir\.env.example`

Compose services:
1. `mlops_service`:
   - runs scheduled `train_pipeline.py --mode scheduler`
2. `trading_service`:
   - runs `trading_engine.py --mode live --broker ib --paper --run-preflight ...`

Shared volume architecture:
- `./production/models:/app/models`
- both services read/write the same model directory

## Validation performed
1. Python compile checks passed:
   - `train_pipeline.py`, `trading_engine.py`, `notifications.py`
2. MLOps quick checks passed:
   - `train_pipeline.py --mode collect`
   - `train_pipeline.py --mode refit --epochs 4`
3. Trading engine smoke check passed:
   - `trading_engine.py --mode backtest --data-path ...`

## Operational run commands
1. Scheduler daemon:
   - `python C:\Tugas Akhir\research\train_pipeline.py --mode scheduler`
2. Live paper trading:
   - `python C:\Tugas Akhir\research\trading_engine.py --mode live --broker ib --paper --symbol BBRI-STK-SMART-USD --run-preflight`
3. Docker:
   - `docker compose up --build -d`
