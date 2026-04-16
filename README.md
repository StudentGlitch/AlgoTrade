# Quant Trading ML Ops (Paper Trading Ready)

This repository contains two isolated microservices and a shared volume:

```
services/
├── mlops/            ← ContinuousTrainingPipeline, data collectors, FinBERT re-scoring
│   ├── train_pipeline.py
│   ├── preflight_warmup.py
│   ├── notifications.py
│   ├── phase1_panel_fe.py … phase7_lstm_volatility.py
│   ├── Dockerfile
│   └── requirements.txt
└── trading/          ← RegimeExecutionStrategy, live IB broker hooks
    ├── trading_engine.py
    ├── preflight_warmup.py  (copy — needed for --run-preflight)
    ├── train_pipeline.py    (copy — needed by preflight_warmup)
    ├── notifications.py     (copy)
    ├── Dockerfile
    └── requirements.txt
shared/               ← Docker-mounted volume shared by both services
├── models/           ← .pkl and .h5 model artefacts written by mlops, read by trading
├── data/             ← master dataset CSV and intermediate data files
└── logs/             ← JSONL training logs, PREFLIGHT_WARMUP_REPORT.md
```

## 1) Setup

1. Copy `.env.example` to `.env`
2. Fill broker + webhook credentials in `.env`
3. (Optional) place seed CSVs in `shared/data/` and starter models in `shared/models/`

## 2) Run on VPS (Docker)

```bash
docker compose up --build -d
docker compose logs -f trading_service
docker compose logs -f mlops_service
```

## 3) Run locally (Python)

```bash
pip install -r services/mlops/requirements.txt
python services/mlops/train_pipeline.py --mode scheduler

pip install -r services/trading/requirements.txt
python services/trading/trading_engine.py --mode live --paper --broker ib --run-preflight
```

## 4) Notes

- `--paper` keeps execution in paper-trading mode.
- Pre-flight (`--run-preflight`) blocks live startup until model refresh succeeds.
- Model sharing is done via `./shared/models:/app/shared/models` (Docker volume).
- Data sharing is done via `./shared/data:/app/shared/data` (Docker volume).

## 5) Verify IBStore in Docker Compose (reproducible trading_service build)

```bash
# Rebuild trading service from pinned lockfile/dependency set
docker compose build --no-cache trading_service

# Start or restart the service
docker compose up -d trading_service

# Verify IBStore exists in the running container
docker compose exec trading_service python -c "import backtrader as bt; print('IBStore available:', hasattr(bt.stores, 'IBStore'))"

# Confirm logs no longer include the old startup error
docker compose logs trading_service | grep -F "IBStore is unavailable in this Backtrader installation." || echo "No IBStore startup error found"
```

- Validate in `--paper` mode first, then remove `--paper` for live execution only after paper mode is stable.
