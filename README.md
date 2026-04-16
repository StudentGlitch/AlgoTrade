# Quant Trading ML Ops (Paper Trading Ready)

This repository contains:
- `research/train_pipeline.py` (daily/weekly retraining scheduler)
- `research/trading_engine.py` (live/paper execution engine)
- `research/preflight_warmup.py` (pre-flight data refresh + warm-up training)
- `research/notifications.py` (Discord/Telegram/webhook alerts)
- Dockerized deployment with shared model volume (`docker-compose.yml`)

## 1) Setup

1. Copy `.env.example` to `.env`
2. Fill broker + webhook credentials in `.env`
3. (Optional) place starter models in `production/models/`

## 2) Run on VPS (Docker)

```bash
docker compose up --build -d
docker compose logs -f trading_service
docker compose logs -f mlops_service
```

## 3) Run locally (Python)

```bash
pip install -r requirements.txt
python research/train_pipeline.py --mode scheduler
python research/trading_engine.py --mode live --paper --broker ib --run-preflight
```

## 4) Notes

- `--paper` keeps execution in paper-trading mode.
- Pre-flight (`--run-preflight`) blocks live startup until model refresh succeeds.
- Model sharing is done via `./production/models:/app/models`.
