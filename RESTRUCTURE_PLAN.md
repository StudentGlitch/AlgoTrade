# RESTRUCTURE PLAN (PHASE 0)

## Scope
Restructure repository for two isolated services (`mlops_service`, `trading_service`) with shared Docker-mounted assets under `shared/`.

## Current Structure (Audited)

```text
AlgoTrade/
├── docker-compose.yml
├── requirements.txt
├── production/
│   └── models/
│       └── .gitkeep
└── research/
    ├── Dockerfile
    ├── train_pipeline.py
    ├── trading_engine.py
    ├── preflight_warmup.py
    ├── notifications.py
    ├── gather_openbb_extended_data.py
    ├── deeper_research_with_skills.py
    ├── phase1_panel_fe.py
    ├── phase2_var_granger.py
    ├── phase3_ml_prediction.py
    ├── phase4_preprocessing.py
    ├── phase4_lstm_model.py
    ├── phase4_event_study.py
    ├── phase5_data_repair_rectangularize.py
    ├── phase6_finbert_lpa.py
    ├── phase7_lstm_volatility.py
    ├── PHASE1_RESULTS.md
    ├── PHASE2_RESULTS.md
    ├── PHASE2_LSTM_RESULTS.md
    └── PHASE3_RESULTS.md
```

## Proposed Target Structure (After Migration)

```text
AlgoTrade/
├── docker-compose.yml
├── services/
│   ├── mlops/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── train_pipeline.py
│   │   ├── preflight_warmup.py
│   │   ├── notifications.py
│   │   ├── gather_openbb_extended_data.py
│   │   ├── deeper_research_with_skills.py
│   │   ├── phase1_panel_fe.py
│   │   ├── phase2_var_granger.py
│   │   ├── phase3_ml_prediction.py
│   │   ├── phase4_preprocessing.py
│   │   ├── phase4_lstm_model.py
│   │   ├── phase4_event_study.py
│   │   ├── phase5_data_repair_rectangularize.py
│   │   ├── phase6_finbert_lpa.py
│   │   └── phase7_lstm_volatility.py
│   └── trading/
│       ├── requirements.txt
│       ├── trading_engine.py
│       ├── preflight_warmup.py
│       └── notifications.py
└── shared/
    ├── models/
    │   └── (all .pkl/.h5 artifacts)
    ├── data/
    │   └── (master/enriched CSV and shared datasets)
    └── logs/
        └── (optional runtime logs, e.g., phase1_train_log.jsonl)
```

## File Mapping Plan

- Move model training and data prep code from `research/` to `services/mlops/`.
- Move execution engine code from `research/trading_engine.py` to `services/trading/trading_engine.py`.
- Keep shared runtime helpers in both services initially (`preflight_warmup.py`, `notifications.py`) to avoid import breakage during first migration pass.
- Move model artifacts from `production/models/` to `shared/models/` (preserve all files; no deletions).
- Move shared datasets (e.g., `phase6_lpa_enriched.csv` when present) to `shared/data/`.
- Replace root `requirements.txt` with service-specific dependencies:
  - `services/mlops/requirements.txt` (TensorFlow, scikit-learn, transformers, etc.)
  - `services/trading/requirements.txt` (Backtrader, broker integrations, runtime deps)

## Path and Import Refactor Plan (Next Phase)

- Refactor hardcoded Windows paths in `train_pipeline.py` and `trading_engine.py` to dynamic `pathlib`/`os.path`.
- Update defaults to point to `../../shared/models` and `../../shared/data` from each service script location.
- Update Docker volume mounts to share `./shared/models:/app/shared/models` across both services.
- Preserve behavior by keeping CLI overrides and env var overrides as first-class path inputs.

## Risk Controls

- Zero data loss: move files only; no deletion of datasets/model artifacts.
- Backward compatibility: keep CLI flags and env vars, adjust defaults.
- Migration safety: perform path fixups immediately after moves before service execution.

## Phase Gate

Phase 0 complete. **STOP here for approval** before starting Phase 1 (directory creation + file moves).
