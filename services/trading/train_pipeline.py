"""
MLOps Continuous Training Pipeline
---------------------------------
NOTE: This file is a required copy for the trading_service container.
Both services run in separate Docker containers and cannot share Python modules
directly. This copy exists because preflight_warmup.py (used by trading_engine.py
via --run-preflight) imports ContinuousTrainingPipeline from here.
Authoritative source: services/mlops/train_pipeline.py
Operational modes:
1) collect  : daily data collection + FinBERT scoring append
2) refit    : weekly LPA/GMM + LSTM volatility retraining
3) once     : collect + refit in one run
4) scheduler: daemon with calendar schedule
"""

from __future__ import annotations

import argparse
import json
import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import schedule
import yfinance as yf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # transformers are optional at runtime
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
DEFAULT_MASTER_DATA = str(_SHARED / "data" / "phase6_lpa_enriched.csv")
DEFAULT_PROD_DIR = str(_SHARED / "models")
DEFAULT_LOG = str(_SHARED / "logs" / "phase1_train_log.jsonl")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_joblib_dump(obj, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, out_path)


def atomic_model_save(model: tf.keras.Model, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp.h5"
    model.save(tmp)
    os.replace(tmp, out_path)


@dataclass
class PipelineConfig:
    master_data_path: str = DEFAULT_MASTER_DATA
    prod_dir: str = DEFAULT_PROD_DIR
    log_path: str = DEFAULT_LOG
    lpa_components: int = 8
    lstm_lookback: int = 10
    retrain_window_days: int = 540
    holdout_days: int = 45
    lstm_epochs: int = 24
    lstm_batch_size: int = 64
    daily_time: str = "16:10"      # local time
    weekly_time: str = "02:00"     # Saturday
    finbert_model_name: str = "ProsusAI/finbert"


class JsonLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def write(self, event: str, payload: dict) -> None:
        ensure_dir(os.path.dirname(self.log_path))
        rec = {"ts": now_utc().isoformat(), "event": event, **payload}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class DataRepository:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Master dataset not found: {self.path}")
        df = pd.read_csv(self.path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def save(self, df: pd.DataFrame) -> None:
        ensure_dir(os.path.dirname(self.path))
        tmp = self.path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)


class MirofishClient:
    """
    Optional external collector. No hardcoded secrets.
    Requires:
      MIROFISH_API_URL
      MIROFISH_API_TOKEN
    """

    def __init__(self):
        self.base_url = os.getenv("MIROFISH_API_URL", "").strip()
        self.token = os.getenv("MIROFISH_API_TOKEN", "").strip()

    def enabled(self) -> bool:
        return bool(self.base_url and self.token)

    def collect_latest(self) -> pd.DataFrame:
        if not self.enabled():
            return pd.DataFrame()
        r = requests.get(
            f"{self.base_url.rstrip('/')}/collect/latest",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        return pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))


class DailyCollector:
    """
    Daily market-close collection:
    - macro (VIX, USD/IDR)
    - text/news (yfinance)
    - FinBERT daily score
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._finbert = None

    def _fetch_macro(self, lookback_days: int = 45) -> pd.DataFrame:
        end = now_utc().date() + timedelta(days=1)
        start = end - timedelta(days=lookback_days)
        vix = yf.download("^VIX", start=start.isoformat(), end=end.isoformat(), interval="1d", progress=False)
        fx = yf.download("USDIDR=X", start=start.isoformat(), end=end.isoformat(), interval="1d", progress=False)
        if vix.empty and fx.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=sorted(set(vix.index.tolist() + fx.index.tolist())))
        if not vix.empty:
            out["vix_close"] = vix["Close"]
        if not fx.empty:
            out["usd_idr_close"] = fx["Close"]
        out = out.sort_index()
        out["vix_close_ret"] = out["vix_close"].pct_change()
        out["usd_idr_close_ret"] = out["usd_idr_close"].pct_change()
        out = out.reset_index().rename(columns={"index": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        return out

    def _fetch_news_text(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for sym in symbols:
            company = sym.replace(".JK", "")
            try:
                items = yf.Ticker(sym).news or []
            except Exception:
                continue
            for item in items:
                c = item.get("content", {})
                pub = c.get("pubDate") or c.get("displayTime")
                if not pub:
                    continue
                try:
                    dt = pd.to_datetime(pub, utc=True).tz_convert(None).normalize()
                except Exception:
                    continue
                txt = ((c.get("title") or "") + ". " + (c.get("summary") or c.get("description") or "")).strip()
                if not txt:
                    continue
                rows.append({"company": company, "date": dt, "text": txt})
        return pd.DataFrame(rows)

    def _load_finbert(self):
        if self._finbert is not None:
            return self._finbert
        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            self._finbert = None
            return None
        tok = AutoTokenizer.from_pretrained(self.cfg.finbert_model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(self.cfg.finbert_model_name)
        mod.eval()
        self._finbert = (tok, mod)
        return self._finbert

    def _score_finbert(self, texts: list[str]) -> list[float]:
        model_ref = self._load_finbert()
        if model_ref is None or not texts:
            return [0.0 for _ in texts]
        tok, mod = model_ref
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
                probs = torch.softmax(mod(**enc).logits, dim=1).cpu().numpy()
                # FinBERT labels: positive, negative, neutral
                for p in probs:
                    scores.append(float(p[0] - p[1]))
        return scores

    def collect_daily(self, master_df: pd.DataFrame, mirofish_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = master_df.copy()
        symbols = sorted({f"{c}.JK" for c in df["company"].dropna().unique().tolist()})

        macro_df = self._fetch_macro()
        news_raw = self._fetch_news_text(symbols)

        if not news_raw.empty:
            news_raw["finbert_score"] = self._score_finbert(news_raw["text"].tolist())
            news_daily = (
                news_raw.groupby(["company", "date"], as_index=False)
                .agg(
                    news_items_day=("text", "count"),
                    finbert_score=("finbert_score", "mean"),
                )
            )
        else:
            news_daily = pd.DataFrame(columns=["company", "date", "news_items_day", "finbert_score"])

        # Merge Mirofish first if present
        if not mirofish_df.empty and {"company", "date"}.issubset(mirofish_df.columns):
            mf = mirofish_df.copy()
            mf["date"] = pd.to_datetime(mf["date"], errors="coerce")
            df = df.merge(mf, on=["company", "date"], how="left", suffixes=("", "_miro"))
            for c in [x for x in df.columns if x.endswith("_miro")]:
                base = c[:-5]
                if base in df.columns:
                    df[base] = df[base].combine_first(df[c])
                    df = df.drop(columns=[c])
                else:
                    df = df.rename(columns={c: base})

        # Macro merge
        if not macro_df.empty:
            df = df.merge(macro_df, on="date", how="left", suffixes=("", "_new"))
            for c in [x for x in macro_df.columns if x != "date"]:
                nc = f"{c}_new"
                if nc in df.columns:
                    df[c] = df[c].combine_first(df[nc]) if c in df.columns else df[nc]
                    df = df.drop(columns=[nc], errors="ignore")

        # News + FinBERT merge
        if not news_daily.empty:
            df = df.merge(news_daily, on=["company", "date"], how="left", suffixes=("", "_new"))
            for c in ["news_items_day", "finbert_score"]:
                nc = f"{c}_new"
                if nc in df.columns:
                    df[c] = df[c].combine_first(df[nc]) if c in df.columns else df[nc]
                    df = df.drop(columns=[nc], errors="ignore")

        # Neutral fallback for missing daily sentiment features
        if "finbert_score" in df.columns:
            df["finbert_score"] = pd.to_numeric(df["finbert_score"], errors="coerce").fillna(0.0)
        if "news_items_day" in df.columns:
            df["news_items_day"] = pd.to_numeric(df["news_items_day"], errors="coerce").fillna(0).astype(int)

        stats = {
            "macro_rows_collected": int(len(macro_df)),
            "news_text_rows_collected": int(len(news_raw)),
            "news_daily_rows": int(len(news_daily)),
            "mirofish_rows_collected": int(len(mirofish_df)),
        }
        return df.sort_values(["company", "date"]), stats


class LPARefitter:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42, n_init=5)

    def fit_transform(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
        features = [c for c in ["finbert_score", "news_items_day", "news_count", "volume_z20", "volatility_20d", "abs_return", "vix_close_ret", "usd_idr_close_ret"] if c in df.columns]
        work = df.copy()
        for c in features:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        imp = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=20)
        X_all = pd.DataFrame(imp.fit_transform(work[features]), columns=features, index=work.index)
        train_idx = work.index[work["date"] <= train_cutoff]
        X_train = self.scaler.fit_transform(X_all.loc[train_idx])
        self.gmm.fit(X_train)

        probs = self.gmm.predict_proba(self.scaler.transform(X_all))
        work["lpa_profile_id"] = self.gmm.predict(self.scaler.transform(X_all)) + 1
        work["lpa_profile_conf"] = probs.max(axis=1)
        return work, {"lpa_features": features, "lpa_train_rows": int(len(train_idx))}

    def save(self, prod_dir: str) -> str:
        out = os.path.join(prod_dir, "lpa_gmm_model.pkl")
        atomic_joblib_dump({"scaler": self.scaler, "gmm": self.gmm, "saved_at": now_utc().isoformat()}, out)
        return out


class LSTMRefitter:
    def __init__(self, lookback: int = 10, epochs: int = 24, batch_size: int = 64):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model: Optional[tf.keras.Model] = None

    def _build(self, n_features: int) -> tf.keras.Model:
        m = Sequential(
            [
                Input(shape=(self.lookback, n_features)),
                LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
                LSTM(32, dropout=0.2, recurrent_dropout=0.1),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return m

    def _to_seq(self, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        X, y, meta = [], [], []
        for company, g in df.groupby("company"):
            g = g.sort_values("date").reset_index(drop=True)
            vals = g[features].values
            tgt = g["abs_return"].values
            dates = g["date"].values
            for i in range(self.lookback, len(g)):
                X.append(vals[i - self.lookback : i, :])
                y.append(tgt[i])
                meta.append((company, pd.to_datetime(dates[i])))
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(meta, columns=["company", "date"])

    def fit(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> dict:
        features = [c for c in ["abs_return", "volatility_5d", "volatility_20d", "volume_z20", "range_pct", "finbert_score", "vix_close_ret", "usd_idr_close_ret", "lpa_profile_id"] if c in df.columns]
        work = df.copy()
        for c in features:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work[features] = work.groupby("company")[features].transform(lambda s: s.ffill().bfill())
        work = work.dropna(subset=features + ["abs_return"])

        X, y, meta = self._to_seq(work, features)
        train_mask = meta["date"] <= train_cutoff
        test_mask = meta["date"] > train_cutoff
        X_train_all, y_train_all = X[train_mask.values], y[train_mask.values]
        X_test, y_test = X[test_mask.values], y[test_mask.values]

        if len(X_train_all) < 200:
            raise RuntimeError("Insufficient sequence rows for LSTM refit.")

        split = int(len(X_train_all) * 0.85)
        X_train, X_val = X_train_all[:split], X_train_all[split:]
        y_train, y_val = y_train_all[:split], y_train_all[split:]

        nf = X.shape[2]
        self.scaler.fit(X_train.reshape(-1, nf))

        def scale(arr: np.ndarray) -> np.ndarray:
            return self.scaler.transform(arr.reshape(-1, nf)).reshape(arr.shape)

        X_train, X_val = scale(X_train), scale(X_val)
        X_test = scale(X_test) if len(X_test) else X_test

        self.model = self._build(nf)
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        hist = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            callbacks=[es],
            verbose=0,
        )

        out = {
            "train_sequences": int(len(X_train_all)),
            "test_sequences": int(len(X_test)),
            "epochs_run": int(len(hist.history["loss"])),
            "final_val_loss": float(hist.history["val_loss"][-1]),
        }
        if len(X_test):
            pred = self.model.predict(X_test, verbose=0).reshape(-1)
            out["test_rmse"] = float(np.sqrt(np.mean((pred - y_test) ** 2)))
        return out

    def save(self, prod_dir: str) -> tuple[str, str]:
        if self.model is None:
            raise RuntimeError("LSTM not trained.")
        model_path = os.path.join(prod_dir, "lstm_vol_model.h5")
        scaler_path = os.path.join(prod_dir, "lstm_feature_scaler.pkl")
        atomic_model_save(self.model, model_path)
        atomic_joblib_dump({"scaler": self.scaler, "saved_at": now_utc().isoformat()}, scaler_path)
        return model_path, scaler_path


class ContinuousTrainingPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.repo = DataRepository(cfg.master_data_path)
        self.logger = JsonLogger(cfg.log_path)
        self.collector = DailyCollector(cfg)
        self.mirofish = MirofishClient()

    def daily_collect_and_score(self) -> dict:
        df = self.repo.load()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company"]).sort_values(["company", "date"])

        mf = self.mirofish.collect_latest() if self.mirofish.enabled() else pd.DataFrame()
        out_df, stats = self.collector.collect_daily(df, mf)
        self.repo.save(out_df)
        self.logger.write("daily_collect_success", stats)
        return stats

    def weekly_refit(self) -> dict:
        df = self.repo.load()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company"]).sort_values(["company", "date"])

        max_date = df["date"].max()
        start_cut = max_date - pd.Timedelta(days=self.cfg.retrain_window_days)
        train_cut = max_date - pd.Timedelta(days=self.cfg.holdout_days)
        train_df = df[df["date"] >= start_cut].copy()

        # Refit LPA
        lpa = LPARefitter(self.cfg.lpa_components)
        train_df, lpa_stats = lpa.fit_transform(train_df, train_cutoff=train_cut)
        lpa_path = lpa.save(self.cfg.prod_dir)

        # Refit LSTM
        lstm = LSTMRefitter(self.cfg.lstm_lookback, self.cfg.lstm_epochs, self.cfg.lstm_batch_size)
        lstm_stats = lstm.fit(train_df, train_cutoff=train_cut)
        lstm_path, scaler_path = lstm.save(self.cfg.prod_dir)

        # Persist refreshed profile ids in master data
        keep_base = df.drop(columns=[c for c in ["lpa_profile_id", "lpa_profile_conf"] if c in df.columns], errors="ignore")
        upd = train_df[["company", "date", "lpa_profile_id", "lpa_profile_conf"]]
        merged = keep_base.merge(upd, on=["company", "date"], how="left")
        merged["lpa_profile_id"] = merged["lpa_profile_id"].fillna(1).astype(int)
        merged["lpa_profile_conf"] = merged["lpa_profile_conf"].fillna(0.0)
        self.repo.save(merged.sort_values(["company", "date"]))

        result = {
            "max_date": str(max_date.date()),
            "train_cutoff": str(train_cut.date()),
            "lpa_model_path": lpa_path,
            "lstm_model_path": lstm_path,
            "lstm_scaler_path": scaler_path,
            "lpa_stats": lpa_stats,
            "lstm_stats": lstm_stats,
        }
        self.logger.write("weekly_refit_success", result)
        return result

    def run_once(self) -> dict:
        stats_collect = self.daily_collect_and_score()
        stats_refit = self.weekly_refit()
        out = {"collect": stats_collect, "refit": stats_refit}
        self.logger.write("run_once_success", out)
        return out

    def scheduler_daemon(self) -> None:
        job_lock = threading.Lock()
        last_heartbeat = 0.0

        def guarded_run(job_name: str, fn):
            if not job_lock.acquire(blocking=False):
                self.logger.write("scheduler_job_skipped", {"job": job_name, "reason": "previous_job_still_running"})
                return
            started = time.time()
            try:
                fn()
                self.logger.write("scheduler_job_ok", {"job": job_name, "duration_sec": round(time.time() - started, 3)})
            except Exception as e:
                # graceful degradation: keep previous models and continue
                self.logger.write(f"{job_name}_error", {"error": str(e), "duration_sec": round(time.time() - started, 3)})
            finally:
                job_lock.release()

        def safe_daily():
            guarded_run("daily_collect", self.daily_collect_and_score)

        def safe_weekly():
            guarded_run("weekly_refit", self.weekly_refit)

        # Daily at market close
        schedule.every().day.at(self.cfg.daily_time).do(safe_daily)
        # Weekly Saturday 02:00
        schedule.every().saturday.at(self.cfg.weekly_time).do(safe_weekly)

        self.logger.write(
            "scheduler_started",
            {"daily_time": self.cfg.daily_time, "weekly_time": self.cfg.weekly_time},
        )

        while True:
            try:
                schedule.run_pending()
                now_ts = time.time()
                if now_ts - last_heartbeat >= 3600:
                    self.logger.write("scheduler_heartbeat", {"status": "alive"})
                    last_heartbeat = now_ts
            except Exception as e:
                # never crash daemon loop; keep previous models active
                self.logger.write("scheduler_loop_error", {"error": str(e)})
            time.sleep(10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operational MLOps training pipeline")
    p.add_argument("--master-data", default=DEFAULT_MASTER_DATA)
    p.add_argument("--prod-dir", default=DEFAULT_PROD_DIR)
    p.add_argument("--log-path", default=DEFAULT_LOG)
    p.add_argument("--mode", choices=["collect", "refit", "once", "scheduler"], default="once")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--daily-time", default="16:10")
    p.add_argument("--weekly-time", default="02:00")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        master_data_path=args.master_data,
        prod_dir=args.prod_dir,
        log_path=args.log_path,
        lstm_epochs=args.epochs,
        lstm_lookback=args.lookback,
        daily_time=args.daily_time,
        weekly_time=args.weekly_time,
    )
    pipe = ContinuousTrainingPipeline(cfg)

    if args.mode == "collect":
        print(json.dumps(pipe.daily_collect_and_score(), indent=2))
    elif args.mode == "refit":
        print(json.dumps(pipe.weekly_refit(), indent=2))
    elif args.mode == "once":
        print(json.dumps(pipe.run_once(), indent=2))
    else:
        pipe.scheduler_daemon()


if __name__ == "__main__":
    main()
