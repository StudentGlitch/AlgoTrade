"""
Phase 1: Pre-Flight Data Fetch & ML Warm-up
-------------------------------------------
1) Pull freshest OHLCV + macro data (yfinance) up to current day
2) Update master dataset in-place
3) Run synchronous training warm-up (collect + refit)
4) Validate model artifacts exist and were updated
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from train_pipeline import PipelineConfig, ContinuousTrainingPipeline


DEFAULT_MASTER = r"C:\Tugas Akhir\research\phase6_lpa_enriched.csv"
DEFAULT_PROD_DIR = r"C:\Tugas Akhir\production\models"
DEFAULT_LOG = r"C:\Tugas Akhir\production\phase1_train_log.jsonl"
DEFAULT_REPORT = r"C:\Tugas Akhir\PREFLIGHT_WARMUP_REPORT.md"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fetch_symbol_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    d = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, repair=True, progress=False)
    if d is None or d.empty:
        return pd.DataFrame()
    d = d.reset_index()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in d.columns]
    return d[keep].copy()


def enrich_price_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d["return"] = d["close"].pct_change()
    d["log_return"] = np.log(d["close"] / d["close"].shift(1))
    d["abs_return"] = d["return"].abs()
    d["range_pct"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["turnover_proxy"] = d["close"] * d["volume"]
    d["sma_5"] = d["close"].rolling(5).mean()
    d["sma_20"] = d["close"].rolling(20).mean()
    d["sma_ratio_5_20"] = d["sma_5"] / d["sma_20"]
    d["mom_5"] = d["close"] / d["close"].shift(5) - 1
    d["mom_20"] = d["close"] / d["close"].shift(20) - 1
    d["volatility_5d"] = d["return"].rolling(5).std()
    d["volatility_20d"] = d["return"].rolling(20).std()
    d["volume_z20"] = (d["volume"] - d["volume"].rolling(20).mean()) / d["volume"].rolling(20).std()
    return d


def fetch_macro(start: str, end: str) -> pd.DataFrame:
    vix = yf.download("^VIX", start=start, end=end, interval="1d", progress=False)
    fx = yf.download("USDIDR=X", start=start, end=end, interval="1d", progress=False)
    if vix.empty and fx.empty:
        return pd.DataFrame(columns=["date"])
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


@dataclass
class PreflightConfig:
    master_data: str = DEFAULT_MASTER
    prod_dir: str = DEFAULT_PROD_DIR
    log_path: str = DEFAULT_LOG
    report_path: str = DEFAULT_REPORT
    lookback_days: int = 90
    warmup_epochs: int = 6


class PreflightWarmup:
    def __init__(self, cfg: PreflightConfig):
        self.cfg = cfg

    def refresh_data(self) -> dict:
        if not os.path.exists(self.cfg.master_data):
            raise FileNotFoundError(f"Master dataset not found: {self.cfg.master_data}")

        df = pd.read_csv(self.cfg.master_data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company"]).copy()
        df = df.sort_values(["company", "date"])

        max_date = df["date"].max().date()
        end_date = now_utc().date() + timedelta(days=1)
        start_date = min(max_date - timedelta(days=self.cfg.lookback_days), end_date - timedelta(days=365))
        start_s, end_s = start_date.isoformat(), end_date.isoformat()

        symbols = sorted({f"{c}.JK" for c in df["company"].dropna().unique().tolist()})
        updates = []
        failed = []
        for sym in symbols:
            p = fetch_symbol_ohlcv(sym, start=start_s, end=end_s)
            if p.empty:
                failed.append(sym)
                continue
            p = enrich_price_features(p)
            p["symbol"] = sym
            p["company"] = sym.replace(".JK", "")
            updates.append(p)

        if updates:
            upd = pd.concat(updates, ignore_index=True)
            macro = fetch_macro(start=start_s, end=end_s)
            upd = upd.merge(macro, on="date", how="left")
            # merge/update by company+date
            key = ["company", "date"]
            base = df.copy()
            base = base.drop(columns=[c for c in upd.columns if c in base.columns and c not in key], errors="ignore")
            merged = base.merge(upd, on=key, how="outer")
            # retain symbol
            if "symbol_x" in merged.columns or "symbol_y" in merged.columns:
                sx = merged.get("symbol_x")
                sy = merged.get("symbol_y")
                merged["symbol"] = sx.combine_first(sy) if sx is not None else sy
                merged = merged.drop(columns=[c for c in ["symbol_x", "symbol_y"] if c in merged.columns], errors="ignore")
            merged = merged.sort_values(["company", "date"])
            merged.to_csv(self.cfg.master_data, index=False)
            rows_added = len(merged) - len(df)
            refreshed_rows = len(upd)
        else:
            rows_added = 0
            refreshed_rows = 0

        return {
            "symbols_attempted": len(symbols),
            "symbols_failed": failed,
            "refreshed_rows": int(refreshed_rows),
            "rows_added_net": int(rows_added),
            "start_fetch": start_s,
            "end_fetch": end_s,
        }

    def run_training_warmup(self) -> dict:
        start = now_utc()
        cfg = PipelineConfig(
            master_data_path=self.cfg.master_data,
            prod_dir=self.cfg.prod_dir,
            log_path=self.cfg.log_path,
            lstm_epochs=self.cfg.warmup_epochs,
        )
        pipe = ContinuousTrainingPipeline(cfg)
        result = pipe.run_once()  # synchronous, waits until done

        # Validation check: verify model artifacts updated after warm-up start
        lstm_path = os.path.join(self.cfg.prod_dir, "lstm_vol_model.h5")
        gmm_path = os.path.join(self.cfg.prod_dir, "lpa_gmm_model.pkl")
        for p in [lstm_path, gmm_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Expected model artifact missing: {p}")
            mtime = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
            if mtime < start:
                raise RuntimeError(f"Model artifact not updated by warm-up run: {p}")

        result["validated_models"] = {"lstm": lstm_path, "gmm": gmm_path}
        return result

    def run(self) -> dict:
        refresh_stats = self.refresh_data()
        warmup_stats = self.run_training_warmup()
        out = {
            "preflight_refresh": refresh_stats,
            "warmup_training": warmup_stats,
            "status": "ready_for_live_phase2",
            "timestamp_utc": now_utc().isoformat(),
        }
        self.write_report(out)
        return out

    def write_report(self, payload: dict) -> None:
        lines = []
        lines.append("# PREFLIGHT_WARMUP_REPORT")
        lines.append("")
        lines.append("## Summary")
        lines.append("- Pre-flight fetch completed.")
        lines.append("- Synchronous warm-up training completed.")
        lines.append("- Required model artifacts validated in production directory.")
        lines.append("")
        lines.append("## Refresh stats")
        lines.append("```json")
        lines.append(json.dumps(payload["preflight_refresh"], indent=2))
        lines.append("```")
        lines.append("")
        lines.append("## Warm-up training stats")
        lines.append("```json")
        lines.append(json.dumps(payload["warmup_training"], indent=2))
        lines.append("```")
        lines.append("")
        lines.append(f"## Status\n`{payload['status']}`")
        with open(self.cfg.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-flight data fetch + synchronous ML warm-up")
    p.add_argument("--master-data", default=DEFAULT_MASTER)
    p.add_argument("--prod-dir", default=DEFAULT_PROD_DIR)
    p.add_argument("--log-path", default=DEFAULT_LOG)
    p.add_argument("--report-path", default=DEFAULT_REPORT)
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--warmup-epochs", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreflightConfig(
        master_data=args.master_data,
        prod_dir=args.prod_dir,
        log_path=args.log_path,
        report_path=args.report_path,
        lookback_days=args.lookback_days,
        warmup_epochs=args.warmup_epochs,
    )
    runner = PreflightWarmup(cfg)
    out = runner.run()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
