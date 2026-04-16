from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from influxdb_client import InfluxDBClient, Point, WriteOptions
except Exception:  # pragma: no cover - optional runtime dependency
    InfluxDBClient = None
    Point = None
    WriteOptions = None


@dataclass
class DatabaseConfig:
    url: str = os.getenv("INFLUXDB_URL", "")
    token: str = os.getenv("INFLUXDB_TOKEN", "")
    org: str = os.getenv("INFLUXDB_ORG", "")
    bucket: str = os.getenv("INFLUXDB_BUCKET", "")
    measurement: str = os.getenv("INFLUXDB_MEASUREMENT", "market_panel")


class DatabaseConnector:
    """InfluxDB connector for market-wide concurrent ingestion and query."""

    def __init__(self, cfg: Optional[DatabaseConfig] = None):
        self.cfg = cfg or DatabaseConfig()
        self.client = None
        self.write_api = None
        if self.enabled() and InfluxDBClient is not None:
            self.client = InfluxDBClient(url=self.cfg.url, token=self.cfg.token, org=self.cfg.org)
            self.write_api = self.client.write_api(
                write_options=WriteOptions(
                    batch_size=5000,
                    flush_interval=1000,  # milliseconds
                    jitter_interval=200,  # milliseconds
                    retry_interval=5000,  # milliseconds
                    max_retries=5,
                    max_retry_delay=30000,  # milliseconds
                    max_close_wait=120000,  # milliseconds
                    exponential_base=2,
                )
            )

    def enabled(self) -> bool:
        return bool(self.cfg.url and self.cfg.token and self.cfg.org and self.cfg.bucket)

    def available(self) -> bool:
        return self.client is not None and self.write_api is not None and Point is not None

    def write_market_panel(
        self,
        df: pd.DataFrame,
        *,
        measurement: Optional[str] = None,
        time_col: str = "date",
        tag_cols: Optional[list[str]] = None,
    ) -> int:
        if df.empty or not self.available() or time_col not in df.columns:
            return 0

        m = measurement or self.cfg.measurement
        tags = tag_cols or [c for c in ["company", "symbol", "sector", "industry"] if c in df.columns]
        numeric_cols = [c for c in df.columns if c not in set(tags + [time_col]) and pd.api.types.is_numeric_dtype(df[c])]

        points = []
        work = df.dropna(subset=[time_col]).copy()
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        work = work.dropna(subset=[time_col])

        for _, row in work.iterrows():
            p = Point(m).time(row[time_col].to_pydatetime())
            for t in tags:
                v = row.get(t)
                if pd.notna(v):
                    p = p.tag(t, str(v))
            for c in numeric_cols:
                v = row.get(c)
                if pd.notna(v):
                    p = p.field(c, float(v))
            points.append(p)

        if points:
            self.write_api.write(bucket=self.cfg.bucket, org=self.cfg.org, record=points)
        return int(len(points))

    def query(self, flux_query: str) -> pd.DataFrame:
        if not self.available() or not flux_query.strip():
            return pd.DataFrame()
        q = self.client.query_api().query_data_frame(flux_query, org=self.cfg.org)
        if isinstance(q, list):
            if not q:
                return pd.DataFrame()
            return pd.concat(q, ignore_index=True)
        return q if isinstance(q, pd.DataFrame) else pd.DataFrame()

    def close(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass


class SectorSentimentProxyImputer:
    """Imputes zero-mention sentiment using sector/day aggregates with causal-safe fallbacks."""

    TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "micro", "microcap", "micro_cap"}
    TIER_LOW_QUANTILE = 0.33
    TIER_HIGH_QUANTILE = 0.66

    def __init__(
        self,
        sentiment_col: str = "finbert_score",
        mention_col: str = "news_items_day",
        date_col: str = "date",
        microcap_flag_col: str = "is_micro_cap",
        sector_candidates: Optional[list[str]] = None,
    ):
        self.sentiment_col = sentiment_col
        self.mention_col = mention_col
        self.date_col = date_col
        self.microcap_flag_col = microcap_flag_col
        self.sector_candidates = sector_candidates or ["sector", "industry", "sector_name", "gics_sector"]

    def _pick_sector_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in self.sector_candidates:
            if c in df.columns:
                return c
        return None

    def _to_bool(self, s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        vals = s.astype(str).str.strip().str.lower()
        return vals.isin(self.TRUTHY_VALUES)

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if df.empty or self.date_col not in df.columns:
            return df, {
                "sector_proxy_rows": 0,
                "global_proxy_rows": 0,
                "neutral_fallback_rows": 0,
                "direct_rows": int(len(df)),
                "transfer_learning_tiers_added": False,
            }

        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col], errors="coerce")

        if self.sentiment_col not in out.columns:
            out[self.sentiment_col] = np.nan
        if self.mention_col not in out.columns:
            out[self.mention_col] = 0

        out[self.sentiment_col] = pd.to_numeric(out[self.sentiment_col], errors="coerce")
        out[self.mention_col] = pd.to_numeric(out[self.mention_col], errors="coerce").fillna(0)

        sector_col = self._pick_sector_col(out)
        if sector_col is None:
            sector_col = "sector_proxy"
            out[sector_col] = "GLOBAL"
        out[sector_col] = out[sector_col].fillna("UNKNOWN").astype(str)

        positive_mentions = out[self.mention_col] > 0
        sector_daily = (
            out.loc[positive_mentions]
            .groupby([self.date_col, sector_col], as_index=False)[self.sentiment_col]
            .mean()
            .rename(columns={self.sentiment_col: "_sector_proxy_sentiment"})
        )
        global_daily = (
            out.loc[positive_mentions]
            .groupby(self.date_col, as_index=False)[self.sentiment_col]
            .mean()
            .rename(columns={self.sentiment_col: "_global_proxy_sentiment"})
        )

        out = out.merge(sector_daily, on=[self.date_col, sector_col], how="left")
        out = out.merge(global_daily, on=[self.date_col], how="left")

        # Phase 1 rule: impute only for zero-mention rows.
        no_mentions = out[self.mention_col] <= 0
        needs_impute = no_mentions

        if self.microcap_flag_col in out.columns:
            microcap_mask = self._to_bool(out[self.microcap_flag_col])
            target_mask = needs_impute & microcap_mask
        else:
            target_mask = needs_impute

        out["sentiment_imputation_source"] = "direct"

        sector_fill_mask = target_mask & out["_sector_proxy_sentiment"].notna()
        out.loc[sector_fill_mask, self.sentiment_col] = out.loc[sector_fill_mask, "_sector_proxy_sentiment"]
        out.loc[sector_fill_mask, "sentiment_imputation_source"] = "sector_proxy"

        still_missing = target_mask & out[self.sentiment_col].isna()
        global_fill_mask = still_missing & out["_global_proxy_sentiment"].notna()
        out.loc[global_fill_mask, self.sentiment_col] = out.loc[global_fill_mask, "_global_proxy_sentiment"]
        out.loc[global_fill_mask, "sentiment_imputation_source"] = "global_proxy"

        still_missing = target_mask & out[self.sentiment_col].isna()
        out.loc[still_missing, self.sentiment_col] = 0.0
        out.loc[still_missing, "sentiment_imputation_source"] = "neutral_fallback"

        out = self.prepare_transfer_learning_metadata(out)

        stats = {
            "sector_proxy_rows": int((out["sentiment_imputation_source"] == "sector_proxy").sum()),
            "global_proxy_rows": int((out["sentiment_imputation_source"] == "global_proxy").sum()),
            "neutral_fallback_rows": int((out["sentiment_imputation_source"] == "neutral_fallback").sum()),
            "direct_rows": int((out["sentiment_imputation_source"] == "direct").sum()),
            "sector_col_used": sector_col,
            "microcap_flag_used": bool(self.microcap_flag_col in out.columns),
            "transfer_learning_tiers_added": bool("liquidity_tier" in out.columns),
        }

        out = out.drop(columns=["_sector_proxy_sentiment", "_global_proxy_sentiment"], errors="ignore")
        return out, stats

    def prepare_transfer_learning_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "liquidity_tier" in out.columns:
            return out
        if "company" not in out.columns:
            out["liquidity_tier"] = "unknown"
            out["transfer_learning_anchor"] = 0
            return out

        liquidity_source = None
        for c in ["turnover_proxy", "volume", "market_cap"]:
            if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
                liquidity_source = c
                break

        if liquidity_source is None:
            out["liquidity_tier"] = "unknown"
            out["transfer_learning_anchor"] = 0
            return out

        grp = out.groupby("company", as_index=False)[liquidity_source].median()
        grp = grp.rename(columns={liquidity_source: "_liquidity_med"})
        q1 = grp["_liquidity_med"].quantile(self.TIER_LOW_QUANTILE)
        q2 = grp["_liquidity_med"].quantile(self.TIER_HIGH_QUANTILE)

        def to_tier(v: float) -> str:
            if pd.isna(v):
                return "unknown"
            if v < q1:
                return "micro_or_small"
            if v < q2:
                return "mid"
            return "large_or_liquid"

        grp["liquidity_tier"] = grp["_liquidity_med"].apply(to_tier)
        grp["transfer_learning_anchor"] = (grp["liquidity_tier"] == "large_or_liquid").astype(int)

        out = out.merge(grp[["company", "liquidity_tier", "transfer_learning_anchor"]], on="company", how="left")
        out["liquidity_tier"] = out["liquidity_tier"].fillna("unknown")
        out["transfer_learning_anchor"] = out["transfer_learning_anchor"].fillna(0).astype(int)
        return out
