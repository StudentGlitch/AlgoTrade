import os
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from openbb import obb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
BASE_DATA = str(_SHARED / "data" / "Full Data.csv")
OUT_DATA = str(_SHARED / "data" / "openbb_enriched_stock_data_2024_onward.csv")
OUT_REPORT = str(_SHARED / "data" / "OPENBB_DATA_EXPANSION_REPORT.md")
START_DATE = "2024-01-01"


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_price(symbol: str, start_date: str) -> pd.DataFrame:
    try:
        d = obb.equity.price.historical(symbol=symbol, start_date=start_date, provider="yfinance").to_df()
        if d is None or d.empty:
            return pd.DataFrame()
        d = d.reset_index().rename(columns={"index": "date"})
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        d["symbol"] = symbol
        return d
    except Exception:
        return pd.DataFrame()


def enrich_symbol(df: pd.DataFrame) -> pd.DataFrame:
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
    d["rsi_14"] = calc_rsi(d["close"], 14)
    return d


def fetch_macro_series(start_date: str) -> pd.DataFrame:
    series_map = {
        "jkse_close": "^JKSE",
        "vix_close": "^VIX",
        "usd_idr_close": "USDIDR=X",
        "oil_close": "CL=F",
        "gold_close": "GC=F",
    }
    out = None
    for col, sym in series_map.items():
        try:
            if sym.endswith("=X"):
                d = obb.currency.price.historical(symbol=sym, start_date=start_date, provider="yfinance").to_df()
            elif sym.startswith("^"):
                d = obb.index.price.historical(symbol=sym, start_date=start_date, provider="yfinance").to_df()
            else:
                d = obb.equity.price.historical(symbol=sym, start_date=start_date, provider="yfinance").to_df()
            d = d.reset_index().rename(columns={"index": "date"})
            d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
            d = d[["date", "close"]].rename(columns={"close": col})
            out = d if out is None else out.merge(d, on="date", how="outer")
        except Exception:
            continue
    if out is None:
        return pd.DataFrame(columns=["date"])
    out = out.sort_values("date")
    for c in [x for x in out.columns if x != "date"]:
        out[f"{c}_ret"] = out[c].pct_change()
    return out


def fetch_news_sentiment(symbol: str, analyzer: SentimentIntensityAnalyzer) -> dict:
    try:
        n = obb.news.company(symbol=symbol, provider="yfinance").to_df()
        if n is None or n.empty:
            return {"news_count": 0, "news_sentiment_mean": np.nan, "news_sentiment_std": np.nan}
        texts = (
            n["title"].fillna("").astype(str)
            + ". "
            + n.get("summary", pd.Series("", index=n.index)).fillna("").astype(str)
        )
        scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
        return {
            "news_count": int(len(scores)),
            "news_sentiment_mean": float(np.mean(scores)),
            "news_sentiment_std": float(np.std(scores)),
        }
    except Exception:
        return {"news_count": 0, "news_sentiment_mean": np.nan, "news_sentiment_std": np.nan}


def main() -> None:
    warnings.filterwarnings("ignore")
    base = pd.read_csv(BASE_DATA, sep=";")
    tickers = sorted(base["company"].dropna().unique().tolist())
    symbols = [f"{t}.JK" for t in tickers]

    price_frames = []
    failed_symbols = []
    for sym in symbols:
        d = fetch_price(sym, START_DATE)
        if d.empty:
            failed_symbols.append(sym)
            continue
        d = enrich_symbol(d)
        price_frames.append(d)

    if not price_frames:
        raise RuntimeError("No price data fetched from OpenBB.")

    market = pd.concat(price_frames, ignore_index=True)
    market["company"] = market["symbol"].str.replace(".JK", "", regex=False)

    macro = fetch_macro_series(START_DATE)
    market = market.merge(macro, on="date", how="left")

    analyzer = SentimentIntensityAnalyzer()
    news_rows = []
    for sym in market["symbol"].dropna().unique():
        ns = fetch_news_sentiment(sym, analyzer)
        ns["symbol"] = sym
        ns["company"] = sym.replace(".JK", "")
        news_rows.append(ns)
    news_df = pd.DataFrame(news_rows)

    market = market.merge(news_df, on=["symbol", "company"], how="left")

    # Keep modeling-ready columns
    keep = [
        "date",
        "company",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "log_return",
        "abs_return",
        "range_pct",
        "turnover_proxy",
        "sma_5",
        "sma_20",
        "sma_ratio_5_20",
        "mom_5",
        "mom_20",
        "volatility_5d",
        "volatility_20d",
        "volume_z20",
        "rsi_14",
        "jkse_close",
        "jkse_close_ret",
        "vix_close",
        "vix_close_ret",
        "usd_idr_close",
        "usd_idr_close_ret",
        "oil_close",
        "oil_close_ret",
        "gold_close",
        "gold_close_ret",
        "news_count",
        "news_sentiment_mean",
        "news_sentiment_std",
    ]
    market = market[[c for c in keep if c in market.columns]].sort_values(["company", "date"])

    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    market.to_csv(OUT_DATA, index=False)

    # Report
    date_min = market["date"].min()
    date_max = market["date"].max()
    n_rows = len(market)
    n_companies = market["company"].nunique()
    n_dates = market["date"].nunique()
    coverage = market.groupby("company")["date"].count().sort_values(ascending=False).to_frame("rows")
    macro_cols = [c for c in market.columns if c.endswith("_close") or c.endswith("_close_ret")]
    model_cols = [c for c in market.columns if c not in ["date", "company", "symbol"]]

    lines = []
    lines.append("# OPENBB Data Expansion Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("Expand the research dataset with newer market data, analysis features, macro context, and news sentiment proxies to support stronger predictive modeling.")
    lines.append("")
    lines.append("## Data collection")
    lines.append(f"- Source platform: **OpenBB** (`yfinance` provider where no API key is required)")
    lines.append(f"- Start date: **{START_DATE}**")
    lines.append(f"- End date: **{datetime.utcnow().date()}**")
    lines.append(f"- Target symbols attempted: **{len(symbols)}**")
    lines.append(f"- Symbols fetched successfully: **{n_companies}**")
    lines.append(f"- Failed symbols: **{len(failed_symbols)}**")
    if failed_symbols:
        lines.append(f"- Failed list: `{', '.join(failed_symbols)}`")
    lines.append("")
    lines.append("## Output dataset")
    lines.append(f"- File: `{OUT_DATA}`")
    lines.append(f"- Shape: **{n_rows} rows x {len(market.columns)} columns**")
    lines.append(f"- Date coverage: **{date_min.date()} to {date_max.date()}** ({n_dates} trading dates)")
    lines.append("")
    lines.append("## Added feature groups")
    lines.append("1. Price/return features: return, log_return, abs_return, range_pct")
    lines.append("2. Momentum/technical: SMA(5/20), SMA ratio, momentum(5/20), RSI14")
    lines.append("3. Volatility/liquidity: volatility_5d, volatility_20d, volume_z20, turnover_proxy")
    lines.append("4. Macro context: JKSE, VIX, USD/IDR, oil, gold (+ daily returns)")
    lines.append("5. News context: per-symbol `news_count`, `news_sentiment_mean`, `news_sentiment_std`")
    lines.append("")
    lines.append("## Coverage by company")
    lines.append(coverage.to_markdown())
    lines.append("")
    lines.append("## Notes for predictive modeling")
    lines.append("- This dataset is ready for chronological train/validation/test workflows.")
    lines.append("- `abs_return` and rolling volatility features are suitable volatility targets.")
    lines.append("- Macro and news columns can be tested as exogenous predictors.")
    lines.append("")
    lines.append("## Column inventory")
    lines.append(f"- Macro columns: `{', '.join(macro_cols)}`")
    lines.append(f"- Model-ready fields count (excluding id/date): **{len(model_cols)}**")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_DATA}")
    print(f"WROTE={OUT_REPORT}")
    print(f"ROWS={n_rows};COMPANIES={n_companies};DATES={n_dates}")


if __name__ == "__main__":
    main()
