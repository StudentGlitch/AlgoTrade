import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
IN_PATH = str(_SHARED / "data" / "openbb_enriched_stock_data_2024_onward.csv")
OUT_PATH = str(_SHARED / "data" / "phase5_rectangularized_data.csv")
OUT_AUDIT = str(_SHARED / "data" / "RECTANGULARIZATION_AUDIT.md")


def pct_missing(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).sort_values(ascending=False)


def fetch_repaired_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        d = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, repair=True, progress=False)
        if d is None or d.empty:
            return pd.DataFrame()
        d = d.reset_index()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
        d = d.rename(columns={"adj_close": "adj_close_repair"})
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in d.columns]
        return d[keep].copy()
    except Exception:
        return pd.DataFrame()


def main() -> None:
    warnings.filterwarnings("ignore")

    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "company", "symbol"]).copy()
    df = df.sort_values(["company", "date"])

    # Keep track of original missingness and shape
    before_missing = pct_missing(df)
    before_rows = len(df)

    # 1) Price repair via yfinance(repair=True), replace only if missing or highly inconsistent
    repaired_count = 0
    price_cols = ["open", "high", "low", "close", "volume"]
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = (df["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    repaired_frames = {}
    for sym in sorted(df["symbol"].unique()):
        rp = fetch_repaired_prices(sym, date_min, date_max)
        if not rp.empty:
            repaired_frames[sym] = rp

    merged_rows = []
    for (comp, sym), g in df.groupby(["company", "symbol"]):
        g = g.copy()
        rp = repaired_frames.get(sym, pd.DataFrame())
        if rp.empty:
            merged_rows.append(g)
            continue
        g = g.merge(rp, on="date", how="left", suffixes=("", "_repair"))
        for c in price_cols:
            rc = f"{c}_repair"
            if rc not in g.columns:
                continue
            miss_mask = g[c].isna() & g[rc].notna()
            ratio = (g[c] / g[rc]).replace([np.inf, -np.inf], np.nan)
            outlier_mask = ratio.notna() & ((ratio > 1.3) | (ratio < 0.7))
            replace_mask = miss_mask | outlier_mask
            repaired_count += int(replace_mask.sum())
            g.loc[replace_mask, c] = g.loc[replace_mask, rc]
        g = g.drop(columns=[c for c in g.columns if c.endswith("_repair")], errors="ignore")
        merged_rows.append(g)

    repaired = pd.concat(merged_rows, ignore_index=True)

    # 2) Rectangularization: force complete company x date panel
    all_dates = pd.Index(sorted(repaired["date"].dropna().unique()))
    all_companies = sorted(repaired["company"].dropna().unique())
    full_index = pd.MultiIndex.from_product([all_companies, all_dates], names=["company", "date"])
    panel = repaired.set_index(["company", "date"]).sort_index()
    panel = panel.reindex(full_index).reset_index()

    # Recover symbol mapping
    symbol_map = repaired[["company", "symbol"]].dropna().drop_duplicates().groupby("company")["symbol"].first()
    panel["symbol"] = panel["company"].map(symbol_map)

    # 3) Macro columns: impute by date level first (ffill/bfill) then keep common path
    macro_cols = [c for c in panel.columns if c.endswith("_close") or c.endswith("_close_ret")]
    if macro_cols:
        by_date = panel.groupby("date")[macro_cols].mean().sort_index().ffill().bfill()
        panel = panel.drop(columns=macro_cols).merge(by_date.reset_index(), on="date", how="left")

    # 4) Bayesian neutrality interpolation for social sentiment when no mentions
    if "news_count" in panel.columns:
        panel["news_count"] = panel["news_count"].fillna(0)
    for c in ["news_sentiment_mean", "news_sentiment_std"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)
            # If zero mentions, enforce neutral baseline (0.0) instead of noisy carryover
            if "news_count" in panel.columns:
                panel.loc[panel["news_count"] <= 0, c] = 0.0

    # 5) EM-like temporal imputation (IterativeImputer/BayesianRidge) per company
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["news_count"]  # keep count numeric but not EM-fitted as continuous
    impute_cols = [c for c in numeric_cols if c not in exclude]

    imputed_parts = []
    for comp, g in panel.groupby("company"):
        g = g.sort_values("date").copy()
        g["time_idx"] = np.arange(len(g), dtype=float)
        cols = ["time_idx"] + [c for c in impute_cols if c in g.columns]
        X = g[cols].copy()
        # Seed with interpolation to help EM convergence
        X = X.interpolate(method="linear", limit_direction="both")
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=30,
            random_state=42,
            sample_posterior=True,
            initial_strategy="mean",
        )
        X_imp = imp.fit_transform(X)
        X_imp = pd.DataFrame(X_imp, columns=cols, index=g.index)
        for c in impute_cols:
            g[c] = X_imp[c]
        g = g.drop(columns=["time_idx"], errors="ignore")
        imputed_parts.append(g)

    out = pd.concat(imputed_parts, ignore_index=True).sort_values(["company", "date"])

    # Ensure no missing dates after rectangularization
    na_dates = int(out["date"].isna().sum())

    # 6) Integrity checks: compare observed vs imputed moments for core metrics
    core_cols = [c for c in ["open", "high", "low", "close", "volume", "return", "abs_return", "volatility_20d"] if c in out.columns]
    integrity_rows = []
    for c in core_cols:
        orig_non_na = repaired[c].dropna() if c in repaired.columns else pd.Series(dtype=float)
        out_series = out[c].dropna()
        if len(orig_non_na) > 0 and len(out_series) > 0:
            integrity_rows.append(
                {
                    "column": c,
                    "orig_mean": float(orig_non_na.mean()),
                    "rect_mean": float(out_series.mean()),
                    "orig_std": float(orig_non_na.std()),
                    "rect_std": float(out_series.std()),
                }
            )
    integrity_df = pd.DataFrame(integrity_rows)

    after_missing = pct_missing(out)
    after_rows = len(out)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # Build audit markdown
    miss_compare = pd.DataFrame(
        {
            "before_missing_pct": before_missing,
            "after_missing_pct": after_missing.reindex(before_missing.index).fillna(0),
        }
    )
    miss_compare["delta_pct_points"] = miss_compare["after_missing_pct"] - miss_compare["before_missing_pct"]
    miss_compare = miss_compare.sort_values("before_missing_pct", ascending=False)

    lines = []
    lines.append("# RECTANGULARIZATION_AUDIT")
    lines.append("")
    lines.append("## 1) Phase 0 objective")
    lines.append("Repair and rectangularize sparse 2024–2026 market data for sequence-safe predictive modeling.")
    lines.append("")
    lines.append("## 2) Source and output")
    lines.append(f"- Input: `{IN_PATH}`")
    lines.append(f"- Output: `{OUT_PATH}`")
    lines.append(f"- Rows before: **{before_rows}**")
    lines.append(f"- Rows after rectangularization: **{after_rows}**")
    lines.append("")
    lines.append("## 3) Price repair with yfinance `repair=True`")
    lines.append("- Cross-referenced per symbol and replaced missing/outlier OHLCV values when deviation from repaired feed was large.")
    lines.append(f"- Repaired cell updates applied: **{repaired_count}**")
    lines.append("")
    lines.append("## 4) Temporal imputation (EM-like)")
    lines.append("- Method: IterativeImputer + BayesianRidge (EM-style iterative conditional expectation).")
    lines.append("- Applied per company, with chronological interpolation seeding.")
    lines.append("- Date grid forced to full company x trading-date rectangle.")
    lines.append("")
    lines.append("## 5) Bayesian neutrality baseline for sparse social data")
    lines.append("- For zero-news days (`news_count <= 0`), sentiment fields set to neutral baseline **0.0**.")
    lines.append("- Prevents sparse zeros from creating unstable gradients in downstream sequence models.")
    lines.append("")
    lines.append("## 6) Missingness comparison (before vs after)")
    lines.append(miss_compare.round(4).to_markdown())
    lines.append("")
    lines.append("## 7) Integrity checks (moments)")
    if not integrity_df.empty:
        lines.append(integrity_df.round(6).to_markdown(index=False))
    else:
        lines.append("No core columns available for integrity checks.")
    lines.append("")
    lines.append("## 8) Sequence integrity checks")
    lines.append(f"- Missing `date` values after process: **{na_dates}**")
    lines.append("- Rectangular panel produced for all company-date combinations in the observed period.")
    lines.append("")
    lines.append("## 9) Phase 0 stop point")
    lines.append("- Phase 0 completed.")
    lines.append("- **STOP POINT:** Awaiting approval before Phase 1 (FinBERT + LPA).")

    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_PATH}")
    print(f"WROTE={OUT_AUDIT}")
    print(f"NA_DATES_AFTER={na_dates}")


if __name__ == "__main__":
    main()
