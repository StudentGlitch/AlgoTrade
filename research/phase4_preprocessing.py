import os
import numpy as np
import pandas as pd


DATA_PATH = r"C:\Tugas Akhir\Full Data.csv"
OUT_DATA = r"C:\Tugas Akhir\research\phase4_preprocessed.csv"
OUT_AUDIT = r"C:\Tugas Akhir\PREPROCESSING_AUDIT.md"


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")
    df = df.dropna(subset=["id_company", "tanggal_stata"]).copy()
    df = df.sort_values(["id_company", "tanggal_stata"])

    numeric_cols = [
        "return",
        "vol",
        "turnover",
        "composite_index_fb",
        "pca_index_yt",
        "views_yt",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Target shift: volatility proxy
    df["absolute_return"] = df["return"].abs()

    # Zero-inflation handling for YT: top 5% threshold excluding zeros
    yt_source = "pca_index_yt" if "pca_index_yt" in df.columns else "views_yt"
    yt_nonzero = df.loc[df[yt_source] > 0, yt_source].dropna()
    if len(yt_nonzero) > 0:
        yt_threshold = float(yt_nonzero.quantile(0.95))
        df["yt_spike"] = ((df[yt_source] >= yt_threshold) & (df[yt_source] > 0)).astype(int)
    else:
        yt_threshold = np.nan
        df["yt_spike"] = 0

    # Feature engineering by company to avoid leakage across firms
    grp = df.groupby("id_company")
    df["fb_ma3"] = grp["composite_index_fb"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["fb_ma5"] = grp["composite_index_fb"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["fb_pct_change"] = grp["composite_index_fb"].transform(lambda s: s.pct_change())
    df["fb_pct_change"] = df["fb_pct_change"].replace([np.inf, -np.inf], np.nan)
    df["vol_lag1"] = grp["vol"].shift(1)
    df["fb_vol_interaction"] = df["composite_index_fb"] * df["vol_lag1"]

    # Basic feature stats
    feat_cols = [
        "absolute_return",
        "yt_spike",
        "composite_index_fb",
        "fb_ma3",
        "fb_ma5",
        "fb_pct_change",
        "vol_lag1",
        "fb_vol_interaction",
    ]
    feat_stats = df[feat_cols].describe().T.round(6)

    # Zero-inflation summary
    yt_zero_pct = float((df[yt_source] == 0).mean() * 100) if yt_source in df.columns else np.nan
    yt_spike_rate = float(df["yt_spike"].mean() * 100)

    # Save preprocessed dataset
    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    df.to_csv(OUT_DATA, index=False)

    lines = []
    lines.append("# PREPROCESSING_AUDIT")
    lines.append("")
    lines.append("## 1) Input and target shift")
    lines.append(f"- Source dataset: `{DATA_PATH}`")
    lines.append(f"- Rows after basic cleaning: **{len(df)}**")
    lines.append("- New target variable: `absolute_return = abs(return)` (volatility proxy)")
    lines.append("- Alternative targets retained: `vol`, `turnover`")
    lines.append("")
    lines.append("## 2) Zero-inflation handling (YouTube)")
    lines.append(f"- YT source used for spike binning: `{yt_source}`")
    lines.append(f"- Zero share in `{yt_source}`: **{yt_zero_pct:.2f}%**")
    lines.append(f"- Top-5% (non-zero) threshold: **{yt_threshold:.6f}**")
    lines.append("- Event variable: `yt_spike = 1 if YT source >= threshold and >0, else 0`")
    lines.append(f"- Spike event rate: **{yt_spike_rate:.2f}%**")
    lines.append("")
    lines.append("## 3) Engineered features")
    lines.append("- `fb_ma3`: 3-day rolling mean of `composite_index_fb` (within company)")
    lines.append("- `fb_ma5`: 5-day rolling mean of `composite_index_fb` (within company)")
    lines.append("- `fb_pct_change`: daily pct change of `composite_index_fb` (within company)")
    lines.append("- `vol_lag1`: lagged `vol` by 1 day (within company)")
    lines.append("- `fb_vol_interaction = composite_index_fb * vol_lag1`")
    lines.append("")
    lines.append("## 4) Basic statistics of new variables")
    lines.append(feat_stats.to_markdown())
    lines.append("")
    lines.append("## 5) Output artifacts")
    lines.append(f"- Preprocessed dataset: `{OUT_DATA}`")
    lines.append(f"- Audit report: `{OUT_AUDIT}`")
    lines.append("")
    lines.append("## 6) Phase 0 stop point")
    lines.append("- Phase 0 preprocessing completed.")
    lines.append("- **STOP POINT:** Awaiting approval before Phase 1 (Event Study around `yt_spike`).")

    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_DATA}")
    print(f"WROTE={OUT_AUDIT}")
    print(f"YT_SOURCE={yt_source}")
    print(f"YT_THRESHOLD={yt_threshold}")


if __name__ == "__main__":
    main()
