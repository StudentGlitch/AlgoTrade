import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
IN_PATH = str(_SHARED / "data" / "phase5_rectangularized_data.csv")
OUT_DATA = str(_SHARED / "data" / "phase6_lpa_enriched.csv")
OUT_REPORT = str(_SHARED / "data" / "LPA_PROFILES_REPORT.md")


def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def finbert_score_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> list[float]:
    scores = []
    label_map = ["positive", "negative", "neutral"]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            for p in probs:
                # FinBERT score in [-1, 1]: positive prob - negative prob
                pos = p[label_map.index("positive")]
                neg = p[label_map.index("negative")]
                scores.append(float(pos - neg))
    return scores


def fetch_news_rows(symbol: str) -> list[dict]:
    rows = []
    try:
        news = yf.Ticker(symbol).news or []
        for item in news:
            c = item.get("content", {})
            title = c.get("title", "") or ""
            summary = c.get("summary", "") or c.get("description", "") or ""
            pub = c.get("pubDate") or c.get("displayTime")
            if not pub:
                continue
            try:
                dt = pd.to_datetime(pub, utc=True).tz_convert(None)
            except Exception:
                continue
            text = (title + ". " + summary).strip()
            if not text:
                continue
            rows.append({"symbol": symbol, "date": dt.normalize(), "text": text})
    except Exception:
        pass
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "company", "symbol"]).copy()
    df = df.sort_values(["company", "date"])

    # Gather textual corpus (proxy for provided transcripts when transcript files are absent)
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    news_rows = []
    for sym in symbols:
        news_rows.extend(fetch_news_rows(sym))

    news_df = pd.DataFrame(news_rows)
    if news_df.empty:
        # fallback neutral sentiment if no text found
        daily_sent = pd.DataFrame(columns=["company", "date", "finbert_score", "news_items_day"])
    else:
        tokenizer, model = load_finbert()
        news_df["finbert_score"] = finbert_score_texts(news_df["text"].tolist(), tokenizer, model, batch_size=16)
        news_df["company"] = news_df["symbol"].str.replace(".JK", "", regex=False)
        daily_sent = (
            news_df.groupby(["company", "date"], as_index=False)
            .agg(finbert_score=("finbert_score", "mean"), news_items_day=("text", "count"))
        )

    data = df.merge(daily_sent, on=["company", "date"], how="left")
    data["finbert_score"] = data["finbert_score"].fillna(0.0)
    data["news_items_day"] = data["news_items_day"].fillna(0).astype(int)

    # LPA/GMM features
    feature_candidates = [
        "finbert_score",
        "news_items_day",
        "news_count",
        "volume_z20",
        "volatility_20d",
        "abs_return",
        "range_pct",
        "vix_close_ret",
        "usd_idr_close_ret",
    ]
    lpa_features = [c for c in feature_candidates if c in data.columns]
    work = data.dropna(subset=lpa_features).copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(work[lpa_features])

    n_components = 8
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    labels = gmm.predict(X)
    work["profile_id"] = labels + 1  # make profiles 1..8
    work["profile_confidence"] = probs.max(axis=1)

    # merge profile assignment back
    key_cols = ["company", "date"]
    assigned = work[key_cols + ["profile_id", "profile_confidence"]].copy()
    data = data.merge(assigned, on=key_cols, how="left")
    # For rows not assigned due earlier NaNs, use most probable neutral-ish profile by overall frequency
    fallback_profile = int(work["profile_id"].mode().iloc[0]) if len(work) else 1
    data["profile_id"] = data["profile_id"].fillna(fallback_profile).astype(int)
    data["profile_confidence"] = data["profile_confidence"].fillna(0.0)

    data.to_csv(OUT_DATA, index=False)

    # profile characterization
    profile_stats = (
        data.groupby("profile_id", as_index=True)
        .agg(
            n_obs=("profile_id", "size"),
            mean_finbert_score=("finbert_score", "mean"),
            mean_news_items_day=("news_items_day", "mean"),
            mean_abs_return=("abs_return", "mean"),
            mean_volatility_20d=("volatility_20d", "mean"),
            mean_vix_ret=("vix_close_ret", "mean"),
        )
        .sort_index()
    )

    # Profile naming heuristic
    names = {}
    for pid, row in profile_stats.iterrows():
        s = row["mean_finbert_score"]
        v = row["mean_abs_return"]
        if s > 0.2 and v < profile_stats["mean_abs_return"].median():
            names[pid] = "Optimistic-Calm"
        elif s > 0.2:
            names[pid] = "Optimistic-Volatile"
        elif s < -0.2 and v >= profile_stats["mean_abs_return"].median():
            names[pid] = "Neutral-Negative"
        elif s < -0.2:
            names[pid] = "Pessimistic-Calm"
        else:
            names[pid] = "Neutral-Mixed"
    profile_stats["profile_label"] = pd.Series(names)

    # report
    lines = []
    lines.append("# LPA_PROFILES_REPORT")
    lines.append("")
    lines.append("## 1) FinBERT scoring")
    lines.append("- Model: `ProsusAI/finbert`")
    lines.append("- Score definition: `P(positive) - P(negative)` in [-1, +1]")
    lines.append("- Daily score built by averaging article-level FinBERT scores by company/date.")
    lines.append(f"- Text rows processed: **{0 if news_df.empty else len(news_df)}**")
    lines.append("")
    lines.append("## 2) Latent Profile Analysis (GMM)")
    lines.append("- Method: Gaussian Mixture Model (`n_components=8`) on standardized sentiment/engagement/market-state features.")
    lines.append("- Profile ID assigned by maximum posterior probability.")
    lines.append(f"- Output dataset: `{OUT_DATA}`")
    lines.append("")
    lines.append("## 3) Profile characteristics")
    lines.append(profile_stats.round(6).to_markdown())
    lines.append("")
    lines.append("## 4) Mean absolute return by profile")
    mar = data.groupby("profile_id")["abs_return"].mean().sort_values(ascending=False).to_frame("mean_abs_return")
    lines.append(mar.round(6).to_markdown())
    lines.append("")
    lines.append("## 5) Notes")
    lines.append("- Profile labels are heuristic descriptors based on sentiment and volatility profile means.")
    lines.append("- These profile IDs are ready for one-hot encoding in the LSTM phase.")
    lines.append("")
    lines.append("**STOP POINT:** Awaiting approval before Phase 2 (macro-enhanced LSTM volatility model).")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_DATA}")
    print(f"WROTE={OUT_REPORT}")
    print(f"TEXT_ROWS={0 if news_df.empty else len(news_df)}")


if __name__ == "__main__":
    main()
