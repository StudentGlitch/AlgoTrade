import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
IN_PATH = str(_SHARED / "data" / "phase6_lpa_enriched.csv")
OUT_MD = str(_SHARED / "data" / "LSTM_FINAL_RESULTS.md")
OUT_PLOT = str(_SHARED / "data" / "phase7_lstm_actual_vs_pred.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_sequences(df: pd.DataFrame, features: list[str], target: str, lookback: int):
    X, y, base, meta = [], [], [], []
    for company, g in df.groupby("company"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[features].values
        t = g[target].values
        d = g["date"].values
        for i in range(lookback, len(g)):
            X.append(vals[i - lookback : i, :])
            y.append(t[i])
            base.append(t[i - 1])  # persistence baseline
            meta.append((company, d[i]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    meta = pd.DataFrame(meta, columns=["company", "date"])
    meta["date"] = pd.to_datetime(meta["date"])
    return X, y, base, meta


def main() -> None:
    set_seed(42)
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "company", "abs_return"]).copy()
    df = df.sort_values(["company", "date"])

    # Core features
    technical = [c for c in ["rsi_14", "sma_ratio_5_20", "mom_5", "mom_20", "volatility_5d", "volatility_20d", "volume_z20", "range_pct"] if c in df.columns]
    macro = [c for c in ["jkse_close_ret", "vix_close_ret", "usd_idr_close_ret", "oil_close_ret", "gold_close_ret"] if c in df.columns]
    sentiment = [c for c in ["finbert_score", "news_items_day", "news_count"] if c in df.columns]

    # Interaction features
    df["sentiment_vol_interaction"] = df["finbert_score"] * df["volume"]
    inter = ["sentiment_vol_interaction"]
    if "fb_vol_interaction" in df.columns:
        inter.append("fb_vol_interaction")

    # One-hot profile IDs
    if "profile_id" in df.columns:
        prof = pd.get_dummies(df["profile_id"].astype(int), prefix="profile", dtype=float)
        df = pd.concat([df, prof], axis=1)
        profile_cols = prof.columns.tolist()
    else:
        profile_cols = []

    feature_cols = technical + macro + sentiment + inter + profile_cols + ["abs_return"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Missing handling
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df.groupby("company")[feature_cols].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=feature_cols + ["abs_return"]).copy()

    lookback = 10
    X, y, baseline, meta = build_sequences(df, feature_cols, "abs_return", lookback)

    # Chronological split (no shuffle)
    train_mask = meta["date"] <= pd.Timestamp("2025-06-30")
    test_mask = meta["date"] > pd.Timestamp("2025-06-30")

    X_train_all, y_train_all = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    baseline_test = baseline[test_mask.values]
    meta_test = meta.loc[test_mask].reset_index(drop=True)

    split_idx = int(len(X_train_all) * 0.85)
    X_train, X_val = X_train_all[:split_idx], X_train_all[split_idx:]
    y_train, y_val = y_train_all[:split_idx], y_train_all[split_idx:]

    # Scale features on train only
    n_features = X.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def transform(arr):
        return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape)

    X_train = transform(X_train)
    X_val = transform(X_val)
    X_test = transform(X_test)

    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            LSTM(48, dropout=0.2, recurrent_dropout=0.1),
            Dropout(0.2),
            Dense(24, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        shuffle=False,
        callbacks=[es],
        verbose=0,
    )

    pred = model.predict(X_test, verbose=0).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_test)))
    base_mae = float(mean_absolute_error(y_test, baseline_test))
    base_r2 = float(r2_score(y_test, baseline_test))

    # sample company plot
    preferred = ["BBRI", "PGAS"]
    sample_company = next((p for p in preferred if (meta_test["company"] == p).any()), meta_test["company"].iloc[0])
    idx = meta_test.index[meta_test["company"] == sample_company].tolist()
    d = meta_test.loc[idx, "date"]
    ya = y_test[idx]
    yp = pred[idx]

    plt.figure(figsize=(11, 5))
    plt.plot(d, ya, label="Actual abs_return", linewidth=2)
    plt.plot(d, yp, label="Predicted abs_return", linewidth=2)
    plt.title(f"LSTM Volatility Forecast — {sample_company}")
    plt.xlabel("Date")
    plt.ylabel("abs_return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)
    plt.close()

    lines = []
    lines.append("# LSTM_FINAL_RESULTS")
    lines.append("")
    lines.append("## 1) Modeling target and features")
    lines.append("- Target: `abs_return` (volatility magnitude, not directional return)")
    lines.append("- Feature blocks: LPA profile one-hot, FinBERT score, technicals, macro factors (JKSE/VIX/USDIDR/Oil/Gold), interaction terms.")
    lines.append(f"- Lookback window: **{lookback}**")
    lines.append("- Sequence generation is per-company; no cross-company sequence mixing.")
    lines.append("")
    lines.append("## 2) Temporal split and leakage control")
    lines.append("- Train period: <= 2025-06-30")
    lines.append("- Test period: > 2025-06-30")
    lines.append("- Validation: tail split of training data")
    lines.append("- No random shuffling (`shuffle=False`).")
    lines.append("")
    lines.append("## 3) Performance")
    lines.append(f"- LSTM RMSE: **{rmse:.6f}**")
    lines.append(f"- LSTM MAE: **{mae:.6f}**")
    lines.append(f"- LSTM R² (variance explained): **{r2:.6f}**")
    lines.append("")
    lines.append(f"- Persistence RMSE: **{base_rmse:.6f}**")
    lines.append(f"- Persistence MAE: **{base_mae:.6f}**")
    lines.append(f"- Persistence R²: **{base_r2:.6f}**")
    lines.append("")
    lines.append("## 4) Benchmark comparison")
    lines.append("- Goal: beat persistence RMSE baseline.")
    lines.append(f"- RMSE improvement (baseline - LSTM): **{(base_rmse - rmse):.6f}**")
    lines.append("")
    lines.append("## 5) Plot")
    lines.append(f"- Actual vs predicted sample plot ({sample_company}): `{OUT_PLOT}`")
    lines.append("")
    lines.append("## 6) Training details")
    lines.append(f"- Epochs run (early stopping): **{len(history.history['loss'])}**")
    lines.append(f"- Number of test sequences: **{len(y_test)}**")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")
    print(f"WROTE={OUT_PLOT}")
    print(f"RMSE={rmse};BASE_RMSE={base_rmse};R2={r2}")


if __name__ == "__main__":
    main()
