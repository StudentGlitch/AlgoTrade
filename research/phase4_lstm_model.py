import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = r"C:\Tugas Akhir\research\phase4_preprocessed.csv"
OUT_MD = r"C:\Tugas Akhir\research\PHASE2_LSTM_RESULTS.md"
OUT_PLOT = r"C:\Tugas Akhir\research\lstm_actual_vs_predicted.png"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_sequences(df: pd.DataFrame, features: list[str], target: str, lookback: int = 7):
    X_list, y_list, date_list, company_list, base_list = [], [], [], [], []

    for cid, g in df.groupby("id_company"):
        g = g.sort_values("tanggal_stata").reset_index(drop=True)
        vals = g[features].values
        yvals = g[target].values
        dates = g["tanggal_stata"].values

        for i in range(lookback, len(g)):
            X_list.append(vals[i - lookback : i, :])
            y_list.append(yvals[i])
            date_list.append(dates[i])
            company_list.append(cid)
            # baseline = yesterday's volatility
            base_list.append(yvals[i - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    baseline = np.array(base_list, dtype=np.float32)
    meta = pd.DataFrame({"id_company": company_list, "tanggal_stata": pd.to_datetime(date_list)})
    return X, y, baseline, meta


def main() -> None:
    set_seed(42)

    df = pd.read_csv(DATA_PATH)
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")
    df = df.dropna(subset=["id_company", "tanggal_stata", "absolute_return"]).copy()
    df = df.sort_values(["id_company", "tanggal_stata"])

    feature_cols = [
        "absolute_return",
        "vol",
        "turnover",
        "composite_index_fb",
        "yt_spike",
        "fb_ma3",
        "fb_ma5",
        "fb_pct_change",
        "vol_lag1",
        "fb_vol_interaction",
    ]
    for c in feature_cols + ["absolute_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Stabilize heavy-tailed pct-change feature
    q1 = df["fb_pct_change"].quantile(0.01)
    q99 = df["fb_pct_change"].quantile(0.99)
    df["fb_pct_change"] = df["fb_pct_change"].clip(lower=q1, upper=q99)

    # Company-wise missing handling to avoid crossing entity info
    df[feature_cols] = (
        df.groupby("id_company")[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    df = df.dropna(subset=feature_cols + ["absolute_return"]).copy()

    # Build sequences (no cross-company mixing by construction)
    lookback = 7
    X, y, baseline_pred, meta = build_sequences(df, feature_cols, "absolute_return", lookback=lookback)

    # Chronological split: train<=2022, val=early 2023, test=later 2023
    train_mask = meta["tanggal_stata"] <= pd.Timestamp("2022-12-31")
    test_mask = meta["tanggal_stata"] >= pd.Timestamp("2023-01-01")

    X_train_all, y_train_all = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    baseline_test = baseline_pred[test_mask.values]
    meta_test = meta.loc[test_mask].reset_index(drop=True)

    # Validation split from tail of training period (chronological, no shuffle)
    split_idx = int(len(X_train_all) * 0.8)
    X_train, X_val = X_train_all[:split_idx], X_train_all[split_idx:]
    y_train, y_val = y_train_all[:split_idx], y_train_all[split_idx:]

    # Scale features using training only
    n_features = X.shape[2]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    def scale_3d(arr):
        arr2 = arr.reshape(-1, n_features)
        out = scaler.transform(arr2)
        return out.reshape(arr.shape)

    X_train = scale_3d(X_train)
    X_val = scale_3d(X_val)
    X_test = scale_3d(X_test)

    # LSTM model
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, n_features), dropout=0.2, recurrent_dropout=0.1),
            LSTM(32, dropout=0.2, recurrent_dropout=0.1),
            Dropout(0.2),
            Dense(16, activation="relu"),
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

    pred_test = model.predict(X_test, verbose=0).reshape(-1)

    rmse_lstm = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae_lstm = float(mean_absolute_error(y_test, pred_test))
    rmse_base = float(np.sqrt(mean_squared_error(y_test, baseline_test)))
    mae_base = float(mean_absolute_error(y_test, baseline_test))

    # Plot sample company (prefer BBRI, else PGAS, else first available)
    preferred = ["BBRI", "PGAS"]
    sample_company = None
    for p in preferred:
        if (meta_test["id_company"] == p).any():
            sample_company = p
            break
    if sample_company is None:
        sample_company = str(meta_test["id_company"].iloc[0])

    sample_idx = meta_test.index[meta_test["id_company"] == sample_company].tolist()
    sample_dates = meta_test.loc[sample_idx, "tanggal_stata"]
    sample_actual = y_test[sample_idx]
    sample_pred = pred_test[sample_idx]

    plt.figure(figsize=(11, 5))
    plt.plot(sample_dates, sample_actual, label="Actual absolute_return", linewidth=2)
    plt.plot(sample_dates, sample_pred, label="LSTM predicted", linewidth=2)
    plt.title(f"LSTM: Actual vs Predicted Volatility Proxy ({sample_company})")
    plt.xlabel("Date")
    plt.ylabel("absolute_return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)
    plt.close()

    lines = []
    lines.append("# PHASE 2 — LSTM Results")
    lines.append("")
    lines.append("## Data and sequence setup")
    lines.append("- Target: `absolute_return` (continuous volatility proxy)")
    lines.append(f"- Lookback window: **{lookback} days**")
    lines.append("- Sequences built strictly within each `id_company` (no cross-company sequence mixing).")
    lines.append("- Chronological split: train(<=2022), test(2023). Validation is the tail of training period.")
    lines.append("- Training uses `shuffle=False` to preserve temporal ordering.")
    lines.append("")
    lines.append("## Evaluation")
    lines.append(f"- LSTM RMSE: **{rmse_lstm:.6f}**")
    lines.append(f"- LSTM MAE: **{mae_lstm:.6f}**")
    lines.append(f"- Baseline RMSE (yesterday volatility): **{rmse_base:.6f}**")
    lines.append(f"- Baseline MAE (yesterday volatility): **{mae_base:.6f}**")
    lines.append("")
    lines.append("## Plot")
    lines.append(f"- Sample company plot ({sample_company}): `{OUT_PLOT}`")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Epochs run (with early stopping): **{len(history.history['loss'])}**")
    lines.append("- Lower RMSE/MAE indicates better predictive performance.")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")
    print(f"WROTE={OUT_PLOT}")
    print(f"SAMPLE_COMPANY={sample_company}")
    print(f"LSTM_RMSE={rmse_lstm}")
    print(f"BASE_RMSE={rmse_base}")


if __name__ == "__main__":
    main()
