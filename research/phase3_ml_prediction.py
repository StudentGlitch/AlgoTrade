import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


DATA_PATH = r"C:\Tugas Akhir\Full Data.csv"
OUT_MD = r"C:\Tugas Akhir\research\PHASE3_RESULTS.md"
OUT_RF_PNG = r"C:\Tugas Akhir\research\feature_importance_rf.png"
OUT_XGB_PNG = r"C:\Tugas Akhir\research\feature_importance_xgb.png"


def plot_importance(names, values, title, out_path):
    order = np.argsort(values)[::-1][:15]
    names = [names[i] for i in order]
    vals = [values[i] for i in order]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")

    social_cols = [
        "composite_index_fb",
        "pca_index_yt",
        "views_fb",
        "views_yt",
        "likes_fb",
        "shares_fb",
        "comments_fb",
        "likes_yt",
        "comments_yt",
    ]

    keep = ["tanggal_stata", "return"] + [c for c in social_cols if c in df.columns]
    work = df[keep].dropna(subset=["tanggal_stata"]).copy()
    for c in keep:
        if c != "tanggal_stata":
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # Daily aggregate for forecasting next-day direction/return
    daily = work.groupby("tanggal_stata", as_index=True).mean(numeric_only=True).sort_index()

    # Targets
    daily["target_return_t1"] = daily["return"].shift(-1)
    daily["target_up_t1"] = (daily["target_return_t1"] > 0).astype(int)

    # Lagged features for all social metrics (t-1, t-2) + current t
    feats = []
    for c in social_cols:
        if c in daily.columns:
            feats.append(c)
            daily[f"{c}_lag1"] = daily[c].shift(1)
            daily[f"{c}_lag2"] = daily[c].shift(2)
            feats.extend([f"{c}_lag1", f"{c}_lag2"])

    model_df = daily.dropna(subset=feats + ["target_return_t1", "target_up_t1"]).copy()

    X = model_df[feats].values
    y_reg = model_df["target_return_t1"].values
    y_cls = model_df["target_up_t1"].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    acc_scores = []
    f1_scores = []

    rf_importances = np.zeros(len(feats))
    xgb_importances = np.zeros(len(feats))

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]
        y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]

        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train_reg)
        pred_reg = rf.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, pred_reg)))
        rmse_scores.append(rmse)
        rf_importances += rf.feature_importances_

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train_cls)
        pred_cls = xgb.predict(X_test)
        acc = float(accuracy_score(y_test_cls, pred_cls))
        f1 = float(f1_score(y_test_cls, pred_cls, zero_division=0))
        acc_scores.append(acc)
        f1_scores.append(f1)
        xgb_importances += xgb.feature_importances_

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "rmse": rmse,
                "accuracy": acc,
                "f1": f1,
            }
        )

    rf_importances /= len(rmse_scores)
    xgb_importances /= len(rmse_scores)

    plot_importance(feats, rf_importances, "Random Forest Feature Importance", OUT_RF_PNG)
    plot_importance(feats, xgb_importances, "XGBoost Feature Importance", OUT_XGB_PNG)

    fi_rf = pd.DataFrame({"feature": feats, "importance": rf_importances}).sort_values("importance", ascending=False)
    fi_xgb = pd.DataFrame({"feature": feats, "importance": xgb_importances}).sort_values("importance", ascending=False)
    folds = pd.DataFrame(fold_rows)

    lines: list[str] = []
    lines.append("# Phase 3 — Advanced Predictive Modeling")
    lines.append("")
    lines.append("## Forecasting setup")
    lines.append("- Objective: use social-media data at time *t* to predict market movement at *t+1*.")
    lines.append("- Regression target: `target_return_t1` (next-day return).")
    lines.append("- Classification target: `target_up_t1` (1 if next-day return > 0 else 0).")
    lines.append("- Features: social metrics at t and lagged social metrics (t-1, t-2).")
    lines.append("- Validation: **TimeSeriesSplit (5 folds)** to prevent temporal leakage.")
    lines.append("")
    lines.append("## Performance (out-of-sample)")
    lines.append(f"- Random Forest RMSE (mean): **{np.mean(rmse_scores):.6f}**")
    lines.append(f"- XGBoost Accuracy (mean): **{np.mean(acc_scores):.6f}**")
    lines.append(f"- XGBoost F1-score (mean): **{np.mean(f1_scores):.6f}**")
    lines.append("")
    lines.append("### Fold-by-fold metrics")
    lines.append(folds.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Feature importance")
    lines.append(f"- RF plot: `{OUT_RF_PNG}`")
    lines.append(f"- XGBoost plot: `{OUT_XGB_PNG}`")
    lines.append("")
    lines.append("### Top RF features")
    lines.append(fi_rf.head(10).round(6).to_markdown(index=False))
    lines.append("")
    lines.append("### Top XGBoost features")
    lines.append(fi_xgb.head(10).round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Leakage control note")
    lines.append("- No random K-fold was used. All splits preserve chronological order.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")
    print(f"WROTE={OUT_RF_PNG}")
    print(f"WROTE={OUT_XGB_PNG}")


if __name__ == "__main__":
    main()
