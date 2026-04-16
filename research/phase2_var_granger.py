import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


DATA_PATH = r"C:\Tugas Akhir\Full Data.csv"
OUT_MD = r"C:\Tugas Akhir\research\PHASE2_RESULTS.md"


def choose_lag(var_data: pd.DataFrame, maxlags: int = 10) -> tuple[int, str]:
    sel = VAR(var_data).select_order(maxlags=maxlags)
    candidates = []
    if sel.aic is not None and not np.isnan(sel.aic):
        candidates.append((int(sel.aic), "AIC"))
    if sel.bic is not None and not np.isnan(sel.bic):
        candidates.append((int(sel.bic), "BIC"))
    if candidates:
        # Prefer BIC if available, else AIC
        for lag, crit in candidates:
            if crit == "BIC":
                return max(1, lag), crit
        lag, crit = candidates[0]
        return max(1, lag), crit
    return 1, "fallback"


def min_granger_p(data: pd.DataFrame, y: str, x: str, maxlag: int) -> tuple[float, int]:
    res = grangercausalitytests(data[[y, x]], maxlag=maxlag, verbose=False)
    best_p = 1.0
    best_lag = 1
    for lag, result in res.items():
        p = float(result[0]["ssr_ftest"][1])
        if p < best_p:
            best_p = p
            best_lag = lag
    return best_p, best_lag


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")

    needed = ["tanggal_stata", "return", "residual", "composite_index_fb", "pca_index_yt", "views_fb", "views_yt"]
    for c in needed:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c != "tanggal_stata" else df[c]

    # Aggregate panel to daily means for VAR/Granger dynamics
    daily = (
        df.dropna(subset=["tanggal_stata"])
        .groupby("tanggal_stata", as_index=True)[["return", "residual", "composite_index_fb", "pca_index_yt", "views_fb", "views_yt"]]
        .mean()
        .sort_index()
    )

    var_data = daily[["return", "composite_index_fb", "pca_index_yt"]].dropna()
    lag, lag_criterion = choose_lag(var_data, maxlags=10)

    var_model = VAR(var_data)
    var_res = var_model.fit(lag)

    p_fb_to_ret, lag_fb_to_ret = min_granger_p(var_data, "return", "composite_index_fb", maxlag=lag)
    p_ret_to_fb, lag_ret_to_fb = min_granger_p(var_data, "composite_index_fb", "return", maxlag=lag)

    # Event study: detect spikes in views
    event_df = daily[["views_fb", "views_yt", "residual", "return"]].dropna()
    q_fb = event_df["views_fb"].quantile(0.99)
    q_yt = event_df["views_yt"].quantile(0.99)

    spike_fb_dates = event_df.index[event_df["views_fb"] >= q_fb]
    spike_yt_dates = event_df.index[event_df["views_yt"] >= q_yt]
    event_dates = sorted(set(spike_fb_dates).union(set(spike_yt_dates)))

    # CAR over [-1, +1] using residual as abnormal return
    car_rows = []
    residual_series = event_df["residual"]
    for d in event_dates:
        if d not in residual_series.index:
            continue
        loc = residual_series.index.get_loc(d)
        if isinstance(loc, slice):
            continue
        if loc - 1 < 0 or loc + 1 >= len(residual_series):
            continue
        window_vals = residual_series.iloc[loc - 1 : loc + 2]
        car = float(window_vals.sum())
        etype = []
        if d in set(spike_fb_dates):
            etype.append("FB")
        if d in set(spike_yt_dates):
            etype.append("YT")
        car_rows.append({"date": d, "event_type": "+".join(etype), "car_m1_p1": car})

    car_df = pd.DataFrame(car_rows)
    avg_car = float(car_df["car_m1_p1"].mean()) if not car_df.empty else np.nan
    med_car = float(car_df["car_m1_p1"].median()) if not car_df.empty else np.nan

    lines: list[str] = []
    lines.append("# Phase 2 — Causality & Time Dynamics (VAR + Granger)")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Time-series input is built as **daily aggregated means** across companies.")
    lines.append("- Variables in VAR: `return`, `composite_index_fb`, `pca_index_yt`")
    lines.append(f"- Lag length selected using {lag_criterion}: **{lag}**")
    lines.append(f"- Sample size for VAR: **{len(var_data)} daily observations**")
    lines.append("")
    lines.append("## Granger causality")
    lines.append(f"- Test 1: `composite_index_fb` Granger-causes `return` -> min p-value **{p_fb_to_ret:.6g}** (best lag {lag_fb_to_ret})")
    lines.append(f"- Test 2: `return` Granger-causes `composite_index_fb` -> min p-value **{p_ret_to_fb:.6g}** (best lag {lag_ret_to_fb})")
    lines.append("- Rule: p-value < 0.05 indicates evidence of Granger causality.")
    lines.append("")
    lines.append("## VAR coefficient snapshot (`return` equation)")
    coef_return = var_res.params["return"].to_frame("coef").round(6)
    lines.append(coef_return.to_markdown())
    lines.append("")
    lines.append("## Event study (spikes in `views_fb` / `views_yt`)")
    lines.append(f"- Spike threshold: top 1% (99th percentile) for each views metric.")
    lines.append(f"- Number of event dates: **{0 if car_df.empty else len(car_df)}**")
    if not car_df.empty:
        lines.append(f"- Average CAR[-1,+1] (using `residual`): **{avg_car:.6f}**")
        lines.append(f"- Median CAR[-1,+1] (using `residual`): **{med_car:.6f}**")
        lines.append("")
        lines.append(car_df.head(15).to_markdown(index=False))
    else:
        lines.append("- No eligible event windows for CAR computation.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Granger results quantify directional predictability, not structural causality.")
    lines.append("- Event-study CAR summarizes short-window abnormal return response around extreme social-attention spikes.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")


if __name__ == "__main__":
    main()
