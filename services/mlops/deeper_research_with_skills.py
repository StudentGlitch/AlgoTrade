import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind
from linearmodels.panel import PanelOLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
DATA_PATH = str(_SHARED / "data" / "Full Data.csv")
OUT_MD = str(_SHARED / "data" / "DEEP_RESEARCH_REPORT.md")
OUT_CSV = str(_SHARED / "data" / "company_level_effects.csv")


def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def fit_panel(panel: pd.DataFrame, dep: str, regs: list[str], time_effects: bool = True):
    d = panel[[dep] + regs].dropna()
    mod = PanelOLS(
        d[dep],
        d[regs],
        entity_effects=True,
        time_effects=time_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    return d, res


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")
    df = df.dropna(subset=["id_company", "tanggal_stata"]).copy()
    df = df.sort_values(["id_company", "tanggal_stata"])

    base_cols = [
        "residual",
        "return",
        "vol",
        "turnover",
        "composite_index_fb",
        "pca_index_yt",
        "likes_fb",
        "shares_fb",
        "comments_fb",
        "views_fb",
        "likes_yt",
        "comments_yt",
        "views_yt",
    ]
    to_num(df, [c for c in base_cols if c in df.columns])

    # Lags by company
    for c in ["composite_index_fb", "pca_index_yt", "residual", "return"]:
        if c in df.columns:
            df[f"{c}_lag1"] = df.groupby("id_company")[c].shift(1)
            df[f"{c}_lag2"] = df.groupby("id_company")[c].shift(2)

    # Interaction term
    df["fb_x_yt"] = df["composite_index_fb"] * df["pca_index_yt"]

    panel = df.set_index(["id_company", "tanggal_stata"]).sort_index()

    # Model A: two-way FE contemporaneous
    regs_a = ["composite_index_fb", "pca_index_yt", "vol"]
    d_a, res_a = fit_panel(panel, "residual", regs_a, time_effects=True)

    # Model B: lagged two-way FE
    regs_b = ["composite_index_fb_lag1", "pca_index_yt_lag1", "composite_index_fb_lag2", "pca_index_yt_lag2", "vol"]
    d_b, res_b = fit_panel(panel, "residual", regs_b, time_effects=True)

    # Model C: interaction
    regs_c = ["composite_index_fb", "pca_index_yt", "fb_x_yt", "vol"]
    d_c, res_c = fit_panel(panel, "residual", regs_c, time_effects=True)

    # VIF (using contemporaneous regressors only)
    x_vif = d_a[regs_a].dropna().copy()
    x_vif = sm.add_constant(x_vif)
    vif_rows = []
    for i, col in enumerate(x_vif.columns):
        if col == "const":
            continue
        vif_rows.append({"variable": col, "vif": float(variance_inflation_factor(x_vif.values, i))})
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)

    # Pooled OLS diagnostics proxy (for residual behavior checks)
    pooled = df[["residual"] + regs_a].dropna().copy()
    X_pool = sm.add_constant(pooled[regs_a])
    ols = sm.OLS(pooled["residual"], X_pool).fit()
    bp = het_breuschpagan(ols.resid, X_pool)
    bp_p = float(bp[1])
    dw_stat = float(durbin_watson(ols.resid))

    # Company-level heterogeneity (HC3 OLS by company)
    rows = []
    for cid, g in df.groupby("id_company"):
        gg = g[["residual"] + regs_a].dropna()
        if len(gg) < 120:
            continue
        X = sm.add_constant(gg[regs_a])
        m = sm.OLS(gg["residual"], X).fit(cov_type="HC3")
        rows.append(
            {
                "id_company": cid,
                "n_obs": len(gg),
                "coef_fb": float(m.params.get("composite_index_fb", np.nan)),
                "p_fb": float(m.pvalues.get("composite_index_fb", np.nan)),
                "coef_yt": float(m.params.get("pca_index_yt", np.nan)),
                "p_yt": float(m.pvalues.get("pca_index_yt", np.nan)),
            }
        )
    company_df = pd.DataFrame(rows)
    if not company_df.empty:
        company_df.to_csv(OUT_CSV, index=False)

    # Engagement benchmark analysis (from social-media-analyzer logic)
    # FB engagement proxy = (likes + comments + shares) / views * 100
    fb_eng = (
        (df["likes_fb"].fillna(0) + df["comments_fb"].fillna(0) + df["shares_fb"].fillna(0))
        / df["views_fb"].replace(0, np.nan)
        * 100
    )
    fb_eng = fb_eng.replace([np.inf, -np.inf], np.nan).dropna()
    fb_mean = float(fb_eng.mean()) if len(fb_eng) else np.nan
    fb_median = float(fb_eng.median()) if len(fb_eng) else np.nan

    # Event asymmetry: top 10% social attention days vs others (next-day residual)
    agg = (
        df.groupby("tanggal_stata", as_index=True)[["composite_index_fb", "pca_index_yt", "residual"]]
        .mean()
        .sort_index()
        .dropna()
    )
    agg["residual_t1"] = agg["residual"].shift(-1)
    attention = agg["composite_index_fb"] + agg["pca_index_yt"]
    thr = attention.quantile(0.9)
    hi = agg.loc[attention >= thr, "residual_t1"].dropna()
    lo = agg.loc[attention < thr, "residual_t1"].dropna()
    if len(hi) > 5 and len(lo) > 5:
        t_stat, p_val = ttest_ind(hi, lo, equal_var=False, nan_policy="omit")
        hi_mean = float(hi.mean())
        lo_mean = float(lo.mean())
    else:
        t_stat, p_val, hi_mean, lo_mean = np.nan, np.nan, np.nan, np.nan

    # Predictive uplift test: RF with vs without social features (time-split)
    daily = (
        df.groupby("tanggal_stata", as_index=True)[
            ["return", "vol", "turnover", "composite_index_fb", "pca_index_yt", "views_fb", "views_yt"]
        ]
        .mean()
        .sort_index()
    )
    daily["target_return_t1"] = daily["return"].shift(-1)
    for c in ["composite_index_fb", "pca_index_yt", "views_fb", "views_yt"]:
        daily[f"{c}_lag1"] = daily[c].shift(1)
        daily[f"{c}_lag2"] = daily[c].shift(2)

    base_feats = ["vol", "turnover", "return"]
    social_feats = [
        "composite_index_fb",
        "pca_index_yt",
        "views_fb",
        "views_yt",
        "composite_index_fb_lag1",
        "composite_index_fb_lag2",
        "pca_index_yt_lag1",
        "pca_index_yt_lag2",
        "views_fb_lag1",
        "views_fb_lag2",
        "views_yt_lag1",
        "views_yt_lag2",
    ]
    valid = daily.dropna(subset=base_feats + social_feats + ["target_return_t1"]).copy()
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_base, rmse_social = [], []
    for tr, te in tscv.split(valid):
        trd = valid.iloc[tr]
        ted = valid.iloc[te]
        m0 = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1)
        m1 = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=3, n_jobs=-1)
        m0.fit(trd[base_feats], trd["target_return_t1"])
        m1.fit(trd[base_feats + social_feats], trd["target_return_t1"])
        p0 = m0.predict(ted[base_feats])
        p1 = m1.predict(ted[base_feats + social_feats])
        rmse_base.append(float(np.sqrt(mean_squared_error(ted["target_return_t1"], p0))))
        rmse_social.append(float(np.sqrt(mean_squared_error(ted["target_return_t1"], p1))))

    uplift = float(np.mean(rmse_base) - np.mean(rmse_social))

    # Write report
    def tab_for(res):
        return (
            pd.DataFrame(
                {
                    "coef": res.params,
                    "std_err": res.std_errors,
                    "t": res.tstats,
                    "p": res.pvalues,
                }
            )
            .round(6)
            .to_markdown()
        )

    lines = []
    lines.append("# DEEP_RESEARCH_REPORT — Skill-Guided Extensions")
    lines.append("")
    lines.append("This deeper pass applies methods inspired by the installed `statsmodels` and `social-media-analyzer` skills.")
    lines.append("")
    lines.append("## 1) Two-way Fixed Effects robustness")
    lines.append("- Entity FE + Time FE with two-way clustered SE (entity and time).")
    lines.append("")
    lines.append("### Model A: Contemporaneous effects (`residual`)")
    lines.append(f"- Within R²: **{res_a.rsquared_within:.6f}**")
    lines.append(tab_for(res_a))
    lines.append("")
    lines.append("### Model B: Lagged effects (t-1, t-2)")
    lines.append(f"- Within R²: **{res_b.rsquared_within:.6f}**")
    lines.append(tab_for(res_b))
    lines.append("")
    lines.append("### Model C: Interaction effect (`FB × YT`)")
    lines.append(f"- Within R²: **{res_c.rsquared_within:.6f}**")
    lines.append(tab_for(res_c))
    lines.append("")
    lines.append("## 2) Diagnostics (`statsmodels` workflow)")
    lines.append(f"- Breusch-Pagan p-value (pooled proxy): **{bp_p:.6g}**")
    lines.append(f"- Durbin-Watson statistic (pooled proxy): **{dw_stat:.4f}**")
    lines.append("- VIF table:")
    lines.append(vif_df.round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## 3) Company-level heterogeneity")
    if not company_df.empty:
        sig_fb = int((company_df["p_fb"] < 0.05).sum())
        sig_yt = int((company_df["p_yt"] < 0.05).sum())
        nco = len(company_df)
        lines.append(f"- Company models estimated: **{nco}**")
        lines.append(f"- Significant FB effects (p<0.05): **{sig_fb}/{nco}**")
        lines.append(f"- Significant YT effects (p<0.05): **{sig_yt}/{nco}**")
        lines.append(f"- Detailed file: `{OUT_CSV}`")
        lines.append("")
        lines.append(company_df.head(12).round(6).to_markdown(index=False))
    else:
        lines.append("- Insufficient per-company observations for robust heterogeneity analysis.")
    lines.append("")
    lines.append("## 4) Social engagement benchmark layer (`social-media-analyzer` style)")
    lines.append(
        f"- FB engagement proxy mean: **{fb_mean:.4f}%**, median: **{fb_median:.4f}%** (formula: (likes+comments+shares)/views_fb * 100)."
    )
    lines.append("- Benchmark reference from skill: Facebook average ~0.07%, good ~0.5–1.0%, excellent >1.0%.")
    lines.append("")
    lines.append("## 5) High-attention regime asymmetry")
    lines.append(
        f"- Next-day residual mean when social attention in top decile: **{hi_mean:.6f}** vs others **{lo_mean:.6f}**."
    )
    lines.append(f"- Welch t-test: t = **{t_stat:.4f}**, p = **{p_val:.6g}**.")
    lines.append("")
    lines.append("## 6) Predictive uplift from social features")
    lines.append(f"- RF baseline RMSE (no social features): **{np.mean(rmse_base):.6f}**")
    lines.append(f"- RF + social RMSE: **{np.mean(rmse_social):.6f}**")
    lines.append(f"- RMSE improvement (positive is better): **{uplift:.6f}**")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("- Two-way FE and lagged/interaction extensions are included for deeper causal structure checks.")
    lines.append("- Diagnostics, heterogeneity, and regime tests provide a stronger robustness layer beyond the initial phases.")
    lines.append("- Predictive uplift quantifies incremental value of social signals over baseline market controls.")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")
    print(f"WROTE={OUT_CSV}")


if __name__ == "__main__":
    main()
