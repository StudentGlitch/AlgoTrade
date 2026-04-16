import os
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
DATA_PATH = str(_SHARED / "data" / "Full Data.csv")
OUT_MD = str(_SHARED / "data" / "PHASE1_RESULTS.md")


def fit_fe(df: pd.DataFrame, dep: str, regressors: list[str]) -> tuple:
    use = [dep] + regressors
    d = df[use].copy().dropna()
    y = d[dep]
    x = d[regressors]
    model = PanelOLS(y, x, entity_effects=True, drop_absorbed=True, check_rank=False)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res, len(d)


def coef_table(res) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "coef": res.params,
            "std_err": res.std_errors,
            "t_stat": res.tstats,
            "p_value": res.pvalues,
        }
    )
    return out


def sig_label(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return "ns"


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]

    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")
    df = df.dropna(subset=["id_company", "tanggal_stata"]).copy()

    numeric_cols = ["residual", "return", "composite_index_fb", "pca_index_yt", "vol", "turnover"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    panel = df.set_index(["id_company", "tanggal_stata"]).sort_index()

    regressors = ["composite_index_fb", "pca_index_yt", "vol"]

    res_resid, n_resid = fit_fe(panel, "residual", regressors)
    res_ret, n_ret = fit_fe(panel, "return", regressors)

    tab_resid = coef_table(res_resid).round(6)
    tab_ret = coef_table(res_ret).round(6)

    fb_sig_resid = sig_label(float(tab_resid.loc["composite_index_fb", "p_value"]))
    yt_sig_resid = sig_label(float(tab_resid.loc["pca_index_yt", "p_value"]))
    fb_sig_ret = sig_label(float(tab_ret.loc["composite_index_fb", "p_value"]))
    yt_sig_ret = sig_label(float(tab_ret.loc["pca_index_yt", "p_value"]))

    lines: list[str] = []
    lines.append("# Phase 1 — Panel Fixed Effects Results")
    lines.append("")
    lines.append("## Model specification")
    lines.append("- Panel index: `id_company` (entity) and `tanggal_stata` (time)")
    lines.append("- Estimator: **Entity Fixed Effects (PanelOLS)** with clustered SE by entity")
    lines.append("- Main regressors: `composite_index_fb`, `pca_index_yt`")
    lines.append("- Control variable: `vol`")
    lines.append("")
    lines.append("## 1) Dependent variable: `residual` (primary)")
    lines.append(f"- Observations used: **{n_resid}**")
    lines.append(f"- R-squared (within): **{res_resid.rsquared_within:.6f}**")
    lines.append(f"- F-stat (robust): **{float(res_resid.f_statistic.stat):.4f}**, p-value **{float(res_resid.f_statistic.pval):.6g}**")
    lines.append("")
    lines.append(tab_resid.to_markdown())
    lines.append("")
    lines.append(
        f"- Significance check: `composite_index_fb` = **{fb_sig_resid}**, `pca_index_yt` = **{yt_sig_resid}**."
    )
    lines.append("")
    lines.append("## 2) Dependent variable: `return` (robustness)")
    lines.append(f"- Observations used: **{n_ret}**")
    lines.append(f"- R-squared (within): **{res_ret.rsquared_within:.6f}**")
    lines.append(f"- F-stat (robust): **{float(res_ret.f_statistic.stat):.4f}**, p-value **{float(res_ret.f_statistic.pval):.6g}**")
    lines.append("")
    lines.append(tab_ret.to_markdown())
    lines.append("")
    lines.append(
        f"- Significance check: `composite_index_fb` = **{fb_sig_ret}**, `pca_index_yt` = **{yt_sig_ret}**."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- This phase tests **contemporaneous** social-media effects on stock performance after controlling for company fixed effects and trading activity (`vol`)."
    )
    lines.append(
        "- Statistical significance is based on p-values in the tables above."
    )

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")


if __name__ == "__main__":
    main()
