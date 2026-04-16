# DEEP_RESEARCH_REPORT — Skill-Guided Extensions

This deeper pass applies methods inspired by the installed `statsmodels` and `social-media-analyzer` skills.

## 1) Two-way Fixed Effects robustness
- Entity FE + Time FE with two-way clustered SE (entity and time).

### Model A: Contemporaneous effects (`residual`)
- Within R²: **0.000193**
|                    |   coef |   std_err |         t |        p |
|:-------------------|-------:|----------:|----------:|---------:|
| composite_index_fb | -2e-06 |   2.6e-05 | -0.059904 | 0.952233 |
| pca_index_yt       | -5e-06 |   3.3e-05 | -0.149825 | 0.880906 |
| vol                | -0     |   0       | -0.676674 | 0.49863  |

### Model B: Lagged effects (t-1, t-2)
- Within R²: **0.000518**
|                         |     coef |   std_err |         t |        p |
|:------------------------|---------:|----------:|----------:|---------:|
| composite_index_fb_lag1 |  1.5e-05 |   3.1e-05 |  0.485148 | 0.627583 |
| pca_index_yt_lag1       |  4.4e-05 |   4.8e-05 |  0.915966 | 0.359708 |
| composite_index_fb_lag2 |  2.2e-05 |   2.3e-05 |  0.962285 | 0.335931 |
| pca_index_yt_lag2       | -1e-05   |   3.6e-05 | -0.270015 | 0.787155 |
| vol                     | -0       |   0       | -1.08688  | 0.277118 |

### Model C: Interaction effect (`FB × YT`)
- Within R²: **0.000193**
|                    |   coef |   std_err |         t |        p |
|:-------------------|-------:|----------:|----------:|---------:|
| composite_index_fb | -1e-06 |   2.7e-05 | -0.036411 | 0.970955 |
| pca_index_yt       | -4e-06 |   3.7e-05 | -0.105004 | 0.916375 |
| fb_x_yt            | -0     |   1e-06   | -0.147755 | 0.88254  |
| vol                | -0     |   0       | -0.67619  | 0.498937 |

## 2) Diagnostics (`statsmodels` workflow)
- Breusch-Pagan p-value (pooled proxy): **1.92761e-15**
- Durbin-Watson statistic (pooled proxy): **2.1279**
- VIF table:
| variable           |    vif |
|:-------------------|-------:|
| vol                | 1.0023 |
| composite_index_fb | 1.002  |
| pca_index_yt       | 1.0016 |

## 3) Company-level heterogeneity
- Company models estimated: **18**
- Significant FB effects (p<0.05): **0/18**
- Significant YT effects (p<0.05): **1/18**
- Detailed file: `C:\Tugas Akhir\research\company_level_effects.csv`

| id_company   |   n_obs |   coef_fb |     p_fb |   coef_yt |     p_yt |
|:-------------|--------:|----------:|---------:|----------:|---------:|
| ANTM         |     326 |  0.000433 | 0.246487 |  0.000127 | 0.124053 |
| ASII         |     543 | -2.5e-05  | 0.850804 | -0.000703 | 0.110343 |
| BBCA         |     613 | -0.000173 | 0.220941 |  0.000165 | 0.082345 |
| BBNI         |     717 | -0.000179 | 0.176974 | -0.00014  | 0.038523 |
| BBRI         |     708 | -8.7e-05  | 0.224998 |  4.9e-05  | 0.617957 |
| BBTN         |     618 | -7e-05    | 0.627748 |  0.000638 | 0.394655 |
| BMRI         |     622 | -2.6e-05  | 0.713997 |  1e-06    | 0.991824 |
| EXCL         |     609 | -0.000116 | 0.2135   | -0.000296 | 0.365501 |
| INCO         |     352 |  0.000208 | 0.669044 |  0.000122 | 0.836419 |
| INDF         |     684 |  1.6e-05  | 0.835447 |  0.000163 | 0.49984  |
| ITMG         |     269 |  0.000156 | 0.509794 |  0.000193 | 0.952128 |
| KLBF         |     634 | -5.7e-05  | 0.633208 | -8.1e-05  | 0.927889 |

## 4) Social engagement benchmark layer (`social-media-analyzer` style)
- FB engagement proxy mean: **23.3942%**, median: **5.6149%** (formula: (likes+comments+shares)/views_fb * 100).
- Benchmark reference from skill: Facebook average ~0.07%, good ~0.5–1.0%, excellent >1.0%.

## 5) High-attention regime asymmetry
- Next-day residual mean when social attention in top decile: **0.000454** vs others **-0.000050**.
- Welch t-test: t = **0.4801**, p = **0.632315**.

## 6) Predictive uplift from social features
- RF baseline RMSE (no social features): **0.009773**
- RF + social RMSE: **0.009266**
- RMSE improvement (positive is better): **0.000508**

## Bottom line
- Two-way FE and lagged/interaction extensions are included for deeper causal structure checks.
- Diagnostics, heterogeneity, and regime tests provide a stronger robustness layer beyond the initial phases.
- Predictive uplift quantifies incremental value of social signals over baseline market controls.