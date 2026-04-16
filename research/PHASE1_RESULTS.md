# Phase 1 — Panel Fixed Effects Results

## Model specification
- Panel index: `id_company` (entity) and `tanggal_stata` (time)
- Estimator: **Entity Fixed Effects (PanelOLS)** with clustered SE by entity
- Main regressors: `composite_index_fb`, `pca_index_yt`
- Control variable: `vol`

## 1) Dependent variable: `residual` (primary)
- Observations used: **10145**
- R-squared (within): **0.000238**
- F-stat (robust): **0.8029**, p-value **0.492073**

|                    |     coef |   std_err |    t_stat |   p_value |
|:-------------------|---------:|----------:|----------:|----------:|
| composite_index_fb | -1.9e-05 |   2.9e-05 | -0.648962 |  0.516378 |
| pca_index_yt       | -9e-06   |   3.2e-05 | -0.285549 |  0.775229 |
| vol                | -0       |   0       | -0.935096 |  0.349761 |

- Significance check: `composite_index_fb` = **ns**, `pca_index_yt` = **ns**.

## 2) Dependent variable: `return` (robustness)
- Observations used: **10154**
- R-squared (within): **0.014353**
- F-stat (robust): **49.1763**, p-value **0**

|                    |   coef |   std_err |    t_stat |   p_value |
|:-------------------|-------:|----------:|----------:|----------:|
| composite_index_fb | -2e-05 |   2.4e-05 | -0.827294 |   0.40809 |
| pca_index_yt       | -4e-05 |   2.8e-05 | -1.41391  |   0.15742 |
| vol                |  0     |   0       |  5.2637   |   0       |

- Significance check: `composite_index_fb` = **ns**, `pca_index_yt` = **ns**.

## Interpretation
- This phase tests **contemporaneous** social-media effects on stock performance after controlling for company fixed effects and trading activity (`vol`).
- Statistical significance is based on p-values in the tables above.