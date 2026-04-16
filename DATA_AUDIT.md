# DATA_AUDIT

## 1) Dataset structure
- **Path:** `C:\Tugas Akhir\Full Data.csv`
- **Shape:** 10154 rows x 25 columns
- **Date column used:** `tanggal_stata` (2021-01-04 00:00:00 to 2023-12-29 00:00:00)
- **Unique companies (`id_company`):** 20

## 2) Descriptive statistics (key financial + social metrics)
|                    |   count |            mean |              std |        min |           25% |            50% |             75% |              max |
|:-------------------|--------:|----------------:|-----------------:|-----------:|--------------:|---------------:|----------------:|-----------------:|
| return             |   10154 |     0.0004      |      0.0199      |    -0.1492 |  -0.0101      |    0           |     0.0098      |      0.25        |
| residual           |   10145 |     0.0001      |      0.0192      |    -0.1621 |  -0.01        |   -0.0006      |     0.0092      |      0.1886      |
| vol                |   10154 |     5.32327e+07 |      7.06518e+07 | 68400      |   1.22358e+07 |    3.55004e+07 |     7.14758e+07 |      1.98321e+09 |
| turnover           |   10154 |     0.0018      |      0.0026      |     0.0001 |   0.0007      |    0.0011      |     0.0019      |      0.0825      |
| composite_index_fb |   10154 |     3.9464      |      7.5784      |     0      |   0.36        |    1.405       |     4.12        |    100           |
| pca_index_yt       |   10154 |     0.8361      |      5.5426      |     0      |   0           |    0           |     0           |    120.73        |
| views_fb           |   10154 | 65058           | 377443           |     0      | 128           | 1128           | 25250           |      1.31166e+07 |
| views_yt           |   10154 | 87367           |      1.0554e+06  |     0      |   0           |    0           |     0           |      4.60111e+07 |
| likes_fb           |   10154 |   501.735       |   2394.2         |     0      |  18           |   72           |   316.75        | 113078           |
| shares_fb          |   10154 |    23.4036      |     86.7968      |     0      |   0           |    3           |    10           |   2847           |
| comments_fb        |   10154 |   235.214       |   1561.95        |     0      |   0           |    3           |    51           |  44254           |
| likes_yt           |   10154 |    92.926       |   1846.96        |     0      |   0           |    0           |     0           | 125000           |
| comments_yt        |   10154 |    11.0627      |    132.677       |     0      |   0           |    0           |     0           |   9431           |

## 3) Data quality issues
### Missing values (non-zero only)
|          |   missing_count |   missing_pct |
|:---------|----------------:|--------------:|
| date     |            6163 |         60.7  |
| residual |               9 |          0.09 |

### Zero-value concentration in social metrics
| column             |   zero_count |   non_na_count |   zero_pct |
|:-------------------|-------------:|---------------:|-----------:|
| composite_index_fb |          264 |          10154 |       2.6  |
| pca_index_yt       |         8734 |          10154 |      86.02 |
| views_fb           |          213 |          10154 |       2.1  |
| views_yt           |         8671 |          10154 |      85.39 |
| likes_fb           |          273 |          10154 |       2.69 |
| shares_fb          |         2800 |          10154 |      27.58 |
| comments_fb        |         3126 |          10154 |      30.79 |
| likes_yt           |         8676 |          10154 |      85.44 |
| comments_yt        |         9208 |          10154 |      90.68 |

- **Potential issue:** substantial zero concentration (>=30%) found in some social metrics, which may indicate sparse posting periods, true inactivity, or data collection gaps.

## 4) Correlation heatmap observations
- Heatmap saved to: `C:\Tugas Akhir\correlation_heatmap.png`
- Correlation matrix:
|                    |   composite_index_fb |   pca_index_yt |   return |   residual |     vol |   turnover |
|:-------------------|---------------------:|---------------:|---------:|-----------:|--------:|-----------:|
| composite_index_fb |               1      |         0.0251 |  -0.0038 |    -0.0095 |  0.0374 |     0.0179 |
| pca_index_yt       |               0.0251 |         1      |  -0.0123 |    -0.0033 |  0.0313 |    -0.0086 |
| return             |              -0.0038 |        -0.0123 |   1      |    -0.0449 |  0.1031 |     0.127  |
| residual           |              -0.0095 |        -0.0033 |  -0.0449 |     1      | -0.0118 |    -0.0057 |
| vol                |               0.0374 |         0.0313 |   0.1031 |    -0.0118 |  1      |     0.6949 |
| turnover           |               0.0179 |        -0.0086 |   0.127  |    -0.0057 |  0.6949 |     1      |
- **FB vs YT multicollinearity check:** Correlation(composite_index_fb, pca_index_yt) = 0.025 (low).
- Strongest absolute pairwise correlation: `vol` vs `turnover` = 0.695.

## 5) Stationarity pre-check (ADF, autolag=AIC)
| series             |   n_obs |   adf_stat |   p_value | assessment   |
|:-------------------|--------:|-----------:|----------:|:-------------|
| return             |     732 |   -29.1382 |    0      | Stationary   |
| residual           |     732 |   -29.2972 |    0      | Stationary   |
| composite_index_fb |     732 |    -3.4719 |    0.0087 | Stationary   |
| pca_index_yt       |     732 |    -6.5794 |    0      | Stationary   |
- Interpretation rule: p-value < 0.05 suggests stationarity.

## 6) Phase 0 conclusion
- Phase 0 audit completed. This document summarizes structure, quality checks, descriptive statistics, correlations, and stationarity pre-checks.
- **STOP POINT:** Awaiting approval before proceeding to Phase 1 (Panel Fixed Effects model).