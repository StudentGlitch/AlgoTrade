# Phase 2 — Causality & Time Dynamics (VAR + Granger)

## Setup
- Time-series input is built as **daily aggregated means** across companies.
- Variables in VAR: `return`, `composite_index_fb`, `pca_index_yt`
- Lag length selected using BIC: **1**
- Sample size for VAR: **732 daily observations**

## Granger causality
- Test 1: `composite_index_fb` Granger-causes `return` -> min p-value **0.232956** (best lag 1)
- Test 2: `return` Granger-causes `composite_index_fb` -> min p-value **0.114921** (best lag 1)
- Rule: p-value < 0.05 indicates evidence of Granger causality.

## VAR coefficient snapshot (`return` equation)
|                       |      coef |
|:----------------------|----------:|
| const                 |  0.001174 |
| L1.return             | -0.07817  |
| L1.composite_index_fb | -0.000177 |
| L1.pca_index_yt       | -0.000189 |

## Event study (spikes in `views_fb` / `views_yt`)
- Spike threshold: top 1% (99th percentile) for each views metric.
- Number of event dates: **16**
- Average CAR[-1,+1] (using `residual`): **-0.000051**
- Median CAR[-1,+1] (using `residual`): **0.002341**

| date                | event_type   |    car_m1_p1 |
|:--------------------|:-------------|-------------:|
| 2021-06-04 00:00:00 | YT           | -0.0256473   |
| 2021-11-19 00:00:00 | YT           |  0.0211599   |
| 2022-04-20 00:00:00 | YT           |  0.0245818   |
| 2022-12-16 00:00:00 | YT           |  0.00527118  |
| 2023-01-03 00:00:00 | YT           | -0.0257685   |
| 2023-03-24 00:00:00 | YT           |  0.0167877   |
| 2023-04-17 00:00:00 | FB           |  0.0153961   |
| 2023-05-02 00:00:00 | FB           | -0.0259297   |
| 2023-05-15 00:00:00 | FB           | -0.0111196   |
| 2023-05-29 00:00:00 | FB           |  0.000280594 |
| 2023-06-12 00:00:00 | FB           | -0.000935959 |
| 2023-06-26 00:00:00 | FB           |  0.0105642   |
| 2023-07-24 00:00:00 | FB           |  0.00744476  |
| 2023-09-20 00:00:00 | YT           |  0.00440157  |
| 2023-12-08 00:00:00 | YT           | -0.0106703   |

## Interpretation
- Granger results quantify directional predictability, not structural causality.
- Event-study CAR summarizes short-window abnormal return response around extreme social-attention spikes.