# LPA_PROFILES_REPORT

## 1) FinBERT scoring
- Model: `ProsusAI/finbert`
- Score definition: `P(positive) - P(negative)` in [-1, +1]
- Daily score built by averaging article-level FinBERT scores by company/date.
- Text rows processed: **23**

## 2) Latent Profile Analysis (GMM)
- Method: Gaussian Mixture Model (`n_components=8`) on standardized sentiment/engagement/market-state features.
- Profile ID assigned by maximum posterior probability.
- Output dataset: `C:\Tugas Akhir\research\phase6_lpa_enriched.csv`

## 3) Profile characteristics
|   profile_id |   n_obs |   mean_finbert_score |   mean_news_items_day |   mean_abs_return |   mean_volatility_20d |   mean_vix_ret | profile_label       |
|-------------:|--------:|---------------------:|----------------------:|------------------:|----------------------:|---------------:|:--------------------|
|            1 |     523 |             0        |                     0 |          0.016325 |              0.021113 |      -0.002411 | Neutral-Mixed       |
|            2 |    6463 |             0        |                     0 |          0.011014 |              0.019808 |      -0.003479 | Neutral-Mixed       |
|            3 |     690 |             0        |                     0 |          0.022208 |              0.024467 |       0.043467 | Neutral-Mixed       |
|            4 |    1681 |             0        |                     0 |          0.040243 |              0.037348 |       0.001513 | Neutral-Mixed       |
|            5 |    1389 |             0        |                     0 |          0.014502 |              0.017473 |      -0.004941 | Neutral-Mixed       |
|            6 |       4 |             0.657192 |                     1 |          0.039046 |              0.029492 |       0.016895 | Optimistic-Volatile |
|            7 |       2 |            -0.909694 |                     1 |          0.002984 |              0.015563 |      -0.013343 | Pessimistic-Calm    |
|            8 |       8 |             0.086077 |                     1 |          0.009365 |              0.025944 |      -0.003757 | Neutral-Mixed       |

## 4) Mean absolute return by profile
|   profile_id |   mean_abs_return |
|-------------:|------------------:|
|            4 |          0.040243 |
|            6 |          0.039046 |
|            3 |          0.022208 |
|            1 |          0.016325 |
|            5 |          0.014502 |
|            2 |          0.011014 |
|            8 |          0.009365 |
|            7 |          0.002984 |

## 5) Notes
- Profile labels are heuristic descriptors based on sentiment and volatility profile means.
- These profile IDs are ready for one-hot encoding in the LSTM phase.

**STOP POINT:** Awaiting approval before Phase 2 (macro-enhanced LSTM volatility model).