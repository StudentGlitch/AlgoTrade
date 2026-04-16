# OPENBB Data Expansion Report

## Objective
Expand the research dataset with newer market data, analysis features, macro context, and news sentiment proxies to support stronger predictive modeling.

## Data collection
- Source platform: **OpenBB** (`yfinance` provider where no API key is required)
- Start date: **2024-01-01**
- End date: **2026-04-15**
- Target symbols attempted: **20**
- Symbols fetched successfully: **20**
- Failed symbols: **0**

## Output dataset
- File: `C:\Tugas Akhir\research\openbb_enriched_stock_data_2024_onward.csv`
- Shape: **10760 rows x 35 columns**
- Date coverage: **2024-01-02 to 2026-04-15** (538 trading dates)

## Added feature groups
1. Price/return features: return, log_return, abs_return, range_pct
2. Momentum/technical: SMA(5/20), SMA ratio, momentum(5/20), RSI14
3. Volatility/liquidity: volatility_5d, volatility_20d, volume_z20, turnover_proxy
4. Macro context: JKSE, VIX, USD/IDR, oil, gold (+ daily returns)
5. News context: per-symbol `news_count`, `news_sentiment_mean`, `news_sentiment_std`

## Coverage by company
| company   |   rows |
|:----------|-------:|
| ADRO      |    538 |
| ANTM      |    538 |
| ASII      |    538 |
| BBCA      |    538 |
| BBNI      |    538 |
| BBRI      |    538 |
| BBTN      |    538 |
| BMRI      |    538 |
| EXCL      |    538 |
| INCO      |    538 |
| INDF      |    538 |
| ITMG      |    538 |
| KLBF      |    538 |
| PGAS      |    538 |
| PTBA      |    538 |
| SMGR      |    538 |
| TLKM      |    538 |
| TPIA      |    538 |
| UNTR      |    538 |
| UNVR      |    538 |

## Notes for predictive modeling
- This dataset is ready for chronological train/validation/test workflows.
- `abs_return` and rolling volatility features are suitable volatility targets.
- Macro and news columns can be tested as exogenous predictors.

## Column inventory
- Macro columns: `jkse_close, jkse_close_ret, vix_close, vix_close_ret, usd_idr_close, usd_idr_close_ret, oil_close, oil_close_ret, gold_close, gold_close_ret`
- Model-ready fields count (excluding id/date): **32**