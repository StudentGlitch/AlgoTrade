# EVENT_STUDY_RESULTS — Phase 1

## Setup
- Event definition: `yt_spike == 1`
- Window: **[-2, +5]** around event day
- Metric: **CAV** (Cumulative Abnormal Volatility)
- Abnormal volatility: `absolute_return - company_mean(absolute_return)`
- Valid event windows used: **70**

## Average event profile
|   tau |    avg_av |   avg_cav |
|------:|----------:|----------:|
|    -2 |  0.000893 |  0.000893 |
|    -1 |  0.000601 |  0.001494 |
|     0 | -0.001105 |  0.00039  |
|     1 | -0.000197 |  0.000193 |
|     2 | -0.000419 | -0.000227 |
|     3 |  0.000605 |  0.000378 |
|     4 | -0.001301 | -0.000922 |
|     5 |  0.000854 | -6.8e-05  |

## Plot
- Saved plot: `C:\Tugas Akhir\research\event_study_cav_plot.png`

## Interpretation
- `avg_av` gives mean abnormal volatility at each event-time tau.
- `avg_cav` accumulates abnormal volatility from tau=-2 to each tau.
- This isolates volatility dynamics specifically around YouTube spike days.