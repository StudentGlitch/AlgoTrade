import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = r"C:\Tugas Akhir\research\phase4_preprocessed.csv"
OUT_MD = r"C:\Tugas Akhir\EVENT_STUDY_RESULTS.md"
OUT_PLOT = r"C:\Tugas Akhir\research\event_study_cav_plot.png"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["tanggal_stata"] = pd.to_datetime(df["tanggal_stata"], errors="coerce")
    df = df.dropna(subset=["id_company", "tanggal_stata", "absolute_return"]).copy()
    df = df.sort_values(["id_company", "tanggal_stata"])

    # Abnormal volatility proxy = absolute_return - company mean absolute_return
    company_mean_abs = df.groupby("id_company")["absolute_return"].transform("mean")
    df["abnormal_volatility"] = df["absolute_return"] - company_mean_abs

    window = list(range(-2, 6))  # [-2, +5]
    rows = []
    event_count = 0

    for cid, g in df.groupby("id_company"):
        g = g.reset_index(drop=True)
        event_idx = g.index[g["yt_spike"] == 1].tolist()
        for eidx in event_idx:
            if eidx - 2 < 0 or eidx + 5 >= len(g):
                continue
            event_count += 1
            for k in window:
                rows.append(
                    {
                        "id_company": cid,
                        "event_idx": eidx,
                        "tau": k,
                        "av": float(g.loc[eidx + k, "abnormal_volatility"]),
                    }
                )

    ev = pd.DataFrame(rows)
    if ev.empty:
        raise ValueError("No valid events with complete [-2,+5] window found.")

    avg_av = ev.groupby("tau", as_index=True)["av"].mean().reindex(window)
    avg_cav = avg_av.cumsum()

    # Plot average CAV
    plt.figure(figsize=(9, 5))
    plt.plot(avg_cav.index, avg_cav.values, marker="o", linewidth=2)
    plt.axvline(0, color="red", linestyle="--", linewidth=1, label="Event day")
    plt.axhline(0, color="gray", linestyle="-", linewidth=1)
    plt.title("Average CAV around YouTube Spike Events (tau = -2..+5)")
    plt.xlabel("Event time (tau)")
    plt.ylabel("Average CAV (abnormal volatility)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)
    plt.close()

    summary = pd.DataFrame({"avg_av": avg_av, "avg_cav": avg_cav}).round(6)

    lines = []
    lines.append("# EVENT_STUDY_RESULTS — Phase 1")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Event definition: `yt_spike == 1`")
    lines.append("- Window: **[-2, +5]** around event day")
    lines.append("- Metric: **CAV** (Cumulative Abnormal Volatility)")
    lines.append("- Abnormal volatility: `absolute_return - company_mean(absolute_return)`")
    lines.append(f"- Valid event windows used: **{event_count}**")
    lines.append("")
    lines.append("## Average event profile")
    lines.append(summary.to_markdown())
    lines.append("")
    lines.append("## Plot")
    lines.append(f"- Saved plot: `{OUT_PLOT}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- `avg_av` gives mean abnormal volatility at each event-time tau.")
    lines.append("- `avg_cav` accumulates abnormal volatility from tau=-2 to each tau.")
    lines.append("- This isolates volatility dynamics specifically around YouTube spike days.")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WROTE={OUT_MD}")
    print(f"WROTE={OUT_PLOT}")
    print(f"EVENTS={event_count}")


if __name__ == "__main__":
    main()
