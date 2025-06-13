import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
PERCENTILES = [10, 30, 50, 70, 90]
OUTDIR = Path("delta_plots")
OUTDIR.mkdir(exist_ok=True)

# Load and preprocess data
df = pd.read_csv("data.csv")

# Encode categorical variables
df["stimulus_type"] = df["stimulus_type"].map({"simple": "Simple", "complex": "Complex"})
df["difficulty"] = df["difficulty"].map({"easy": "Easy", "hard": "Hard"})

# Helper function to get RT percentiles
def get_percentiles(condition_df):
    return [np.percentile(condition_df["rt"], p) for p in PERCENTILES]

# Loop through each participant
for pid in df["participant_id"].unique():
    pdf = df[df["participant_id"] == pid]
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Plot 1: Hard - Easy for each Stimulus Type
    for stim_type in ["Simple", "Complex"]:
        easy = pdf[(pdf["difficulty"] == "Easy") & (pdf["stimulus_type"] == stim_type)]
        hard = pdf[(pdf["difficulty"] == "Hard") & (pdf["stimulus_type"] == stim_type)]

        if not easy.empty and not hard.empty:
            easy_pcts = get_percentiles(easy)
            hard_pcts = get_percentiles(hard)
            delta = np.array(hard_pcts) - np.array(easy_pcts)
            axs[0].plot(PERCENTILES, delta, marker="o", label=f"{stim_type}")

    axs[0].axhline(0, linestyle="--", color="gray")
    axs[0].set_title("Hard - Easy RT Differences by Stimulus Type")
    axs[0].set_ylabel("ΔRT (s)")
    axs[0].legend()

    # Plot 2: Complex - Simple for each Difficulty
    for difficulty in ["Easy", "Hard"]:
        simple = pdf[(pdf["stimulus_type"] == "Simple") & (pdf["difficulty"] == difficulty)]
        complex_ = pdf[(pdf["stimulus_type"] == "Complex") & (pdf["difficulty"] == difficulty)]

        if not simple.empty and not complex_.empty:
            simple_pcts = get_percentiles(simple)
            complex_pcts = get_percentiles(complex_)
            delta = np.array(complex_pcts) - np.array(simple_pcts)
            axs[1].plot(PERCENTILES, delta, marker="o", label=f"{difficulty}")

    axs[1].axhline(0, linestyle="--", color="gray")
    axs[1].set_title("Complex - Simple RT Differences by Difficulty")
    axs[1].set_xlabel("RT Percentile")
    axs[1].set_ylabel("ΔRT (s)")
    axs[1].legend()

    plt.tight_layout()
    out_path = OUTDIR / f"participant{int(pid)}_delta_plots.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_path}")


#Created with the help of AI