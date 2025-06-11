import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the delta_processed.csv file
df = pd.read_csv("delta_processed.csv")

# Reshape from wide to long format for averaging
df_long = df.melt(
    id_vars=["pnum", "condition", "mode"],
    value_vars=["p10", "p30", "p50", "p70", "p90"],
    var_name="percentile",
    value_name="rt"
)

# Convert 'p10' to 10, 'p30' to 30, etc.
df_long["percentile"] = df_long["percentile"].str.extract("(\d+)").astype(int)

# Average across participants
grouped = df_long.groupby(["condition", "mode", "percentile"])["rt"].mean().reset_index()

# Set up plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

CONDITION_NAMES = {
    0: "Easy Simple",
    1: "Easy Complex",
    2: "Hard Simple",
    3: "Hard Complex"
}
colors = {"overall": "black", "accurate": "green", "error": "red"}

for i, cond in enumerate(sorted(grouped["condition"].unique())):
    ax = axes[i]
    for mode in ["overall", "accurate", "error"]:
        sub = grouped[(grouped["condition"] == cond) & (grouped["mode"] == mode)]
        if not sub.empty:
            ax.plot(sub["percentile"], sub["rt"], label=mode, color=colors[mode], marker='o')
    
    ax.set_title(CONDITION_NAMES[cond])
    ax.set_xlabel("RT Percentile (%)")
    ax.set_ylabel("Mean RT (s)")
    ax.axhline(0, linestyle="--", color="gray")
    ax.legend()

plt.tight_layout()
plt.savefig("population_delta_plots.png", dpi=300)
plt.show()
