import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the processed delta data
df = pd.read_csv("delta_processed.csv")

# Melt the data to long format
df_melted = df.melt(
    id_vars=["pnum", "condition", "mode"],
    value_vars=["p10", "p30", "p50", "p70", "p90"],
    var_name="percentile",
    value_name="rt"
)

# Convert percentile strings to numeric
df_melted["percentile"] = df_melted["percentile"].str.extract(r'(\d+)').astype(int)

# Map condition to difficulty and stimulus
df_melted["difficulty"] = df_melted["condition"].map({0: "Easy", 1: "Easy", 2: "Hard", 3: "Hard"})
df_melted["stimulus"] = df_melted["condition"].map({0: "Simple", 1: "Complex", 2: "Simple", 3: "Complex"})

# Population-level delta plots
# 1. Hard vs Easy (within Simple and Complex)
delta_diff = (
    df_melted
    .pivot_table(index=["pnum", "percentile", "stimulus"], columns="difficulty", values="rt")
    .reset_index()
)
delta_diff["delta"] = delta_diff["Hard"] - delta_diff["Easy"]

# 2. Complex vs Simple (within Easy and Hard)
delta_stim = (
    df_melted
    .pivot_table(index=["pnum", "percentile", "difficulty"], columns="stimulus", values="rt")
    .reset_index()
)
delta_stim["delta"] = delta_stim["Complex"] - delta_stim["Simple"]

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot delta by stimulus (Simple vs Complex) within difficulty
sns.lineplot(
    data=delta_diff,
    x="percentile", y="delta", hue="stimulus",
    marker="o", ax=axs[0]
)
axs[0].set_title("Delta Plot: Hard − Easy RT\nGrouped by Stimulus Type")
axs[0].set_xlabel("RT Percentile (%)")
axs[0].set_ylabel("RT Difference (Hard − Easy)")
axs[0].axhline(0, color='gray', linestyle='--')

# Plot delta by difficulty (Easy vs Hard) within stimulus
sns.lineplot(
    data=delta_stim,
    x="percentile", y="delta", hue="difficulty",
    marker="o", ax=axs[1]
)
axs[1].set_title("Delta Plot: Complex − Simple RT\nGrouped by Trial Difficulty")
axs[1].set_xlabel("RT Percentile (%)")
axs[1].set_ylabel("RT Difference (Complex − Simple)")
axs[1].axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig("population_delta_plots.png", dpi=300)
plt.show()

#Created with the help of AI