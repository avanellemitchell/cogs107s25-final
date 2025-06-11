import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Define constants
PERCENTILES = [10, 30, 50, 70, 90]

# Condition index to readable label
CONDITION_NAMES = {
    0: "Easy Simple",
    1: "Easy Complex",
    2: "Hard Simple",
    3: "Hard Complex"
}

# -----------------------------
# Step 1: Prepare the data
# -----------------------------
def prepare_data_for_delta_plots(file_path):
    data = pd.read_csv(file_path)

    # Convert string labels to numeric codes
    mappings = {
        "stimulus_type": {"simple": 0, "complex": 1},
        "difficulty": {"easy": 0, "hard": 1},
        "signal": {"present": 0, "absent": 1}
    }

    for col, mapping in mappings.items():
        data[col] = data[col].map(mapping)

    # Combine into condition index
    data["pnum"] = data["participant_id"]
    data["condition"] = data["stimulus_type"] + data["difficulty"] * 2

    rows = []

    for p in data["pnum"].unique():
        for c in data["condition"].unique():
            sub = data[(data["pnum"] == p) & (data["condition"] == c)]

            if len(sub) < 10:
                continue

            for mode, df in [("overall", sub),
                             ("accurate", sub[sub["accuracy"] == 1]),
                             ("error", sub[sub["accuracy"] == 0])]:

                row = {"pnum": p, "condition": c, "mode": mode}
                for pctl in PERCENTILES:
                    row[f"p{pctl}"] = np.percentile(df["rt"], pctl) if len(df) > 0 else np.nan

                rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# Step 2: Draw delta plots
# -----------------------------
def draw_delta_plots(data, pnum):
    data = data[data['pnum'] == pnum]
    conditions = sorted(data['condition'].unique())
    n = len(conditions)

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            ax = axes[i, j]
            if i == j:
                ax.axis("off")
                continue

            if i < j:
                mode = "overall"
            else:
                mode = None  # lower triangle = both error & accurate

            for label, color in [("accurate", "green"), ("error", "red")]:
                if mode or label != "accurate":
                    continue

                d1 = data[(data["condition"] == c1) & (data["mode"] == label)]
                d2 = data[(data["condition"] == c2) & (data["mode"] == label)]

                if d1.empty or d2.empty:
                    continue

                q1 = [d1[f"p{p}"].values[0] for p in PERCENTILES]
                q2 = [d2[f"p{p}"].values[0] for p in PERCENTILES]
                delta = np.array(q2) - np.array(q1)
                ax.plot(PERCENTILES, delta, label=label, color=color, marker='o')

            if mode == "overall":
                d1 = data[(data["condition"] == c1) & (data["mode"] == "overall")]
                d2 = data[(data["condition"] == c2) & (data["mode"] == "overall")]
                if not d1.empty and not d2.empty:
                    q1 = [d1[f"p{p}"].values[0] for p in PERCENTILES]
                    q2 = [d2[f"p{p}"].values[0] for p in PERCENTILES]
                    delta = np.array(q2) - np.array(q1)
                    ax.plot(PERCENTILES, delta, label="overall", color="black", marker='o')

            ax.axhline(0, linestyle="--", color="gray")
            ax.set_xticks(PERCENTILES)
            ax.set_xlabel("Percentile (%)")
            ax.set_ylabel("RT Δ (s)")

            if i == n - 1:
                ax.set_title(f"{CONDITION_NAMES[c2]}")
            if j == 0:
                ax.set_ylabel(f"{CONDITION_NAMES[c1]}\nRT Δ (s)")

            if i > j:
                ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    out_path = Path("delta_plots_participant_{}.png".format(pnum))
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved delta plot to {out_path}")
    plt.show()

# -----------------------------
# Main block
# -----------------------------
if __name__ == "__main__":
    df = prepare_data_for_delta_plots("data.csv")
    df.to_csv("delta_processed.csv", index=False)
    print("✅ Processed delta plot data saved.")

    draw_delta_plots(df, pnum=1)
