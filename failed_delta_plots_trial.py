import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the delta_processed.csv (ensure this file is in the same directory)
df = pd.read_csv("delta_processed.csv")

# Define constants
PERCENTILES = [10, 30, 50, 70, 90]
CONDITION_NAMES = {
    0: "Easy Simple",
    1: "Easy Complex",
    2: "Hard Simple",
    3: "Hard Complex"
}

# Function to plot RTs for each condition per participant
def plot_participant_rt_curves(df, participant_id):
    participant_data = df[(df['pnum'] == participant_id) & (df['mode'] == 'overall')]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, condition in enumerate(sorted(participant_data['condition'].unique())):
        cond_data = participant_data[participant_data['condition'] == condition]
        if cond_data.empty:
            continue
        rts = [cond_data[f'p{p}'].values[0] for p in PERCENTILES]
        
        ax = axes[i]
        ax.plot(PERCENTILES, rts, marker='o', color='blue')
        ax.set_title(CONDITION_NAMES[condition])
        ax.set_xlabel("Percentile (%)")
        ax.set_ylabel("RT (s)")
        ax.grid(True)
    
    plt.suptitle(f"Participant {participant_id} RTs by Condition and Percentile", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"participant_{participant_id}_rt_curves.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.show()

# Run for all unique participants
if __name__ == "__main__":
    for pid in df["pnum"].unique():
        plot_participant_rt_curves(df, participant_id=pid)


#Created with the help of AI
