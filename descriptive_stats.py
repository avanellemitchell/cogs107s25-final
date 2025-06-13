import pandas as pd

# Load the data
df = pd.read_csv("data.csv")

# Convert to categorical for grouping
df["difficulty"] = df["difficulty"].astype("category")
df["stimulus_type"] = df["stimulus_type"].astype("category")
df["signal"] = df["signal"].astype("category")

# Compute descriptive statistics grouped by condition
grouped = df.groupby(["difficulty", "stimulus_type", "signal"])

# Aggregate statistics
stats = grouped.agg(
    mean_rt=("rt", "mean"),
    std_rt=("rt", "std"),
    median_rt=("rt", "median"),
    accuracy_mean=("accuracy", "mean"),
    n_trials=("rt", "count")
).reset_index()

# Print it
print(stats)

# Optionally: save it to a CSV for reference
stats.to_csv("descriptive_statistics_by_condition.csv", index=False)
