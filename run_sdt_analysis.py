import arviz as az
from sdt_ddm import read_data, apply_hierarchical_sdt_model
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and process data for SDT
data = read_data("data.csv", prepare_for="sdt", display=True)

# Step 2: Build model
model = apply_hierarchical_sdt_model(data)

# Step 3: Sample from the posterior
with model:
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        return_inferencedata=True
    )

# Step 4: Save trace to disk
az.to_netcdf(trace, "sdt_trace.nc")

# Step 5: Print convergence diagnostics
summary = az.summary(trace, var_names=[
    "mean_d_prime", "mean_criterion",
    "stdev_d_prime", "stdev_criterion"
])
print(summary)

# -----------------------
# Step 6: Plot Posterior Means for d′ and Criterion
# -----------------------

conditions = ["Easy\nSimple", "Easy\nComplex", "Hard\nSimple", "Hard\nComplex"]

# Extract means and standard deviations for plotting
means = az.summary(trace, var_names=["mean_d_prime", "mean_criterion"])

d_means = means.filter(like="mean_d_prime", axis=0)["mean"].values
d_err = means.filter(like="mean_d_prime", axis=0)["sd"].values

c_means = means.filter(like="mean_criterion", axis=0)["mean"].values
c_err = means.filter(like="mean_criterion", axis=0)["sd"].values

print("\nPosterior Means for d':")
for i, d in enumerate(d_means):
    print(f"Condition {i}: {d:.3f}")

print("\nPosterior Means for Criterion:")
for i, c in enumerate(c_means):
    print(f"Condition {i}: {c:.3f}")

# Plotting
x = np.arange(len(conditions))
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# d′ plot
axs[0].bar(x, d_means, yerr=d_err, color='skyblue', capsize=5)
axs[0].set_title("Mean d′ per Condition")
axs[0].set_xticks(x)
axs[0].set_xticklabels(conditions)
axs[0].set_ylabel("Sensitivity (d′)")

# Criterion plot
axs[1].bar(x, c_means, yerr=c_err, color='lightcoral', capsize=5)
axs[1].set_title("Mean Criterion per Condition")
axs[1].set_xticks(x)
axs[1].set_xticklabels(conditions)
axs[1].set_ylabel("Bias (Criterion)")

plt.tight_layout()
plt.savefig("sdt_summary_plots.png", dpi=300)
plt.show()

#Created with the help of AI