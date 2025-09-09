#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

log_path = "logs/varied_length.json"

with open(log_path, "r") as f:
    data = json.load(f)

records = []
for sample in data.get("samples", []):
    target: str = sample["target"][0]
    length = len(target)  # The length of the target string

    scores = sample.get("scores", {})
    graded = scores.get("model_graded", {})
    val_dict = graded.get("value", {})

    transmitted = val_dict.get("transmitted", 0)
    if transmitted == "I":
        transmitted = 0.0
    elif transmitted == "C":
        transmitted = 1.0
    else:
        # In case the JSON logs have it as numeric string
        transmitted = float(transmitted)

    records.append((length, transmitted))

# Create a DataFrame
df = pd.DataFrame(records, columns=["length", "transmitted"])

# If you know the data only contains target lengths {4, 8, 16, 32, 64, 128},
# you could directly group by length. However, if there's any noise
# or small deviations, bin them explicitly with pd.cut:

bins = [0, 6, 12, 24, 48, 96, 192, 384, 10_000]
labels = ["4", "8", "16", "32", "64", "128", "256", "512"]
df["binned_length"] = pd.cut(df["length"], bins=bins, labels=labels)

# Assert equal group sizes
group_sizes = df.groupby("binned_length").size()
try:
    assert (
        group_sizes == group_sizes.iloc[0]
    ).all(), "Not all bins have the same number of samples!"
except AssertionError:
    print("Group sizes are not equal:")
    print(group_sizes)
    raise

# Compute statistics by bin
grouped = df.groupby("binned_length")["transmitted"]
stats = grouped.agg(["mean", "std", "count"])
# Standard error of the mean

# Plot
x = np.arange(len(labels))
plt.figure(figsize=(8, 5))

plt.bar(x, stats["mean"], capsize=5, color="skyblue", edgecolor="black")

plt.xticks(x, labels)
plt.xlabel("Binned Length of Target String")
plt.ylabel("Mean Transmitted (Success = 1, Failure = 0)")
plt.title("Transmission Success Rate by Target Length Bin")
plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()
