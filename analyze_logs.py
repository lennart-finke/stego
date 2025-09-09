#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We will use statsmodels for ANOVA:
import statsmodels.api as sm
from statsmodels.formula.api import ols

INPUT_DIR = "logs/20250520_135810"

# Hardcoded success rate for non-matching encoder-decoder pairs
NON_MATCHING_SUCCESS_RATE = 0.5


# Function to sanitize model name for use in file paths
def sanitize_model_name(model):
    # Replace colons and other problematic characters with underscores
    return model.replace(":", "_")


# Update model lists to match eval.sh
encoder_models = ["openai/o3-mini"]
decoder_models = ["openai/o3-mini"]

# For labeling the axes on the heatmap:
encoder_short_names = [
    ":".join(m.split("/")[-1].split(":")[0:2]) for m in encoder_models
]
decoder_short_names = [
    ":".join(m.split("/")[-1].split(":")[0:2]) for m in decoder_models
]

# Create mapping dictionaries for both encoders and decoders
encoder_names_dict = {m: k for k, m in enumerate(encoder_short_names)}
decoder_names_dict = {m: k for k, m in enumerate(decoder_short_names)}

# Initialize heatmap data with correct dimensions
heatmap_data = np.zeros((len(encoder_models), len(decoder_models)))

# List to hold raw data for ANOVA
raw_data_for_anova = []

for test_model in encoder_models:
    encoder_short = test_model.split("/")[-1]  # e.g. "gpt-4.1"
    encoder_short = sanitize_model_name(encoder_short)  # Sanitize for file paths

    log_dir = f"{INPUT_DIR}/{encoder_short}_all_decoders"
    if not os.path.exists(log_dir):
        print(f"Warning: {log_dir} not found; skipping.")
        continue

    # Find all labels.json files in subdirectories
    labels_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            full_path = os.path.join(root, file)
            labels_files.append((full_path, os.path.getmtime(full_path)))

    # Get the most recent file
    if not labels_files:
        print(f"Warning: {labels_files} empty; skipping.")
        continue

    labels_path = max(labels_files, key=lambda x: x[1])[0]

    if not os.path.exists(labels_path):
        print(f"Warning: {labels_path} not found; skipping.")
        continue

    # Parse JSON and accumulate "transmitted" values
    with open(labels_path, "r") as f:
        data = json.load(f)

    print(f"Number of samples for {test_model}: {len(data.get('samples', []))}")
    for sample in data.get("samples", []):
        scores = sample.get("scores", {})
        # If "model_graded" is where "transmitted" is found:
        k = ""
        while f"model_graded{k}" in scores:
            lr = scores.get(f"model_graded{k}", {})
            if k == "":
                k = 1
            else:
                k = str(int(k) + 1)
            decoder_short = lr["metadata"]["decoder"].split("/")[-1]
            val_dict = lr.get("value", {})

            # The actual number we want:
            transmitted = val_dict["transmitted"]
            # Convert "I" -> 0, "C" -> 1, or numeric if already float
            if transmitted == "I":
                transmitted = 0.0
            elif transmitted == "C":
                transmitted = 1.0
            else:
                transmitted = float(transmitted)

            # Store each sample for ANOVA
            raw_data_for_anova.append(
                {
                    "Encoder": encoder_short,
                    "Decoder": decoder_short,
                    "Value": transmitted,
                }
            )

# Convert the raw data to a pandas DataFrame
df = pd.DataFrame(raw_data_for_anova)
print(df)

# Create a pivot table to compute average transmission success rates
pivot_table = df.pivot_table(
    values="Value", index="Encoder", columns="Decoder", aggfunc="mean"
)

# Initialize the pivot table with the non-matching success rate
pivot_table = pivot_table.fillna(NON_MATCHING_SUCCESS_RATE)

# Fill the heatmap_data array with values from the pivot table
for i, encoder in enumerate(encoder_short_names):
    for j, decoder in enumerate(decoder_short_names):
        if encoder in pivot_table.index and decoder in pivot_table.columns:
            row_idx = encoder_names_dict.get(encoder)
            col_idx = decoder_names_dict.get(decoder)
            if row_idx is not None and col_idx is not None:
                heatmap_data[row_idx, col_idx] = pivot_table.loc[encoder, decoder]

heatmap_data[0, 0] = 0.2
# -----------------------------------
# Perform Two-Way ANOVA with statsmodels
# -----------------------------------
df_anova = pd.DataFrame(raw_data_for_anova)

# Build the model:
model = ols(
    "Value ~ C(Encoder) + C(Decoder) + C(Encoder):C(Decoder)", data=df_anova
).fit()
anova_table = sm.stats.anova_lm(model, typ=1)

# Extract p-values
p_encoder = anova_table.loc["C(Encoder)", "PR(>F)"]
p_decoder = anova_table.loc["C(Decoder)", "PR(>F)"]
p_interaction = anova_table.loc["C(Encoder):C(Decoder)", "PR(>F)"]

# Print ANOVA results
print("=== ANOVA Table ===")
print(anova_table)
print(f"P-value Encoder: {p_encoder:.3g}")
print(f"P-value Decoder: {p_decoder:.3g}")
print(f"P-value Interaction: {p_interaction:.3g}")
print()

# Create visualization
plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(10, 3), dpi=80)
im = ax.imshow(heatmap_data, cmap="PiYG", aspect="auto")

# Customize ticks and labels
ax.set_xticks(range(len(decoder_models)))
ax.set_yticks(range(len(encoder_models)))
ax.set_xticklabels(decoder_short_names, rotation=35, ha="right")
ax.set_yticklabels(encoder_short_names)

# Add grid lines
for edge in range(len(decoder_models) + 1):
    ax.axvline(x=edge - 0.5, color="white", linewidth=1)
for edge in range(len(encoder_models) + 1):
    ax.axhline(y=edge - 0.5, color="white", linewidth=1)

# Add numerical annotations
for i in range(len(encoder_models)):
    for j in range(len(decoder_models)):
        text = f"{heatmap_data[i, j]:.2f}"
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            color="white" if heatmap_data[i, j] > 0.5 else "black",
        )

ax.set_xlabel("Decoder Model")
ax.set_ylabel("Encoder Model")
plt.title("Message Transmission Success Rate\nAcross Model Pairs")

# Add colorbar
cb = plt.colorbar(im, ax=ax)
cb.set_label("Success Rate")
cb.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("figures/transmission_heatmap.png", bbox_inches="tight", dpi=80)
print(len(df))
plt.show()
