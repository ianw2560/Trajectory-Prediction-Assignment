#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path="metrics.csv"
out_path="ade_mean_bar.png"

# Read the CSV; index is the first column (args.output_name)
df = pd.read_csv(csv_path, index_col=0)

# Get x labels (row index) and ade_mean values
x_labels = df.index.to_list()
ade_values = df["ade_mean"].values

# Make bar chart
fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(ade_values))
ax.bar(x, ade_values)  # default matplotlib colors

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("ADE Mean")
ax.set_xlabel("Configuration")
ax.set_title("ADE Mean per Configuration")
fig.tight_layout()

# Save to file (and/or show)
fig.savefig(out_path, dpi=300)
