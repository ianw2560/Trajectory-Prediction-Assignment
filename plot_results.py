#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_min_ade_all(metric, csv_path="metrics.csv", out_path="images/ade_mean_2x2.png"):
    os.makedirs("images", exist_ok=True)

    # Read the CSV file, index is the first column
    df = pd.read_csv(csv_path, index_col=0)

    idx = df.index

    # Figure out the base model names in the order they appear
    base_order = []
    for name in idx:
        base = name.split("_")[0]
        if base not in base_order:
            base_order.append(base)

    def group_df(mask):
        """Return df for a mask, sorted by base model order."""
        sub = df[mask].copy()
        names_sorted = sorted(sub.index, key=lambda n: base_order.index(n.split("_")[0]))
        return sub.loc[names_sorted]

    # Masks for each group
    base_mask    = ~idx.str.contains("_")
    dc_mask      = idx.str.endswith("_dc") & ~idx.str.endswith("_fnl_dc")
    fnl_mask     = idx.str.endswith("_fnl") & ~idx.str.endswith("_fnl_dc")
    fnl_dc_mask  = idx.str.endswith("_fnl_dc")

    groups = [
        ("No Dynamic Clustering or Fully Non-linear", group_df(base_mask)),
        ("Dynamic Clustering", group_df(dc_mask)),
        ("Fully Non-linear", group_df(fnl_mask)),
        ("Fully Non-linear + Dynamic Clustering", group_df(fnl_dc_mask)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

    for ax, (title, sub) in zip(axes.flat, groups):
        if sub.empty:
            ax.set_visible(False)
            continue

        x_labels = sub.index.to_list()
        ade_values = sub[metric].values

        # Index of the smallest value within this group
        min_idx = ade_values.argmin()

        # Colors (highlight min)
        colors = ["C0"] * len(ade_values)
        colors[min_idx] = "C3"

        x = range(len(ade_values))
        bars = ax.bar(x, ade_values, color=colors)

        ax.bar_label(bars, fmt='{:.3f}')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        if metric == "ade_mean":
            ax.set_ylabel("Average minADE")
        elif metric == "fde_mean":
            ax.set_ylabel("Average minFDE")
        else:
            raise ValueError

        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

plot_min_ade_all("ade_mean", csv_path="metrics_sorted.csv", out_path="images/ade_mean_2x2.png")
plot_min_ade_all("fde_mean", csv_path="metrics_sorted.csv", out_path="images/fde_mean_2x2.png")