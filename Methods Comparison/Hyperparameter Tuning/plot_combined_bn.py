"""
Generate a single combined loss curve plot from bn_runs/epoch_losses.csv.
Run from the Hyperparameter Tuning folder:
    python plot_combined_bn.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

here    = os.path.dirname(os.path.abspath(__file__))
csv_in  = os.path.join(here, "bn_runs", "epoch_losses.csv")
out_png = os.path.join(here, "bn_runs", "combined_loss_curves.png")

df = pd.read_csv(csv_in)

m_values = sorted(df["m"].unique().tolist())

fig, ax = plt.subplots(figsize=(11, 6))

colors = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#F97316"]

for m, color in zip(m_values, colors):
    subset = df[df["m"] == m].sort_values("epoch")
    label = f"m={int(m)}"

    ax.plot(
        subset["epoch"], subset["train_loss"],
        color=color, linewidth=1.8, linestyle="-",
        marker="o", markersize=3,
        label=f"{label}  train",
    )
    ax.plot(
        subset["epoch"], subset["val_loss"],
        color=color, linewidth=1.8, linestyle="--",
        marker="s", markersize=3,
        label=f"{label}  val",
    )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)  
ax.set_title(
    "Training & Validation Loss — Bottleneck Adapter Hyperparameter Tuning",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.35)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.tight_layout()
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")
