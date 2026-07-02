"""
Generate a single combined loss curve plot from epoch_losses.csv.
Run from the Hyperparameter Tuning folder:
    python plot_combined.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

here    = os.path.dirname(os.path.abspath(__file__))
csv_in  = os.path.join(here, "lora_runs", "epoch_losses.csv")
out_png = os.path.join(here, "lora_runs", "combined_loss_curves.png")

df = pd.read_csv(csv_in)

configs = df[["r", "alpha"]].drop_duplicates().sort_values(["r", "alpha"]).values.tolist()

fig, ax = plt.subplots(figsize=(11, 6))

colors = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#F97316", "#0891B2"]

for (r, alpha), color in zip(configs, colors):
    subset = df[(df["r"] == r) & (df["alpha"] == alpha)].sort_values("epoch")
    label = f"r={int(r)}, α={int(alpha)}"

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
# Binary Cross-Entropy Loss 
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title(
    "Training & Validation Loss — LoRA Hyperparameter Tuning",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.35)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.tight_layout()
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")
