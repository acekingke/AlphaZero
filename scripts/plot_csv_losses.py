import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/kyc/homework/tmp/AlphaZero/logs/alphazero_20260414_083446.csv"
)
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/Users/kyc/homework/tmp/AlphaZero/training_metrics_all30.png"
)

iters, pol, val, tot = [], [], [], []
with csv_path.open() as f:
    for row in csv.DictReader(f):
        iters.append(int(row["iteration"]))
        pol.append(float(row["policy_loss"]))
        val.append(float(row["value_loss"]))
        tot.append(float(row["total_loss"]))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, y, title in zip(
    axes, [pol, val, tot], ["Policy Loss", "Value Loss", "Total Loss"]
):
    ax.plot(iters, y, marker="o", markersize=4, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Training Iterations")
    ax.set_xticks(iters)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.grid(True, alpha=0.4)
    for x, v in zip(iters, y):
        ax.annotate(f"{v:.3f}", (x, v), fontsize=6,
                    xytext=(0, 4), textcoords="offset points", ha="center")

fig.tight_layout()
fig.savefig(out_path, dpi=130)
print(f"saved -> {out_path}")
