"""
图3：诊断实验柱状图 — 同一个模型，两种下棋方式
来源：docs/training-debug-playbook.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

labels = ["纯策略\n（绕过 MCTS）", "MCTS 25 sims\n（走完整搜索）"]
values = [0.433, 0.067]
colors = ["#6aa84f", "#cc0000"]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white", linewidth=0.8)

# 柱顶数值标注
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.011,
        f"{val*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_ylim(0, 0.55)
ax.set_ylabel("vs random 胜率", fontsize=12)
ax.set_title("同一个模型，两种下棋方式", fontsize=14, fontweight="bold", pad=12)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "/Users/kyc/homework/tmp/AlphaZero/docs/articles/figures/fig3_diagnostic_bars.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved fig3_diagnostic_bars.png")
