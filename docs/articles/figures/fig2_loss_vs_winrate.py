"""
Figure 2: Loss perfect vs winrate floored
同一次训练的两面 — loss 完美下降，胜率贴地
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.RandomState(42)

iters = np.arange(1, 31)  # 30 次迭代

# 左图：loss 光滑衰减
loss = 2.5 * np.exp(-iters / 5) + 0.3 + rng.randn(len(iters)) * 0.03

# 右图：胜率贴在 0.05 附近，小幅正弦波 + 少量噪声
winrate = 0.05 + 0.02 * np.sin(iters * 0.6) + rng.randn(len(iters)) * 0.015
winrate = np.clip(winrate, 0.0, 1.0)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('同一次训练的两面', fontsize=15, fontweight='bold', y=1.01)

# ── 左图 ──────────────────────────────────────────────
ax_left.plot(iters, loss, color='#2166ac', linewidth=2.2, marker='o',
             markersize=4, label='训练 loss')
ax_left.set_title('训练 loss（看起来很完美）', fontsize=13, pad=8)
ax_left.set_xlabel('训练迭代轮次', fontsize=11)
ax_left.set_ylabel('Loss', fontsize=11)
ax_left.set_ylim(bottom=0)
ax_left.grid(True, linestyle='--', alpha=0.4)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# 添加"完美下降"标注箭头
ax_left.annotate('平滑下降', xy=(15, loss[14]), xytext=(20, 1.2),
                 fontsize=10, color='#2166ac',
                 arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.5),
                 ha='center')

# ── 右图 ──────────────────────────────────────────────
ax_right.plot(iters, winrate, color='#d6604d', linewidth=2.2, marker='o',
              markersize=4, label='vs random 胜率')

# y=0.5 参考线
ax_right.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                 label='随机对手的期望胜率 (50%)')
ax_right.text(30.5, 0.5, '随机对手\n期望胜率', va='center', ha='left',
              fontsize=8.5, color='gray')

ax_right.set_title('vs random 胜率（始终贴地）', fontsize=13, pad=8)
ax_right.set_xlabel('训练迭代轮次', fontsize=11)
ax_right.set_ylabel('胜率', fontsize=11)
ax_right.set_ylim(-0.02, 0.75)
ax_right.set_xlim(1, 34)  # 留出右侧标注空间
ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax_right.grid(True, linestyle='--', alpha=0.4)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

# 双向箭头标注距离
ax_right.annotate('', xy=(3, 0.5), xytext=(3, 0.05),
                  arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5))
ax_right.text(4.2, 0.275, '差距\n~45%', fontsize=9, color='#555555', va='center')

# 图例
ax_right.legend(loc='upper left', fontsize=9, framealpha=0.6)

plt.tight_layout()

out_path = '/Users/kyc/homework/tmp/AlphaZero/docs/articles/figures/fig2_loss_vs_winrate.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
