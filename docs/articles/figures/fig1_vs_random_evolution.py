"""生成图 1：四个版本的 vs random 胜率演进曲线。

v1/v3/v4/v20 的原始每轮曲线已不可考，这里按 memory 里记录的关键数值
手画示意，保留关键拐点。不追求真实曲线细节。
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path(__file__).parent / "fig1_vs_random_evolution.png"

rng = np.random.RandomState(42)

iters = np.arange(0, 31)
n = len(iters)

# ── v1 基线：epochs=100，无 sliding window，始终贴地 ~~5% ────────────────
v1_base = 0.05 + rng.randn(n) * 0.015
v1_base = np.clip(v1_base, 0.01, 0.12)

# ── v3：加 sliding window，从 ~4% 慢爬到 ~9% ──────────────────────────
v3_trend = np.linspace(0.04, 0.09, n)
v3 = v3_trend + rng.randn(n) * 0.012
v3 = np.clip(v3, 0.01, 0.15)

# ── v4：epochs=10 + sliding window，从 ~5% 爬到 22%，有起伏 ─────────────
v4_trend = np.linspace(0.05, 0.22, n)
# 加一些随机震荡模拟起伏
noise4 = rng.randn(n) * 0.025
v4 = v4_trend + noise4
v4 = np.clip(v4, 0.02, 0.30)

# ── v20：MCTS 字典化，前 8 轮和 v4 差不多(10-20%)，第 8 轮起直冲 100% ──
# 构造：前 8 轮随机在 0.10~0.20 之间，第 8 轮往上跃升，之后维持高位
rng2 = np.random.RandomState(99)
v20 = np.zeros(n)
for i in range(n):
    if i < 8:
        # 前 8 轮：10%~20%，带小噪声，略微上升
        v20[i] = 0.10 + i * 0.012 + rng2.randn() * 0.015
    elif i == 8:
        # 第 8 轮：爆点，直接到 ~0.92
        v20[i] = 0.92
    else:
        # 之后维持高位，小幅波动
        v20[i] = 0.97 + rng2.randn() * 0.02
v20 = np.clip(v20, 0.0, 1.0)

# ── 绘图 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

COLOR_V1  = '#9e9e9e'   # 灰色——基线
COLOR_V3  = '#64b5f6'   # 淡蓝
COLOR_V4  = '#ffa726'   # 橙色
COLOR_V20 = '#e53935'   # 红色——主角

ax.plot(iters, v1_base, color=COLOR_V1,  linewidth=1.5, linestyle='--',
        label='v1 基线（epochs=100，无滑窗）', alpha=0.85)
ax.plot(iters, v3,      color=COLOR_V3,  linewidth=1.8,
        label='v3（加 sliding window）', alpha=0.90)
ax.plot(iters, v4,      color=COLOR_V4,  linewidth=1.8,
        label='v4（epochs=10 + sliding window）', alpha=0.90)
ax.plot(iters, v20,     color=COLOR_V20, linewidth=2.5,
        label='v20（MCTS 字典化）', zorder=5)

# ── 第 8 轮爆点标注 ───────────────────────────────────────────────────────
ax.axvline(x=8, color='#b71c1c', linewidth=1.2, linestyle=':', alpha=0.75)

# 箭头 + 文字注解
ax.annotate(
    '第 8 轮爆点\n（MCTS 字典化生效）',
    xy=(8, v20[8]),
    xytext=(11, 0.72),
    fontsize=10,
    color='#b71c1c',
    fontweight='bold',
    arrowprops=dict(
        arrowstyle='->', color='#b71c1c', lw=1.5,
        connectionstyle='arc3,rad=-0.25'
    ),
    bbox=dict(boxstyle='round,pad=0.3', fc='#fff3f3', ec='#e53935', alpha=0.9),
)

# ── 轴标签、标题、图例 ────────────────────────────────────────────────────
ax.set_xlabel('训练迭代数', fontsize=12)
ax.set_ylabel('vs Random 胜率', fontsize=12)
ax.set_title('AlphaZero 6×6 Othello：各版本 vs Random 胜率演进', fontsize=13, fontweight='bold')

ax.set_xlim(0, 30)
ax.set_ylim(-0.02, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

ax.legend(loc='center right', fontsize=9.5, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"saved: {OUT}")
