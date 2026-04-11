# 训练 Debug 实战手册

**目标读者**：将来训练 AlphaZero（或任何 RL/ML 模型）的你或继任者
**核心信条**：**ML 训练 debug 的最大陷阱是：代码没 bug，做的事却是错的**

---

## 为什么要写这份手册

这个项目的训练 debug 历史是个反面教材：

- 修了 MCTS value 计算 bug → 没用
- 修了 canonical state 不一致 → 没用
- 修了 pass action reward → 没用
- 反复调超参数 → 没用
- 最终发现真正的问题：**`num_epochs=100` 实际是 100 个梯度步**，加上 **buffer 保留太多陈旧数据**

代码 review 永远找不到这类问题，因为代码本身没 bug。这份手册总结的就是**怎么找到这类隐藏的训练方法论问题**。

---

## 12 个核心技巧

### 第一类：找到正确的"真理标准"

#### 1. 永远有一个参考实现 ⭐⭐⭐ 最重要

任何有论文的算法都有开源实现。**第一天**就拉下来对照：

| 项目 | 参考实现 |
|------|---------|
| AlphaZero | `alpha-zero-general`, OpenSpiel, leela-zero |
| Transformer | HuggingFace transformers |
| GAN | StyleGAN 官方代码 |
| RL 通用 | Stable-Baselines3, CleanRL |

**对每个超参数问**：我的值 vs 参考的值，差异在哪？为什么？
**对每个核心循环问**：我的逻辑 vs 参考的逻辑，差异在哪？为什么？

**当 ML 训练效果不好，第一个动作不是 debug 自己的代码，是和参考实现对比**。

#### 2. 用固定的外部 baseline 持续评估 ⭐⭐⭐

**Loss 下降 ≠ 模型变强**。永远不要只看 loss。

```python
# 每 N 轮训练后
win_rate = evaluate_vs_random(model, num_games=50)
win_rate_vs_iter_0 = evaluate_vs_model(model, model_iter_0)
print(f"Iter {i}: loss={loss:.3f}, vs_random={win_rate:.1%}, vs_iter0={win_rate_vs_iter_0:.1%}")
```

**Random 不会进步**，所以 vs random 的胜率是绝对参考系。如果训练 30 轮后胜率反而下降，立刻知道有问题。

**实操**：在训练循环里加一个回调，每 5 轮评估一次，画出曲线：

```
迭代:  0    5    10   15   20   25   30
胜率:  10%  25%  40%  55%  60%  62%  ?
```

下降立刻警报。

---

### 第二类：让模型"先简单再复杂"

#### 3. Overfitting 测试 ⭐⭐⭐

**最有用的 sanity check**。让模型故意过拟合 10 个样本：

```python
tiny_data = sample_dataset[:10]
for _ in range(1000):
    train_step(tiny_data)

# 现在模型应该能完美预测这 10 个样本
# 如果 loss 不能接近 0 → forward/backward/loss/数据有 bug
```

**意义**：验证 forward pass、backward pass、loss 函数、数据格式全部正确。如果模型连 10 个样本都记不住，问题在底层基础设施，更高层的训练逻辑都白搞。

#### 4. 从更小的问题开始

```
4x4 棋盘 → 训练几轮，能学到东西吗？
随机对手 → 100 局能赢 80% 吗？
单线程 → 多进程之前先单线程跑通
不用 MCTS → 直接神经网络贪心，能玩吗？
```

**你的项目历史教训**：直接上 6x6 + 多进程 + 完整 self-play + Arena，问题来了根本不知道哪一层坏了。

---

### 第三类：质疑一切假设

#### 5. 变量名会撒谎

**本项目完美中招**：`num_epochs=100` 听起来像"100 个 epoch"，实际是"100 个 batch"。

**规则**：永远读代码，不要相信变量名。特别是：

- `epochs`、`steps`、`iterations`、`batches` 这种容易混淆的术语
- 别人写的、或者你自己几个月前写的训练代码
- 通过 PR 合并的"看起来对"的代码

**实操**：每次怀疑训练有问题，先读 30 行核心训练循环。

#### 6. Loss 下降 ≠ 模型变强

| Loss 行为 | 可能含义 |
|----------|---------|
| Loss 下降，胜率上升 | ✅ 真的在学 |
| Loss 下降，胜率持平/下降 | ⚠️ 过拟合 |
| Loss 下降到 0 | ⚠️⚠️ 几乎肯定过拟合 |
| Loss 不下降 | 🔴 学不到东西（lr/数据/loss 函数问题） |
| Loss 震荡 | 🔴 lr 太大或 batch 太小 |
| Loss = NaN | 🔴 数值溢出 |

#### 7. 分别看 Policy Loss 和 Value Loss

不要只看 total loss。分开看：

| 现象 | 解读 |
|------|------|
| policy loss ↓，value loss → | 模型记住走子但预测不出胜负 |
| value loss ↓，policy loss → | 模型只会预测胜负但不会下棋 |
| 都 ↓ | 可能在学，也可能在过拟合 |
| 都 → | 学不到东西 |

---

### 第四类：让模型告诉你它在干嘛

#### 8. 看模型实际下的棋

```python
# 训练几轮后
play_game(model, opponent=random_player, render=True)
```

**直接看几局棋**比看任何指标都直观。模型在下蠢棋？开局糟糕？后期翻车？这种信息 loss 数字无法传达。

**人眼是最强的 debug 工具**。

#### 9. 看 MCTS 的访问分布

```python
# search() 返回的 visit_counts
visits = action_probs * num_simulations
print(visits)
```

| 分布 | 解读 |
|------|------|
| 集中在 1-2 个动作 | 过度利用，没探索 |
| 均匀分布 | 没学到任何偏好（神经网络在乱猜） |
| 有合理的偏好 | 健康 |
| 集中在无效动作 | 严重 bug，神经网络坏了 |

#### 10. 跟踪权重的变化幅度

```python
# 训练前后
prev_weights = {k: v.clone() for k, v in model.state_dict().items()}
train_one_iteration()
delta = sum(
    (model.state_dict()[k] - prev_weights[k]).norm().item()
    for k in prev_weights
)
print(f"Weight delta: {delta:.4f}")
```

| Delta 大小 | 解读 |
|-----------|------|
| 太大（如 > 1.0） | 训练步太激进，模型在大幅震荡（如 num_epochs=100 的情况） |
| 接近 0 | 学习率太小或梯度消失 |
| 适中且稳定下降 | 健康 |

---

### 第五类：消除变量法

#### 11. 一次只改一个东西

**常见错误**：同时改 MCTS、canonical state、value 计算、超参数。改完发现还是不行，但不知道哪个改对了哪个改错了。

**正确做法**：

```
基线：当前代码 → 胜率 5%
分支 A：只改 MCTS value → 胜率 6% (+1%)  ← 有效但很弱
回到基线，分支 B：只改 buffer 大小 → 胜率 30% (+25%)  ← 真正的关键
```

**Git 分支是你的朋友**。每个改动一个分支，分别在固定 baseline 上评估，记录数据。

#### 12. Bisect 法

如果**之前能跑通的代码现在不行了**：

```bash
git bisect start
git bisect bad   # 当前
git bisect good <某个早期 commit>
# git 自动二分查找
```

但如果**从来就没跑通过**，bisect 没用 — 你需要的是参考实现对比（技巧 1）。

---

## 决策树：训练效果不好怎么办

```
训练效果不好
    │
    ├─ 是否有外部 baseline 评估？
    │     └─ 没有 → 立刻添加（技巧 2），先确认问题真的存在
    │
    ├─ 模型能过拟合 10 个样本吗？
    │     └─ 不能 → forward/backward/loss/数据有 bug（技巧 3）
    │     └─ 能 → 训练流水线 OK，问题在更高层
    │
    ├─ 是否对照过参考实现？
    │     └─ 没有 → 立刻对照，逐项检查超参数和核心循环（技巧 1）
    │
    ├─ Loss 在下降但胜率不升？
    │     └─ 过拟合或数据/标签问题（技巧 6）
    │
    ├─ 看模型实际下的棋
    │     └─ 下蠢棋？记录具体错误模式（技巧 8）
    │
    └─ 还是找不到？
          └─ 一次禁用一个组件（augmentation/MCTS/temperature/...），
             看哪个组件被禁用后训练突然变好/变坏（技巧 11）
```

---

## 本项目的具体教训

| 我们之前做错的 | 应该做的 |
|--------------|---------|
| 反复修 MCTS value bug | ✅ 是对的，但只是表面 |
| 看 loss 下降以为在学 | ❌ 应该看 vs random 胜率 |
| 反复改超参数没记录 | ❌ 应该每次只改一个 + 记录数据 |
| 没做 overfit test | ❌ 应该做 |
| 信任 `num_epochs` 这个名字 | ❌ 应该读 train_network 代码 |
| 一次改多个东西 | ❌ 应该每次只改一个 |
| 没对照 alpha-zero-general | ❌ 应该第一天就对照 |

如果早做了这些，10 分钟就能发现 `num_epochs=100` 和 buffer 太大的问题，不用花几周改 MCTS。

---

## 最有效的 3 件事

如果时间紧只能做 3 件：

1. **每 5 轮训练评估一次 vs random 胜率**，画出曲线
2. **从第一天就和参考实现 side-by-side 对照超参数**
3. **每次只改一个变量，记录改动前后的胜率**

---

## 检查清单：开始训练前

复制这个清单，每次开始新训练前过一遍：

- [ ] 我有参考实现了吗？所有超参数都对照过吗？
- [ ] 我能在 < 5 分钟内评估当前模型 vs random 胜率吗？
- [ ] 我做过 overfit test 吗？模型能记住 10 个样本吗？
- [ ] 我的训练循环代码我亲手读过吗（不是看变量名）？
- [ ] 我的训练数据格式正确吗？（state shape、policy 归一化、value 范围）
- [ ] 我会记录每次改动和对应的胜率吗？
- [ ] 我会画 loss 曲线 和 胜率曲线 吗？
- [ ] 我有 git 分支策略，每次改动一个分支吗？

---

## 如何诊断"网络容量不够" (Capacity Ceiling)

> **背景**：本项目曾经修了**所有**能想到的训练方法论问题（MCTS 视角、真 epochs、滑动窗口、Dropout），胜率仍然卡在 20% vs random。最后发现真正的瓶颈是**网络太小**（128 通道）。这一节总结如何**快速判断**是否撞上了容量天花板。

### 症状识别

当你看到这些现象同时出现时，**极可能是网络容量不够**：

| 症状 | 说明 |
|------|------|
| 🔴 **多次修改无效** | 改 lr、batch_size、num_epochs、正则化，vs random 胜率**不动** |
| 🔴 **天花板数字稳定** | 不管怎么改，vs random 胜率最高都是那个数字 (例如 20%) |
| 🔴 **第 0 轮就达到天花板** | 自动接受的 iter 0（仅少量训练）就已经达到最终的胜率峰值 |
| 🔴 **后续迭代变弱或持平** | 从第 1 轮开始，候选模型都无法超越 iter 0 |
| 🔴 **Arena 高拒绝率** | 拒绝率 > 80%，候选模型持续 0% 或 50% 胜率 |
| 🟡 **Loss 持续下降** | 但 vs random 不变（模型在拟合噪声，不是学真知识）|
| 🟡 **训练多样化但无效** | 所有超参数搜索的结果都差不多 |

**单独看**任何一个症状都不够。但**同时出现 3 个以上**，基本可以确定。

### 诊断流程

#### Step 1: 画胜率曲线

```bash
grep "vs random:" training*.log | \
    awk -F'vs random: ' '{print $2}' | awk -F'%' '{print $1}'
```

画出所有训练轮次的 vs random 胜率。**看曲线形状**：

```
容量不够:       训练正常:
                                          ↑
 ██                                       ██
 ██  ██  ██                               ████
 ██  ██  ██  ██  ██                       ██████
 ──────────────                           ────────
 0  5  10 15 20 轮                         0  5  10 15 20 轮
```

**容量不够**的曲线：横着走，没有趋势，起点几乎就是终点。
**训练正常**的曲线：稳步上升。

#### Step 2: 对比 iter 0 和 global_best

```python
global_best.pt 的 vs random 胜率 ≈ checkpoint_0.pt 的胜率
```

如果是的，说明**所有后续训练都白费**，问题不在训练方法，而在更基础的地方。

#### Step 3: 对比参考实现

查看 alpha-zero-general 或同类项目的网络配置：

| 项目 | 通道 | 参数量 |
|------|------|--------|
| 本项目 (旧) | 128 | ~1.2M |
| alpha-zero-general | 512 | ~5-8M |
| AlphaGo Zero (Go 9x9) | 256 | ~10M+ |
| AlphaGo Zero (Go 19x19) | 256 | ~40M+ |

**简单算术**：你的网络参数量 vs 参考的 4-10 倍差距，就是容量嫌疑。

#### Step 4: 排除训练方法论（这一步最耗时间）

按以下顺序尝试，每次只改一项：

1. **修复 MCTS bug**（如果有）
2. **真正的 epochs**（num_epochs 不是梯度步数）
3. **滑动窗口**（20 iterations）
4. **分段温度**（前 N 步探索，后贪心）
5. **Dropout**（正则化）

如果**这 5 项都做了**胜率还是不动，**容量问题几乎确定**。

### 确认方法：直接加大网络

这是**唯一确定的判定方法**：

```python
# 旧
num_channels = 128  # 1.2M 参数

# 新
num_channels = 256  # 5.1M 参数 (4x)
```

**只改这一个参数**，其他全部保持。训练新网络，看胜率天花板是否上升。

- 如果**天花板上升**（例如 20% → 40%）→ 确认是容量问题 ✓
- 如果**天花板不变**（还是 20%）→ 容量不是唯一问题，更深的 bug 或数据质量问题

### 为什么容量不够难以察觉

**与训练方法论问题不同**，容量不够有几个隐蔽之处：

1. **Loss 依然下降**：一个小网络在其容量限内 loss 可以降得很漂亮
2. **没有报错**：一切正常，没有 assertion 失败
3. **Arena 工作正常**：候选模型确实输了，不是 bug
4. **相似问题在文献里少有记录**：大多数论文假定你已经选对了网络规模

**唯一的信号是 vs random 胜率不动**。所以**外部 baseline 评估是救命稻草**。

### 选择合适的网络规模

经验值（基于状态空间大小）：

| 游戏状态空间 | 推荐通道数 | 参考 |
|-------------|----------|------|
| ~10^4 (4x4 棋盘) | 64 | 玩具级 |
| ~10^8 (5x5 棋盘) | 128 | 入门级 |
| **~10^12 (6x6 Othello)** | **256-384** | 本项目 |
| ~10^20 (Connect 4, 8x8 Othello) | 512 | alpha-zero-general 水准 |
| ~10^80 (19x19 Go) | 256 with 40+ ResBlocks | AlphaGo Zero |

**不要凭感觉小**：当你不确定时，选**略大**的。训练时间能用算力堆，但容量不够是硬天花板。

### 本项目踩坑记录（作为教训）

- v1-v3 (bug+128ch): 2-8% vs random
- v4 (滑动窗口+epochs=10 hack): 22%
- v5 (分段温度): 8% (退步)
- v6 (MCTS 视角修复): 23.3%
- v7 (+real epochs 2): 20%
- v8 (+real epochs 1): 20%
- v9 (+dropout 0.3): 20%
- **v10 (+256 channels)**: **待验证**

看到 v6-v9 几乎平行吗？**就是容量天花板**。早该换网络了，但我们花了好几轮修训练方法论才意识到。

### 检查清单（新增）

除了主清单，遇到多次调整无效时加问这些：

- [ ] 我对比过参考实现的网络规模了吗？
- [ ] 我的网络参数量在合理范围吗？
- [ ] 我画过 vs random 胜率 vs 训练轮次的曲线吗？
- [ ] 我尝试过**只**加大网络（不改其他）看是否有效吗？
- [ ] 我有"至少试一次大网络"的底线吗？

---

## MCTS 均匀陷阱 (Uniform Trap)

> **背景**：本项目出现过一个诡异现象：修了所有能想到的 bug，vs random 胜率仍然卡在 20%（甚至有时候更低）。最后发现是 **MCTS 无法产生有意义的训练信号**，陷入"均匀 → 均匀"的死循环。这是 AlphaZero 的**经典 bootstrap 失败**。

### 现象描述

```
未训练 NN (权重随机) → policy 输出 ≈ 均匀分布
    ↓
MCTS (50 模拟) 基于 ≈ 均匀的先验探索
    ↓
MCTS 访问次数 ≈ 均匀 (例如 13/13/12/12 visits 给 4 个合法动作)
    ↓
训练 NN 去匹配 ≈ 均匀的目标
    ↓ (1 个 epoch 不够学到那 2-4% 的微小差异)
NN 继续输出 ≈ 均匀
    ↓
死循环，训练永远不启动
```

### 如何检测

**关键诊断**：直接打印 MCTS 对初始局面的输出。

```python
import numpy as np
import torch
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from mcts.mcts import MCTS

torch.manual_seed(42)
model = AlphaZeroNetwork(6, device='cpu')
model.eval()
mcts = MCTS(model, c_puct=2.0, num_simulations=50)

env = OthelloEnv(size=6)
env.reset()
canonical = env.board.get_canonical_state()
state = mcts.canonical_to_observation(canonical, env)
action_probs = mcts.search(state, env, temperature=1.0)

valid = np.where(env.get_valid_moves_mask() == 1)[0]
print(f'MCTS probs at valid moves: {[f"{action_probs[i]:.3f}" for i in valid]}')
```

**症状**：

```
MCTS probs at valid moves: ['0.260', '0.260', '0.240', '0.240']
```

所有动作概率在 **24%-26% 之间**（1/N ± 一点点扰动）。这就是均匀陷阱的指纹。

**正常情况**下，即使未训练的网络 + 足够的 MCTS 模拟，访问次数应该有**明显偏差**（例如 60%, 20%, 15%, 5%）。

### 为什么会发生

1. **先验几乎均匀**：未训练的 NN 输出接近 uniform
2. **50 次模拟不够探索**：PUCT 公式 `u = c * prior * sqrt(N) / (1+n)` 在 prior 均匀时会尽可能平均分配访问次数
3. **访问次数 ≈ 均匀**：50 visits / 4 actions ≈ 12-13 each
4. **训练目标 ≈ 均匀**：NN 被训练去预测这种 "1/N + 小噪声" 的分布
5. **1 个 epoch 不够学到那 2-4% 的差异**：模型收敛到 uniform，不是偏差
6. **下一轮 MCTS 还是均匀**：因为 NN 仍然输出 uniform

整个系统卡在一个**均匀不动点**里。

### 为什么训练强度可以打破陷阱

alpha-zero-general 用 **10 个 epoch**（每次完整跑 446K 样本 10 遍）。在 10 轮完整训练下：

- NN 不仅匹配均值，还能**学到 2-4% 的微小偏差**
- 下一轮 MCTS 用这个略有偏差的 NN 做先验
- PUCT 公式放大先验差异，访问次数分布变得**不均匀**
- MCTS 输出有真正的偏好
- NN 学到更强的偏好
- **bootstrap 循环启动** 🎉

而 1 个 epoch 在数学上就不够：

```
每轮迭代:
  - 数据池 ~446K samples
  - 1 epoch = ~3500 batches
  - 学习率 0.001 × 每个 batch 的梯度
  - 权重更新总量 ≈ 3500 * 0.001 * avg_gradient
  - 不足以改变 NN 的 softmax 输出从 "完全均匀" 变成 "略有偏差"
```

10 个 epoch 就是 **10x 的权重更新量**，足以让 NN 产生明显偏差。

### 解决方案

**按优先级排序**：

#### ⭐ 最有效：增加训练强度

```python
num_epochs = 10   # 不是 1！
```

**效果**：彻底打破均匀陷阱。代价：训练时间 × 10。

#### 🟢 辅助：先验温度 < 1

在 MCTS 使用 NN 策略之前，稍微放大差异：

```python
policy = torch.softmax(policy_logits / 0.5, dim=1)  # temperature < 1 放大
```

这让 NN 输出的微小差异变得更明显，加速 bootstrap。但过度放大会让模型过于自信。

#### 🟢 辅助：增加 MCTS 模拟数

更多模拟（例如 200 而不是 50）能让 PUCT 公式的探索项更早衰减，让访问次数反映更多的 value 信号而不是 prior 信号。

#### 🟡 Dirichlet 噪声帮助有限

Dirichlet 噪声只在**根节点**加一次，增加探索的多样性。但它不能从根本上解决均匀陷阱 — 因为它只影响根节点的先验，不影响子节点的 bootstrap。

### 验证是否突破了均匀陷阱

训练几轮后，重新运行上面的诊断脚本。如果看到：

```
MCTS probs at valid moves: ['0.450', '0.300', '0.150', '0.100']
```

— 非均匀！不同动作有**明显不同**的概率，说明 NN 已经学到了实际策略，bootstrap 循环启动了。

如果还是：

```
MCTS probs at valid moves: ['0.260', '0.260', '0.240', '0.240']
```

— 还在陷阱里。需要更多 epochs、更大网络、或更多 MCTS 模拟。

### 为什么代码 review 找不到这个 bug

**因为它不是代码 bug**：

- MCTS 代码正确
- NN 代码正确
- 训练代码正确
- 每一块单独测试都 pass
- 但三者组合在一起有一个**动力学失败模式**

这种"系统性"问题只有通过**可视化 MCTS 输出分布**才能看到。

### 与 "Capacity Ceiling" 的区别

两个都表现为"训练后胜率不动"，但原因不同：

| | 均匀陷阱 | Capacity Ceiling |
|---|---|---|
| **NN 能否学合成数据？** | ✅ 能 | ✅ 能 |
| **Loss 是否下降？** | ✅ 略下降 | ✅ 正常下降 |
| **MCTS 输出分布** | ❌ 几乎均匀 | ✅ 非均匀，但学不到更多 |
| **解决方式** | 增加训练强度/epochs | 加大网络 |

**诊断顺序**：先检查 MCTS 输出分布。如果均匀 → 均匀陷阱。如果非均匀但学不到更多 → 容量问题。

### 本项目的完整时间线（作为教训）

- v1-v3: MCTS bug 时期，胜率 2-8%
- v4: epochs=100 (假 10 epochs) → 22%
- v5-v6: 修各种 MCTS bug → 23%
- v7-v8: 真 epochs=2 → 20% (轻微退步)
- v9: +dropout → 20%
- v10: +256 channels → 10% (更差)
- **v11: 真 epochs=10** → 待验证 (预期突破 40%+)

**关键教训**：我们花了 v4-v10 共 **7 次实验**，全都在改训练方法论的不同方面，但**没有检查过 MCTS 的输出分布**。如果第一天就运行那 10 行诊断代码，立刻就能看到均匀陷阱。

**每次新项目开始训练前**，先运行这个诊断。它只花几秒钟。

---

## 一句话总结

> **代码 review 找不到训练方法论问题。只有外部 baseline + 参考实现对比才能发现。**

这就是为什么 ML 工程师要花一半时间做"写代码"之外的事 — 评估、对比、记录、可视化。

---

## 参考

- [Andrej Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — 神经网络训练 debug 的经典文章
- alpha-zero-general: 本项目主要的对照参考
- AlphaGo Zero (Nature 2017) Methods 章节
