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

## 一句话总结

> **代码 review 找不到训练方法论问题。只有外部 baseline + 参考实现对比才能发现。**

这就是为什么 ML 工程师要花一半时间做"写代码"之外的事 — 评估、对比、记录、可视化。

---

## 参考

- [Andrej Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — 神经网络训练 debug 的经典文章
- alpha-zero-general: 本项目主要的对照参考
- AlphaGo Zero (Nature 2017) Methods 章节
