# 对齐 alpha-zero-general：关键差异与修复

> **背景**：v13/v14 训练 policy loss 卡在 1.48（uniform entropy），Arena 通过率极低。
> 逐一对照 alpha-zero-general 的 Coach.py / NNet.py / OthelloNNet.py / MCTS.py，
> 发现 5 个行为差异，修复后（v16）loss 立即下降，Arena 开始接受新模型。

---

## 差异一览

| # | 差异 | alpha-zero-general | 我们（修复前） | 影响 |
|---|------|-------------------|---------------|------|
| 1 | Optimizer 生命周期 | **每轮新建** `optim.Adam()` | 单个 Adam 跑全程 + LR 衰减 + weight_decay | ⭐⭐⭐ 致命 |
| 2 | 输入通道数 | **1 通道** `{-1, 0, +1}` 棋盘 | 3 通道 (己方/对方/玩家指示) | ⭐⭐ 高 |
| 3 | 批采样方式 | **有放回随机** `np.random.randint` | 无放回 shuffle + 顺序遍历 | ⭐ 低 |
| 4 | 网络输出 | **log_softmax** + `exp()` 解码 | raw logits + `softmax()` | ⭐ 无（数学等价） |
| 5 | MCTS 树生命周期 | 同局内**跨步缓存** Qsa/Nsa | 每步新建树 | ⭐ 低 |

---

## 差异 1（致命）：Optimizer 每轮重建

### alpha-zero-general 的做法

```python
# NNet.py:40 — 每次 train() 调用都新建
def train(self, examples):
    optimizer = optim.Adam(self.nnet.parameters())  # 无 weight_decay, 无 LR 衰减
    for epoch in range(args.epochs):
        ...
```

### 我们之前的做法

```python
# __init__ 里创建一次，100 轮共用
self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

# 每轮还做 LR 衰减
current_lr = self.lr * (0.98 ** iteration_count)
```

### 为什么这是致命差异

Adam 维护两个状态：一阶动量 m 和二阶动量 v。

- **Fresh Adam**：每轮 m=0, v=0 → 前几步的梯度直接生效，不受历史影响
- **持续 Adam**：动量从上一轮继承 → 如果上一轮的梯度方向是错的（早期很常见），
  动量会把模型继续推向错误方向，新的梯度信号被"淹没"

对 AlphaZero 来说，每轮的训练数据来自不同 iteration 的自对弈，
数据分布本身在变化。Fresh optimizer 相当于**每轮重新适应当前数据分布**，
而持续 optimizer 会保留对旧数据分布的记忆。

加上 `weight_decay=1e-4`（alpha-zero-general 没有），额外抑制了参数更新幅度。

### 症状

v13/v14 的 policy loss 卡在 **1.48**（≈ ln(5) ≈ 开局 4 个合法动作的 uniform entropy），
持续 100 轮不动。这说明 Adam 的动量锁住了模型，梯度更新被自己的历史抵消。

### 修复后

v16 的 policy loss：**1.24 → 0.98 → 0.85**，持续下降。

---

## 差异 2（高影响）：1 通道 vs 3 通道输入

### alpha-zero-general

```python
# OthelloNNet.py:19
self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)

# forward():
s = s.view(-1, 1, self.board_x, self.board_y)  # 直接喂 canonical board
```

输入是 canonical board，一个 6×6 矩阵，值为 `{-1, 0, +1}`。

### 我们之前

```python
self.conv1 = nn.Conv2d(3, num_channels, ...)

# 输入 3 通道:
# ch0 = (board == 1)    → 当前玩家棋子 (binary)
# ch1 = (board == -1)   → 对手棋子 (binary)
# ch2 = ones(6, 6)      → 玩家指示器（canonical 下恒为 1）
```

### 为什么 3 通道有问题

1. **ch2 恒为全 1**：在 canonical form 下，当前玩家永远是 +1，所以 ch2 是常数。
   网络必须学会忽略它 → 浪费容量
2. **ch0 + ch1 是 canonical board 的线性变换**：`ch0 = (board+1)/2` 当 board∈{0,1}。
   没有增加任何信息，只是换了一种编码
3. **conv1 参数多 3x**：`(3×512×3×3 = 13.8K)` vs `(1×512×3×3 = 4.6K)`，
   更多参数需要更多数据才能训练好

### 修复

```python
# 现在
self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
# forward: x.view(-1, 1, game_size, game_size)
```

MCTS 直接喂 `canonical_state`（6×6 float32），不再做 3 通道转换。

---

## 差异 3（低影响）：有放回 vs 无放回采样

### alpha-zero-general

```python
# NNet.py:52
for _ in range(batch_count):
    sample_ids = np.random.randint(len(examples), size=args.batch_size)
```

每个 batch 从全量数据中**独立随机抽取**（有放回）。
某些样本可能被多次采到，某些可能被漏掉。

### 我们之前

```python
perm = torch.randperm(n)
for batch_start in range(0, n, self.batch_size):
    batch_idx = perm[batch_start : batch_start + self.batch_size]
```

Shuffle 后顺序遍历，每个样本恰好被看到一次。

### 差异

有放回采样更随机（隐式正则化），但对训练质量影响很小。
主要是为了和 alpha-zero-general 保持一致。

---

## 差异 4（数学等价）：log_softmax 输出

### alpha-zero-general

```python
# OthelloNNet.py:54
return F.log_softmax(pi, dim=1), torch.tanh(v)

# NNet.py:94 — predict 时转回概率
return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

# NNet.py:97 — loss 直接用 log_prob
def loss_pi(self, targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]
```

### 我们之前

```python
return policy_logits, torch.tanh(value)  # 输出 raw logits
# MCTS: torch.softmax(logits)
# loss: -sum(targets * log_softmax(logits))
```

### 差异

`-sum(targets * log_softmax(logits))` 和 `-sum(targets * log_probs)` 数学等价。
修复只是为了代码一致性，不影响训练行为。

---

## 差异 5（未修复）：MCTS 树跨步缓存

### alpha-zero-general

```python
# MCTS.__init__: 缓存字典在整局游戏中持续存在
self.Qsa = {}  # Q values
self.Nsa = {}  # visit counts
self.Ps = {}   # policies

# Coach.py:88 — 每局新建 MCTS（跨局不缓存）
self.mcts = MCTS(self.game, self.nnet, self.args)
```

同一局游戏内，前面步骤的 MCTS 搜索结果**被后续步骤复用**。

### 我们

```python
def search(self, state, env, ...):
    root = Node(0)  # 每步新建树，无缓存
```

### 为什么没修复

需要重写整个 MCTS 从 Node-based 到 dict-based 架构。影响低（50 sims 足够
重新探索），暂不修复。

---

## 额外修复：每轮保存 checkpoint

### alpha-zero-general

```python
# Coach.py:100 — 每轮都保存 training examples
self.saveTrainExamples(i - 1)  # 无论 accept 还是 reject
```

### 我们之前

`save_checkpoint()` 只在 Arena **接受**时调用 → 连续 reject 时 training examples 全丢。

### 修复

新增 `save_latest_checkpoint()`，每轮调用，保存到 `checkpoint_latest.pt` + `.examples`。

---

## 效果对比

| 指标 | v13 (修复前) | v14 (+Dirichlet) | **v16 (全面对齐)** |
|------|-------------|------------------|-------------------|
| Policy loss | 1.48 卡住 | 1.55 卡住 | **1.24 → 0.85 下降** |
| Arena 接受 (前 6 轮) | 1/6 | 0/6 | **3/6** |
| vs random 最高 | 3.3% | 3.3% | **10.0%** |
| Loss 趋势 | 平坦 | 上升 | **持续下降** |

### 结论

> **最关键的修复是 fresh optimizer**。持续 Adam 的动量累积 + weight_decay
> 把模型锁死在随机初始化附近。每轮重建 optimizer 让梯度信号能真正生效，
> 这是 alpha-zero-general 能收敛而我们不能的根本原因。
