# 未来优化计划

**Created**: 2026-04-10
**Context**: 对比 `alpha-zero-general` 实现后，识别出本项目可优化的训练策略
**Status**: 待 `001-arena-model-selection` 训练完成后验证

## 背景

当前 Arena 机制已实现，但训练初期出现高拒绝率（76%）的现象。通过对比 `alpha-zero-general` 的成熟实现，识别出以下需要优化的差异点。这些优化按优先级排序，应在当前训练完成后，根据 `best.pt` 的实际战力评估再决定实施。

## 关键差异对比表

| 对比项 | alpha-zero-general | 本项目当前 | 影响 |
|--------|-------------------|------------|------|
| 温度策略 | 前 15 步 temp=1，之后 temp=0 | 全程 temperature=1.0 | 🔴 高 |
| 训练 Epochs | 10 | 100 | 🔴 高 |
| 数据 Buffer | 滑动窗口 20 轮 | 10M 样本 deque (~50 轮) | 🟡 中 |
| Dirichlet 噪声 | 不使用 | 在使用 | 🟢 低 |
| MCTS 模拟数 | 25 | 50 | 🟢 低 |
| 学习率 | 0.001 | 0.001 | ✅ 一致 |
| Arena 阈值 | 60% | 60% | ✅ 一致 |

---

## 优化项 1: 分段温度策略 🔴 高优先级

### 问题

当前自我对弈全程使用 `temperature=1.0`，导致即使到棋局后期，模型仍按概率分布随机选动作。这造成：

- **后期决策质量差**：模型本来知道哪步最好，但还是有概率走次优棋
- **价值标签噪声大**：胜负结果可能因为后期"随机失误"而失真，让神经网络学到错误的 value 信号
- **策略学习被污染**：MCTS 访问次数分布反映的是探索而非真正的最佳策略

### 解决方案

模仿 alpha-zero-general 的 `tempThreshold` 机制：

```python
# 在 self_play() 和 _worker_play() 中
TEMP_THRESHOLD = 15  # 前 15 步用于探索

while not env.board.is_done():
    canonical_state = env.board.get_canonical_state()
    state = mcts.canonical_to_observation(canonical_state, env)
    
    # 分段温度
    current_temp = self.temperature if step < TEMP_THRESHOLD else 0.0
    action_probs = mcts.search(state, env, current_temp, add_noise=True)
    ...
```

### 预期效果

- 前 15 步保持探索，学习开局多样性
- 第 16 步开始贪心，确保 value 标签反映真实最优棋力
- 减少自我对弈数据中的"自残"游戏（明明赢面大却走输）

### 验证方法

实施后跑 5 轮训练，对比 Arena 拒绝率是否下降。

---

## 优化项 2: 降低训练 Epochs 🔴 高优先级

### 问题

当前每轮训练 **100 epochs**，alpha-zero-general 只用 **10 epochs**。在小数据集上跑 100 epochs 会导致：

- **严重过拟合**：模型记住了自我对弈的具体局面，但泛化能力下降
- **战力倒退**：训练 loss 在降，但实际对局表现变差
- **算力浪费**：90% 的训练时间在做无用功

这很可能就是项目历史上 **胜率 45% → 5% 下降的根本原因**。

### 解决方案

```python
# main.py
train_parser.add_argument('--num_epochs', type=int, default=10, 
                          help='Training epochs per iteration (default: 10)')
```

并在 `AlphaZeroTrainer.__init__()` 把默认值从 100 改为 10。

### 预期效果

- 单轮训练时间减少 90%
- 模型不会过度拟合当轮数据
- 整体训练效率提升

### 验证方法

对比 epochs=10 和 epochs=100 训练 5 轮后的 best.pt vs random 胜率。

---

## 优化项 3: 缩小数据 Buffer 🟡 中优先级

### 概念：什么是"滑动窗口 20 轮"

"滑动窗口 20 轮"是 alpha-zero-general 管理训练数据的方式 — **训练时只用最近 20 轮自我对弈生成的数据，更早的数据会被丢弃**。

#### 形象示意

假设当前在第 25 轮训练：

```
迭代:    1   2   3   4   5   6  ...   7   8  ...  23  24  25
数据:    ❌  ❌  ❌  ❌  ❌  ✓         ✓   ✓        ✓   ✓   ✓
         ↑___________↑   ↑_______________________________________↑
         已被丢弃         保留的窗口（最近 20 轮）
```

第 1-5 轮的数据被"挤出"窗口，不再参与训练。

#### 用代码表达

```python
self.train_examples_history = []  # 列表的列表

# 每轮训练后
self.train_examples_history.append(本轮新生成的数据)

if len(self.train_examples_history) > 20:
    self.train_examples_history.pop(0)  # 丢掉最老的一轮

# 训练时把保留的所有轮数据合并
all_data = []
for iter_data in self.train_examples_history:
    all_data.extend(iter_data)
random.shuffle(all_data)
# 用 all_data 训练神经网络
```

#### 为什么这能解决问题

举例说明：
- **第 1 轮**：模型还很弱，自我对弈生成的数据里"赢家"其实是个弱菜，value 标签 = +1 但其实这步走得很烂
- **第 25 轮**：模型已经很强了，知道这步是烂招，但 buffer 里第 1 轮的数据还在告诉它"这步 = 赢"
- **冲突**：模型在矛盾的标签上反复横跳，无法稳定收敛

滑动窗口让早期弱模型的错误判断自然过期，避免污染后期训练。

#### 为什么是 20 而不是 5 或 50

- **太小（如 5）**：数据量不够，神经网络容易过拟合当前几轮
- **太大（如 50）**：又开始包含陈旧数据
- **20 是经验值**：足够多样性 + 仍然新鲜

### 问题

当前 buffer 是 `deque(maxlen=10_000_000)`，约 50+ 轮训练数据。这导致：

- **训练在陈旧数据上**：早期弱模型生成的数据仍在 buffer 中，拖累后期训练
- **价值标签漂移**：早期数据的 value 标签来自弱策略，与当前模型不一致
- **内存浪费**

### 解决方案

模仿 alpha-zero-general 的滑动窗口：

```python
# AlphaZeroTrainer.__init__()
self.train_examples_history = []  # list of lists, one per iteration
self.num_iters_for_history = 20

# train() 中
def add_iteration_examples(self, examples):
    self.train_examples_history.append(examples)
    if len(self.train_examples_history) > self.num_iters_for_history:
        self.train_examples_history.pop(0)
    
    # 训练时合并所有保留的迭代
    all_examples = [e for iter_examples in self.train_examples_history for e in iter_examples]
    random.shuffle(all_examples)
```

### 预期效果

- 训练数据始终来自最近 20 轮（更接近当前模型的能力）
- 避免被早期弱模型的数据污染
- 内存占用大幅下降

### 验证方法

对比 buffer 大小不同时的 Arena 拒绝率。

---

## 优化项 3.5: 训练历史持久化（Resume Bug 修复）🔴 高优先级

### 问题

当前的滑动窗口实现存在一个 bug：`train_examples_history` **只存在内存中**，不会被保存到 checkpoint。这导致：

- 用 `--resume` 恢复训练时，**所有历史数据都丢失**
- 恢复后第一轮训练只有 1 轮新数据，相当于从零开始填窗口
- 滑动窗口的设计目的（保留近期多样化数据）完全失效

### alpha-zero-general 的做法

参考 `alpha-zero-general/Coach.py`：

```python
def saveTrainExamples(self, iteration):
    folder = self.args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(self.trainExamplesHistory)

def loadTrainExamples(self):
    modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
    examplesFile = modelFile + ".examples"
    if not os.path.isfile(examplesFile):
        log.warning(f'File "{examplesFile}" with trainExamples not found!')
        r = input("Continue? [y|n]")
        if r != "y":
            sys.exit()
    else:
        log.info("File with trainExamples found. Loading it...")
        with open(examplesFile, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
```

关键设计：

| 设计点 | 做法 |
|--------|------|
| **存储格式** | pickle 二进制 |
| **存储位置** | 与 checkpoint 同名 + `.examples` 后缀 |
| **存储时机** | 每轮训练后立刻保存 |
| **滑动窗口** | 只在内存中维护，磁盘上保留所有历史文件 |
| **加载** | resume 时根据指定 checkpoint 加载对应 examples |
| **失败处理** | 文件丢失时弹出 input() 让用户决定 |
| **垃圾回收** | 不删除旧文件 |

### 解决方案

```python
import pickle

def save_checkpoint(self, iteration, accepted_count=0, rejected_count=0):
    # 保存模型状态（已有）
    torch.save({...}, f"{self.checkpoint_path}_{iteration}.pt")

    # 单独保存训练历史（pickle）
    examples_path = f"{self.checkpoint_path}_{iteration}.pt.examples"
    with open(examples_path, "wb") as f:
        pickle.dump(self.train_examples_history, f)

def load_checkpoint(self, path):
    # 加载模型（已有）
    checkpoint = torch.load(path)
    ...

    # 加载训练历史
    examples_path = path + ".examples"
    if os.path.exists(examples_path):
        with open(examples_path, "rb") as f:
            self.train_examples_history = pickle.load(f)
        print(f"Loaded {len(self.train_examples_history)} iterations of training history")
    else:
        print(f"Warning: {examples_path} not found, starting with empty history")
        self.train_examples_history = []
```

### 文件大小估算

20 轮 × 100 局/轮 × ~30 步/局 × 6 数据增强 × ~3KB/样本 ≈ **~100 MB**

可以接受。alpha-zero-general 也是这个量级。

### 验证方法

1. 训练 5 轮，停止
2. 用 `--resume checkpoint_4.pt` 恢复
3. 检查日志确认 `Loaded 5 iterations of training history`
4. 第一轮恢复后训练应使用 5 轮的累积数据，而不是只 1 轮

---

## 优化项 3.7: 修复 `num_epochs` 命名误导 🟡 中优先级

### 问题

`AlphaZeroTrainer.__init__()` 中的 `num_epochs` 参数**名字误导**，实际行为不是传统意义上的 epoch。

当前代码 (`train.py`):

```python
mini_batch = random.sample(
    all_examples, min(len(all_examples), self.batch_size * self.num_epochs)
)

for i in range(0, len(mini_batch), self.batch_size):
    batch_states = states[i : i + self.batch_size]
    # ... forward, backward, update ...
```

实际行为：

| 配置 | 实际含义 |
|------|---------|
| `num_epochs=100, batch_size=128` | 取 12800 个样本，跑 100 个 batch（每个样本只看 1 次） |
| `num_epochs=10, batch_size=128` | 取 1280 个样本，跑 10 个 batch（每个样本只看 1 次） |

它实际上是 **"每轮迭代做几次梯度更新"**，不是真正的 "数据集跑几遍"。

### 与 alpha-zero-general 对比

alpha-zero-general 的 `epochs=10` 是**真正的 10 个 epoch**：
- 每个 epoch 把所有训练数据完整跑一遍
- 10 个 epoch 意味着每个样本被看 10 次

两个项目用同一个名字，但语义完全不同。

### 后果

- **理解混乱**：阅读代码或文档时会误以为模型在数据上跑了多遍
- **调参困难**：如果想模仿 alpha-zero-general 的 epochs=10，按当前实现实际只是 10 个梯度步
- **实际效果"歪打正着"**：把 `num_epochs` 从 100 改到 10 确实有效，但原因不是减少 epoch 数，而是减少梯度步数

### 解决方案

#### 方案 A: 改名 (推荐)

把 `num_epochs` 重命名为 `num_gradient_steps_per_iter`，反映真实语义。

```python
# train.py
def __init__(self, ..., num_gradient_steps_per_iter=10, ...):
    self.num_gradient_steps = num_gradient_steps_per_iter

# train_network()
mini_batch = random.sample(
    all_examples,
    min(len(all_examples), self.batch_size * self.num_gradient_steps),
)
```

#### 方案 B: 实现真正的 epochs

保留 `num_epochs` 名字，但让它真的工作：

```python
def train_network(self, examples):
    # ... 滑动窗口处理 ...
    
    all_examples = [...]
    
    for epoch in range(self.num_epochs):
        random.shuffle(all_examples)
        for i in range(0, len(all_examples), self.batch_size):
            batch = all_examples[i : i + self.batch_size]
            # train on batch
```

⚠️ 注意：方案 B 会大幅增加单轮训练时间（10 epochs × 数千 batch = 数万步梯度更新）。

### 推荐

**方案 A**（改名）。原因：
- 当前的 "10 个梯度步" 配置已被验证有效（v4 训练）
- 不会改变现有行为，只是修正命名
- 改成方案 B 需要重新调整其他超参数

### 验证

改名后所有测试仍应通过（命名变更不影响逻辑）。

---

## 优化项 4: 自适应反卡死机制 🟡 中优先级

### 问题

当前实现没有"卡住"检测。如果连续多轮 Arena 拒绝，训练会浪费大量算力但 best.pt 不变。

### 解决方案

在 `train()` 中添加：

```python
consecutive_rejects = 0
EARLY_STOP_THRESHOLD = 10  # 连续 10 次拒绝就停止
NOISE_BOOST_THRESHOLD = 5   # 连续 5 次拒绝就加大噪声

for iteration in range(...):
    # ... 训练 + arena ...
    
    if accepted:
        consecutive_rejects = 0
    else:
        consecutive_rejects += 1
        
        if consecutive_rejects >= NOISE_BOOST_THRESHOLD:
            old_weight = self.dirichlet_weight
            self.dirichlet_weight = min(0.5, self.dirichlet_weight * 1.5)
            print(f"连续 {consecutive_rejects} 轮拒绝，增大噪声: {old_weight} → {self.dirichlet_weight}")
        
        if consecutive_rejects >= EARLY_STOP_THRESHOLD:
            print(f"连续 {consecutive_rejects} 轮拒绝，停止训练")
            break
```

### 预期效果

- 检测到训练停滞时自动注入更多探索
- 避免长时间无效训练
- 节省算力

---

## 优化项 5: 学习率调度 🟢 低优先级

### 问题

学习率全程固定 0.001。后期模型应该精细收敛，使用更小的学习率。

### 解决方案

```python
self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
self.scheduler = optim.lr_scheduler.StepLR(
    self.optimizer, step_size=10, gamma=0.5
)  # 每 10 轮 lr 减半

# train() 末尾
self.scheduler.step()
```

### 预期效果

- 后期训练更稳定
- 减少震荡

---

## 优先级与实施顺序

1. **先做**：优化项 1（分段温度）+ 优化项 2（降低 epochs）— 这两项最可能直接提升战力
2. **再做**：优化项 3（缩小 buffer）— 配合前两项进一步提升数据质量
3. **观察**：优化项 4（反卡死）— 在多次训练中观察是否真的需要
4. **可选**：优化项 5（学习率调度）— 收益较小，最后考虑

## 决策依据

当前 `001-arena-model-selection` 训练完成后：

- 如果 `best.pt` vs random 胜率 ≥ 50% → 继续当前策略，仅做 buffer 优化
- 如果胜率 30%–50% → 实施优化项 1 + 2
- 如果胜率 < 30% → 实施所有 🔴 和 🟡 优化项

## 参考

### 代码参考

- `alpha-zero-general/main.py`: 默认参数配置
- `alpha-zero-general/Coach.py`: 训练循环、滑动窗口实现
- `alpha-zero-general/MCTS.py`: 温度策略
- 本项目当前实现: `train.py`, `main.py`, `mcts/mcts.py`

### 学术论文

#### 滑动窗口 / 经验回放 (Experience Replay)

- **DQN (经验回放的理论基础)**:
  - Mnih et al., "Playing Atari with Deep Reinforcement Learning", arXiv:1312.5602, 2013
  - Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518, 529–533, 2015
  - 首次提出经验回放缓冲区，用于打破训练样本的时间相关性

- **AlphaGo Zero (将 replay buffer 用于 self-play)**:
  - Silver et al., "Mastering the game of Go without human knowledge", *Nature* 550, 354–359, 2017
  - Methods 章节明确使用 "**最近 500,000 局** self-play 数据" 作为训练 buffer

- **AlphaZero (推广到多种棋类)**:
  - Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", *Science* 362, 1140–1144, 2018

#### 注意

- "最近 20 轮" 这个具体数字 **没有论文出处**，是 `alpha-zero-general` 作者基于教学项目规模的工程经验值
- DeepMind 论文使用的是**最近 N 局**（按对局数计算），不是**最近 N 轮**（按训练迭代计算）
- 严谨引用时应使用 **AlphaGo Zero (Nature 2017)** 的 Methods 部分作为权威依据

#### 温度退火 (Temperature Annealing)

- **AlphaGo Zero**：Methods 部分描述 self-play 中前 30 步使用 temperature=1（探索），之后切换为接近确定性的选择
- 这是优化项 1 (分段温度) 的论文依据
