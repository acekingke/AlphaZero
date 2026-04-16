# Sample / Batch / Epoch / Iteration 概念详解

> **背景**：本项目曾因 `num_epochs` 参数命名误导，导致训练数据利用率只有 0.3%。本文解释这些核心概念，避免再次踩坑。

## 核心概念

### 1. Sample (样本)

**一条训练数据**。最小的学习单元。

在 AlphaZero 项目里：
- 一个 sample = `(state, policy, value)` 元组
- `state`: 棋盘状态（神经网络的输入）
- `policy`: MCTS 给出的动作概率分布（policy head 的目标）
- `value`: 这一手最终的胜负结果（value head 的目标）

每轮 self-play 生成约 22,000 个 samples（100 局 × ~30 步 × ~7 次数据增强）。

### 2. Batch (批次)

**模型一次前向+反向传播处理的一组样本**。`batch_size = 128` 意思是"每次训练步骤一起算 128 个样本的损失，然后更新一次权重"。

**为什么用 batch 而不是逐个样本**：
- **效率**：GPU 并行算 128 个样本几乎和算 1 个一样快
- **梯度更稳定**：128 个样本的平均梯度噪声比单个小

**1 个 batch = 1 次梯度更新**。

### 3. Epoch (轮次)

**所有训练数据被完整看一遍** 叫一个 epoch。

如果数据池有 446,000 samples，batch_size=128：
- 1 epoch = 446,000 / 128 = **3,484 个 batches**
- 1 epoch 后，**每个样本恰好被看了 1 次**

`num_epochs=2` 意思是：把整个数据池**完整跑 2 遍**。每次开始新 epoch 前**重新洗牌**，所以同一个样本两次出现的位置不同。

### 4. Iteration (迭代) — 容易混的词

**这个词有两种含义**，要看上下文：

| 含义 | 典型用法 |
|------|---------|
| **训练步骤** = 1 个 batch 的 forward/backward/update | "我训练了 1000 iterations" 通常指 batches |
| **AlphaZero 轮次** = self_play + train + arena 一整轮 | 本项目 `--num_iterations` 用的是这个 |

⚠️ 看到 "iterations" 一定要先确认作者指哪个含义。

## 用图表示层级

```
1 个 sample = 1 条 (state, policy, value) 数据
                    ↓ 集合
1 个 batch = batch_size 个 sample (例如 128)
                    ↓ 集合
1 个 epoch = 数据池中所有 sample 都被看一次 (例如 3,484 个 batch)
                    ↓ 集合
1 次 train_network() = num_epochs 个 epoch
                    ↓ 集合
1 个 AlphaZero iteration = self_play + train_network() + arena_evaluation
                    ↓ 集合
完整训练 = num_iterations 个 AlphaZero iteration
```

## 用具体数字（本项目典型配置）

| 概念 | 数量 |
|------|------|
| 1 sample | 1 条数据 |
| 1 batch | 128 samples |
| 1 epoch (446K 样本池) | **3,484 batches** = 446,000 samples 被看一次 |
| 1 train_network() (num_epochs=2) | 2 epochs = 6,968 batches = 892,000 sample-views |
| 1 AlphaZero iteration | 1 train_network() + ~22,000 个新 self-play sample + 40 局 Arena |
| 完整训练 (num_iterations=30) | 30 个 AlphaZero iteration |

## ⚠️ 本项目曾经的 num_epochs Bug

### 旧代码 (错误)

```python
def train_network(self, examples):
    # ... 加入数据池 ...
    
    # 从池里随机抽 batch_size * num_epochs 个样本
    mini_batch = random.sample(all_examples, batch_size * num_epochs)
    # = random.sample(446000, 128 * 10) = 1280 个样本
    
    # 切成 num_epochs 个 batch
    for i in range(0, len(mini_batch), batch_size):
        # 总共 10 个 batch
        train_one_batch(...)
```

它实际做了什么：
1. 从 446K 中**随机抽 1,280 个样本**
2. 切成 10 个 batch
3. 每个 batch 训练一次
4. 总共 10 个 batch 后停止

**这不是 10 个 epoch！** 这是**单次"小型采样训练"**：
- 看了 1,280 个样本（每个看 1 次）
- 没看的有 **444,720 个 (99.7%)**

但参数名叫 `num_epochs=10`，会让人以为"对所有数据跑了 10 遍"。**完全不是**。

### 新代码 (正确)

```python
def train_network(self, examples):
    # ... 加入数据池 ...
    
    n = len(all_examples)  # 446,000
    
    for epoch in range(self.num_epochs):  # 真正的 epochs
        perm = torch.randperm(n)  # 每个 epoch 重新洗牌
        for batch_start in range(0, n, batch_size):
            batch_idx = perm[batch_start : batch_start + batch_size]
            train_one_batch(all_examples[batch_idx])
```

它实际做了什么：
1. 把所有 446K 数据 shuffle
2. 切成 ~3,484 个 batch
3. 每个 batch 训练一次
4. 重复 num_epochs 次 (每次重新 shuffle)
5. 总共 num_epochs × 3,484 个 batch
6. 每个样本被看了 num_epochs 次

### 数字对比

| 指标 | 旧 (num_epochs=10) | 新 (num_epochs=2) |
|------|---------------------|---------------------|
| 处理的 batch 数 | 10 | ~7,000 |
| 样本被看次数 | 1,280 | 892,000 |
| 数据池中样本平均被看的次数 | 0.003 次 | 2 次 |
| 训练时间 | 0.5 秒 | 3 分钟 |
| 数据利用率 | 0.3% | 200% |

**新代码的 num_epochs=2 比旧代码的 num_epochs=10 多训练了约 700 倍**。

## 类比记忆

把训练数据想象成一本书 (一本书 = 一个 epoch 的内容)：

- **Sample** = 一句话
- **Batch** = 一段（128 句话）
- **Epoch** = 把整本书读一遍
- **num_epochs=2** = 把整本书读 2 遍

**旧 bug 相当于**：你说"我要读这本书 10 遍"，但实际上你随机翻到了几页，读了 10 段就合上了书。你觉得你读了 10 遍，其实总共只读了不到 1 页。

新代码是真的读 2 遍。

## 为什么 batch 和 epoch 容易混

它们是**正交的两个维度**：

- **Batch** 是**空间维度**：一次处理多少样本（横向）
- **Epoch** 是**时间维度**：所有样本被处理几遍（纵向）

```
                    epoch 1                  epoch 2
    sample 0  ┌─────────────────────┐  ┌─────────────────────┐
    sample 1  │                     │  │                     │
    sample 2  │                     │  │                     │
    sample 3  │                     │  │                     │
        ...   │       ...           │  │       ...           │
    sample N  └─────────────────────┘  └─────────────────────┘
              ↑ batch_size ↑
```

每个 batch 是一个垂直切片（多个样本一起处理），每个 epoch 是一个完整的横向覆盖（所有样本被处理一次）。

**公式**：

```
total_batches_processed = num_epochs × batches_per_epoch
                       = num_epochs × ceil(n_samples / batch_size)
```

## 选 num_epochs 的考量

| num_epochs | 优点 | 缺点 |
|------------|------|------|
| 1 | 训练快，每个样本只看一次 | 模型可能没充分学习 |
| **2-5** | **平衡**，alpha-zero-general 用 10 | — |
| 10+ | 充分学习 | 慢；过多可能过拟合（特别是数据少时）|

本项目当前默认 `num_epochs=2`，每轮训练约 3 分钟，30 轮 iteration 约 3 小时。

## 选 batch_size 的考量

| batch_size | 优点 | 缺点 |
|-----------|------|------|
| 32-64 | 梯度噪声大，利于跳出局部最优 | 慢，GPU 利用率低 |
| **128-256** | **平衡**，GPU 利用率高 | — |
| 512+ | 训练快，梯度稳定 | 内存占用高，可能陷入"尖锐"最优解 |

本项目用 `batch_size=128`。

## 检查清单：评估训练设置是否合理

每次开始新训练前，问自己：

- [ ] 数据池有多大？(check `len(all_examples)`)
- [ ] 1 个 epoch 是多少 batches？(`n / batch_size`)
- [ ] num_epochs 设的是什么含义？(真 epochs vs 梯度步数)
- [ ] 每轮迭代真的会看完所有数据吗？(`num_epochs * n` vs 实际处理量)
- [ ] 训练时间预估对吗？(`num_epochs * batches_per_epoch * time_per_batch`)
- [ ] 数据利用率多少？(被看次数 / 总样本数)

如果数据利用率 < 50%，就有问题。

## 进一步阅读

- [Andrej Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- [PyTorch DataLoader docs](https://pytorch.org/docs/stable/data.html) — 标准的 epoch + shuffle 实现
- 本项目 `docs/training-debug-playbook.md` — 训练 debug 的 12 个技巧
- 本项目 `specs/001-arena-model-selection/future-optimizations.md` — 优化项 3.7 详述这个 bug
