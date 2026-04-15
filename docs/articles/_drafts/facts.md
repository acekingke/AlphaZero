# facts.md — 文章素材卡片

> 用途：为"从 5% 到 100%"技术复盘文章备稿。所有数值均注明来源，冲突以 playbook 为准。

---

## 关键数值（vs random 胜率）

- **v1（pre-arena baseline）**：~2–8%（区间），来源：`memory/project_alphazero.md` 行 18："Pre-arena baseline: ~2-8% win rate vs random (very weak, model was getting worse over time)"
- **v3（epochs=100, 无 sliding window → 有 sliding window）**：8% vs random，after 20 iters，4 accepts / 16 rejects，来源：`memory/project_alphazero.md` 行 18
- **v4（epochs=10, sliding window=20）**：**22%** vs random（50 games, 50 MCTS sims），来源：`memory/project_v4_training.md` 行 42："vs random win rate: 22.0%（4% draw, 74% loss）"
  - 对比关系：v1 ~5% → v3 8% → v4 22%（来源同上）
- **v16（全面对齐 alpha-zero-general）峰值**：**13.3%** vs random，来源：`docs/training-debug-playbook.md`（playbook）："v16 原版峰值 vs_random 13.3%"（第 939 行）
- **v17–v19（tree 实现 + 各种调参）天花板**：**10–13%**，来源：`docs/training-debug-playbook.md` 行 939："多次换参数（Dirichlet、epochs、threshold）都无效"；`memory/project_mcts_tree_reuse.md`："v17 … v18, v19 tuned Arena threshold, Dirichlet, epochs → still stuck at 10-13%"
- **v20（dict-based MCTS）**：**第 8 轮打到 100% vs random**，来源：`memory/project_mcts_tree_reuse.md`："v20 rewrote MCTS to dict-based → iter 8 hit 100% vs random"；playbook 确认："v20 iter 8 vs_random 100%"（行 987）

---

## 诊断实验

实验时间：v17 卡住期（2026-04-14 左右）

| 测试 | 数值 | 含义 |
|------|------|------|
| 纯策略（绕过 MCTS）vs random | **43.3%** | 网络本身没坏 |
| MCTS 25 sim vs random | **6.7%** | MCTS 把 43% 砍到 6.7%，真凶确认 |

- 来源：`docs/training-debug-playbook.md` 行 957–961（playbook 原文）：
  ```
  iter 0 冷网络      裸策略:  43.3%  | MCTS 25 sim:  6.7%
  ```
- `memory/project_mcts_tree_reuse.md` 描述一致（"43%/7%"为近似写法）
- 脚本路径（来自 memory）：`scripts/exp_pure_policy_vs_random.py` + `scripts/exp_mcts_reuse_vs_random.py`

**额外数据点**（playbook 行 986）：v19 iter 2 模型的 MCTS 胜率：树实现 13.3% → 字典实现 63.3%（5x）

---

## 训练配置（v4）

来源：`memory/project_v4_training.md`

| 参数 | 值 |
|------|----|
| 总迭代数 | 40 |
| 每迭代自对弈局数 | 100 games/iter |
| MCTS sims（训练） | 50 |
| MCTS sims（arena 评估） | 25 |
| num_epochs | 10（gradient epochs） |
| batch_size | 128 |
| Arena 判定局数 | 40 games |
| Arena 接受阈值 | 60% |
| Sliding window | 20 iterations |
| 多进程 | 6 workers, MPS |

**各阶段 accept/reject**（来源同上）：

| 阶段 | 迭代 | Accepts | Rejects | 备注 |
|------|------|---------|---------|------|
| Early | 0-12 | 3 | 10 | 77% rejection |
| Mid | 13-19 | 3 | 4 | 57% rejection，两次 100% 胜 |
| Window onset | 20-25 | 0 | 6 | **100% rejection**（滑窗首次丢掉 iter_0，分布突变） |
| Recovery | 26 | 1 | 0 | 100% win rate，冲击是可恢复的 |
| Post-recovery | 27-28 | 0 | 2 | — |

最终：8 accepted / 32 rejected（20% acceptance rate），accepted iters: 0, 2, 4, 13, 14, 19, 26, 35，best model: `checkpoint_35.pt`

---

## 代码片段 1：num_epochs 语义对比

### 旧实现（错）

无法从 git 历史或现有文件中找到旧代码原文。根据 `memory/project_num_epochs_bug.md` 的语义描述：

> "在旧版中，num_epochs 实际上是 gradient steps（梯度更新次数），不是'过一遍数据集'的意义。
> for 循环只跑 num_epochs 次，每次随机抽一个 batch 做一次梯度更新。
> num_epochs=100 实际只对 100×batch_size 个样本做了 100 次单步更新，
> 没有'epoch=遍历整个 buffer'的概念。"

（这是 fallback 方案：playbook 无旧代码片段，memory 只有语义描述，未保留旧代码。）

参照 `docs/alpha-zero-general-alignment.md` 差异 3 节，旧版也曾使用无放回 shuffle：
```python
perm = torch.randperm(n)
for batch_start in range(0, n, self.batch_size):
    batch_idx = perm[batch_start : batch_start + self.batch_size]
```

### 新实现（正确，对齐 alpha-zero-general）

来源：`train.py` 行 641–645

```python
for epoch in range(self.num_epochs):
    # Random sampling WITH replacement — matches alpha-zero-general:
    #   sample_ids = np.random.randint(len(examples), size=args.batch_size)
    for _ in range(batch_count):
        sample_ids = np.random.randint(n, size=self.batch_size)
```

其中 `batch_count = int(n / self.batch_size)`（`train.py` 行 623）。
每个 epoch 做 `batch_count` 次有放回随机采样，`num_epochs=10` 约等于对每个样本期望看 10 次。

---

## 代码片段 2：MCTS 五字典声明

> **注意**：memory 里写的是"六字典（Qsa/Nsa/Ns/Ps/Es/Vs）"，但当前 `mcts/mcts.py` 实际只有**五个字典**（无 Es）。以代码为准。

来源：`mcts/mcts.py` 行 72–77

```python
# alpha-zero-general's six dicts, keyed by canonical state bytes
self.Qsa = {}   # (s_key, a) -> Q value (from s_key's current player's perspective)
self.Nsa = {}   # (s_key, a) -> visit count
self.Ns = {}    # s_key -> sum of child visits
self.Ps = {}    # s_key -> masked+normalized policy prior vector
self.Vs = {}    # s_key -> valid-moves mask
```

（代码注释写 "six dicts" 但实际初始化了五个。Es 在 alpha-zero-general 原版用于缓存终局结果，当前实现未使用该字典。）

---

## 代码片段 3：局面做 key 的一行

来源：`mcts/mcts.py` 行 130

```python
return canonical_state.astype(np.int8).tobytes()
```

完整上下文（行 127–130）：
```python
@staticmethod
def _state_key(canonical_state):
    """Use raw bytes of int canonical board as hashable key. Fast and exact."""
    return canonical_state.astype(np.int8).tobytes()
```

> memory 里写的是 `board.tostring()`（alpha-zero-general 原版用法，NumPy 旧 API），
> 当前代码已改为 `.tobytes()`（NumPy 新 API，等价）。以当前代码为准。

---

## 关键引用句

**v20 结果描述**（来源：`memory/project_mcts_tree_reuse.md`）：
> "v20 rewrote MCTS to dict-based (Qsa/Nsa/Ns/Ps/Vs keyed by canonical state bytes) → **iter 8 hit 100% vs random**"

playbook 版本（行 987，更有力）：
> "v20 iter 8 vs_random 100%，8 轮就突破了 v16 花 17 轮才到的 13.3% 天花板"

**MCTS 天花板故事的一句话总结**（来源：`memory/project_mcts_tree_reuse.md`）：
> "同一canonical position reached by different move orders = TWO independent nodes in a tree, stats not shared. In dict-based MCTS they share. With only 25 sims/move, the shared-stats amplification is critical — trees effectively have 1-2x stats per unique state, dicts have 5-10x."

playbook 的诊断总结（行 961）：
> "**纯策略 43% 说明网络本身没坏。MCTS 把 43% 砍到 6.7% 才是真凶**。"

---

## 开场/收束候选（备选，写作时再挑）

- **开头钩子**："我训练了一个 AlphaZero。它打不过随机下棋。"
- **第一幕末**："调完这三件事，胜率爬到了 22%。我以为故事结束了。"
- **第二幕末**："我盯着那个 43% 看了很久。如果让搜索介入反而让模型变弱，那答案只有一个 —— 搜索本身坏了。"
- **第三幕末**："前面调了三周的所有超参、加的网络容量、修的代码 bug，都没这一行 `s = board.tostring()` 管用。"

---

## 数值冲突备注

| 项目 | memory 写法 | playbook / 代码实际 | 采用 |
|------|-------------|---------------------|------|
| MCTS 字典数量 | "六字典（Qsa/Nsa/Ns/Ps/Es/Vs）" | 代码实际只有五个（无 Es）；注释误写 "six" | **代码实际（五个）** |
| key 生成方法 | `board.tostring()` | `.tobytes()`（NumPy 新 API） | **代码实际（tobytes）** |
| MCTS 25 sim 胜率 | "7%" | playbook 精确值 "6.7%" | **playbook（6.7%）** |
| v4 final 胜率 | memory 写 "22%" | 与 playbook 一致，无冲突 | 两者一致 |

---

## 事实自检

- [x] 所有数值都有明确来源（文件路径 + 行号或引文）
- [x] 三处代码片段都是从真实代码摘的（不是编造的）：
  - 片段 1（新 train_network）：`train.py` 行 641–645
  - 片段 2（五字典初始化）：`mcts/mcts.py` 行 72–77
  - 片段 3（tobytes key）：`mcts/mcts.py` 行 130
  - 片段 1 旧实现：无原始代码，使用 fallback 语义描述（已标注）
- [x] 冲突的数值选了 playbook / 代码实际为准（均已在"数值冲突备注"表格中标注）
