# AlphaZero 6x6 Othello

参考 [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 的 AlphaZero 复现，在 6x6 黑白棋上训练和对弈。

## 项目结构

```
AlphaZero/
├── env/
│   └── othello.py              # 6x6 黑白棋游戏环境
├── models/
│   └── neural_network.py       # 残差卷积网络（4 层残差块 + 策略/价值双头）
├── mcts/
│   └── mcts.py                 # MCTS（字典实现，以局面 bytes 为 key）
├── utils/
│   ├── data_augmentation.py    # 棋盘对称性数据增强
│   ├── device.py               # MPS/CUDA/CPU 设备选择
│   ├── training_logger.py      # 训练日志
│   └── visualization.py        # 训练曲线可视化
├── scripts/
│   ├── exp_pure_policy_vs_random.py   # 诊断：纯策略 vs random
│   ├── exp_mcts_reuse_vs_random.py    # 诊断：MCTS vs random
│   ├── exp_value_head_sanity.py       # 诊断：价值头检查
│   └── plot_csv_losses.py             # 绘制 loss 曲线
├── tests/                      # 测试套件（13 个测试文件）
├── train.py                    # 训练器（Arena 门控 + 滑动窗口）
├── play.py                     # 命令行对弈
├── gui_play.py                 # GUI 对弈
├── evaluate.py                 # 模型评估
├── main.py                     # CLI 入口
├── requirements.txt
└── environment.yml
```

## 安装

```bash
# Conda（推荐）
conda env create -f environment.yml
conda activate alphazero_env

# 或 pip
pip install -r requirements.txt
```

依赖：Python 3.11+、PyTorch、NumPy、tqdm、matplotlib

## 训练

```bash
# 从零开始训练
python train.py

# 从检查点恢复
python main.py train --resume ./models/checkpoint_10.pt --use_mps

# 多进程加速自我对弈
python main.py train --iterations 40 --self_play_games 100 \
    --mcts_simulations 50 --c_puct 3 \
    --use_multiprocessing --mp_num_workers 6 --use_mps
```

关键训练参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iterations` | 40 | 训练迭代数（每轮 = 自我对弈 + 训练 + Arena 评估） |
| `--self_play_games` | 100 | 每轮自我对弈局数 |
| `--mcts_simulations` | 50 | 训练时每步 MCTS 模拟次数 |
| `--num_epochs` | 10 | 每轮训练的 epoch 数 |
| `--batch_size` | 128 | 训练 batch 大小 |
| `--c_puct` | 3.0 | UCB 探索常数 |
| `--arena_games` | 40 | Arena 评估局数 |
| `--arena_threshold` | 0.6 | Arena 接受阈值（新模型胜率需超过此值） |
| `--sliding_window` | 20 | 训练数据滑动窗口（保留最近 N 轮数据） |

## 对弈

```bash
# GUI 对弈
python gui_play.py --mcts_simulations 150 --c_puct 2.5

# 命令行对弈
python main.py play --model ./models/best.pt --mcts_simulations 200
```

## 评估

```bash
# vs random
python main.py evaluate --model ./models/best.pt --num_games 50

# 两个模型对战
python main.py compare --model1 ./models/checkpoint_A.pt \
    --model2 ./models/checkpoint_B.pt --num_games 20
```

## 诊断脚本

当训练遇到问题时，`scripts/` 下的诊断脚本可以帮助定位原因：

```bash
# 纯策略 vs random（绕过 MCTS，检验网络本身是否学到了东西）
python scripts/exp_pure_policy_vs_random.py

# MCTS vs random（检验搜索是否在帮助还是伤害网络）
python scripts/exp_mcts_reuse_vs_random.py
```

如果纯策略胜率远高于 MCTS 胜率，说明搜索在伤害网络——问题出在 MCTS 实现，不在网络。

## 测试

```bash
pytest
```

## 核心设计

- **MCTS 用字典实现**（不是树）：以 `canonical_state.tobytes()` 为 key，同构局面共享搜索统计。这在搜索预算有限（25 sims）时至关重要。
- **Arena 门控**：新模型必须在 40 局中胜率超过 60% 才被接受，防止退化。
- **滑动窗口**：只保留最近 20 轮自我对弈数据，防止旧策略污染新策略。
- **vs random 评估**：每轮训练后和固定的随机对手打 50 局，作为不会漂移的外部 baseline。

## 许可证

MIT
