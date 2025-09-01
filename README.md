# AlphaZero 黑白棋（奥赛罗）

本项目使用 PyTorch 为黑白棋（又称奥赛罗）实现了 AlphaZero 算法。AlphaZero 是 DeepMind 开发的深度强化学习算法，它结合了蒙特卡洛树搜索（MCTS）和深度神经网络，在棋类游戏中达到超人水平的表现。

## 项目结构

```
AlphaZero/
├── env/
│   └── othello.py         # 黑白棋游戏环境
├── models/
│   └── neural_network.py  # 神经网络架构
├── mcts/
│   └── mcts.py            # 蒙特卡洛树搜索实现
├── utils/                 # 工具函数
├── train.py               # 训练模块
├── play.py                # 与训练好的模型对战
├── evaluate.py            # 评估模型性能
├── main.py                # 主入口点
└── requirements.txt       # 项目依赖
```

## Requirements

本项目推荐使用 Conda 来管理环境，同时也支持使用 pip。

### 使用 Conda (推荐)

项目根目录下的 `environment.yml` 文件包含了所有必需的依赖项。

1.  **创建 Conda 环境:**
    打开终端，运行以下命令来创建名为 `alphazero_env` 的环境。
    ```bash
    conda env create -f environment.yml
    ```

2.  **激活环境:**
    在运行任何项目脚本之前，都需要先激活该环境。
    ```bash
    conda activate alphazero_env
    ```

### 使用 Pip

如果您不使用 Conda，也可以通过 `requirements.txt` 文件来安装依赖。

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm

运行以下命令安装：
```bash
pip install -r requirements.txt
```

## 工作原理

AlphaZero 由三个主要组件组成：

1. **神经网络**：一个深度残差网络，接收游戏状态作为输入并输出：
   - 策略（所有可能走法的概率分布）
   - 价值（游戏结果的预估）

2. **蒙特卡洛树搜索（MCTS）**：使用神经网络的预测来指导搜索最佳走法。

3. **自我对弈训练**：模型通过与自己对弈并从生成的数据中学习来不断提升。

## Usage

您可以通过以下两种方式之一来运行项目脚本：

1.  **使用 `conda run` (推荐):**
    这种方式无需手动激活环境，可以直接在命令前指定环境名称。下面的所有示例都将采用这种方式。

2.  **先激活环境:**
    您也可以先激活环境，然后在该终端会话中直接运行 `python` 命令。
    ```bash
    conda activate alphazero_env
    # 之后就可以直接运行, 例如: python main.py train ...
    ```

### Training

从零开始训练 AlphaZero 模型:

```bash
conda run -n alphazero_env python main.py train - `--iterations`: 50 # 训练迭代次数，每次迭代包含了自我对弈、训练和评估三个步骤
- `--self_play_games`: 100 # 每次训练迭代中，自我对弈的对局数目
- `--mcts_simulations`: 100 # 每个动作的MCTS搜索模拟次数。训练推荐100次（平衡探索与速度），人机对战推荐200次（更好的性能），评估和比较推荐100次
- `--c_puct`: 3.0 # MCTS中的UCB1公式中的参数，控制了探索与利用的权重，建议值为3.0 --use_mps
```


例如，基于上述默认参数的训练示例命令（快速启动）：

```bash
conda run -n alphazero_env python main.py train --iterations 50 --self_play_games 100 \
    --mcts_simulations 25 --c_puct 3 --use_mps
```

从检查点恢复训练：

```bash
conda run -n alphazero_env python main.py train --resume ./models/checkpoint_10.pt --use_mps
```

从命令行设置 MCTS 探索常数（`c_puct`）（默认为 1.0）。例如，从 `checkpoint_7.pt` 恢复训练并使用 `c_puct=2.5`：

```bash
conda run -n alphazero_env python main.py train --resume ./models/checkpoint_7.pt --c_puct 2.5 --use_mps
```

您还可以在创建训练器时设置其他训练超参数，例如：

```bash
conda run -n alphazero_env python main.py train --iterations 20 --self_play_games 100 --mcts_simulations 25 --batch_size 256 --c_puct 3 --use_mps
```

您可以启用多进程来加速自我对弈生成（推荐在 CPU 核心较多的系统上使用）：

```bash
conda run -n alphazero_env python main.py train --iterations 20 --self_play_games 100 \
    --mcts_simulations 25 --c_puct 3 --use_multiprocessing --mp_num_workers 4 --mp_games_per_worker 5
```

注意：
- `--use_multiprocessing` 启用自我对弈的多进程处理（使用 `multiprocessing.Pool`）。
- `--mp_num_workers` 设置工作进程数（如果未提供，默认为 CPU 核心数 - 1）。
- `--mp_games_per_worker` 控制每个工作进程每个任务运行多少局游戏；增加此值可减少进程调度开销。

### 与训练好的模型对战

与训练好的模型对战：

```bash
conda run -n alphazero_env python main.py play --model ./models/checkpoint_XXX.pt --player_color white --mcts_simulations 200
```

### 评估模型

对抗随机玩家评估模型：

```bash
conda run -n alphazero_env python main.py evaluate --model ./models/checkpoint_XXX.pt --num_games 20 --mcts_simulations 100
```

### 比较不同模型

比较两个不同的模型：

```bash
python main.py compare --model1 ./models/checkpoint_XXX.pt --model2 ./models/checkpoint_YYY.pt --num_games 20 --mcts_simulations 100
```

## 黑白棋规则

黑白棋在 8×8 的棋盘上进行。游戏开始时中央有四个棋子：两黑两白。玩家轮流放置自己颜色的棋子，黑方先行。

有效走法必须：
1. 放在空格上
2. 至少吃掉对手的一个棋子

吃子是通过在新放置的棋子和玩家已有的棋子之间"夹住"对手的棋子来实现的。

游戏结束条件：
1. 双方连续跳过（无子可走）
2. 棋盘完全填满

棋盘上棋子多的玩家获胜。

## 实现细节

### 神经网络架构

神经网络由以下部分组成：
- 输入卷积层
- 多个残差块
- 两个输出头：
  - 策略头：输出每个走法的概率
  - 价值头：预估游戏结果

### MCTS 算法

蒙特卡洛树搜索算法使用神经网络指导搜索：
1. **选择**：使用 UCB（上置信界）公式遍历树
2. **扩展和评估**：使用神经网络评估新位置
3. **反向传播**：更新树中的值
4. **选择走法**：选择最有希望的走法

MCTS在每个节点的选择阶段使用UCB公式：

$$Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

其中：
- $Q(s, a)$ 是行动 $a$ 在状态 $s$ 的预期奖励
- $P(s, a)$ 是神经网络预测的先验概率
- $N(s, a)$ 是行动 $a$ 在状态 $s$ 被访问的次数
- $c_{\text{puct}}$ 是控制探索与利用平衡的常数

### 训练过程

1. **自我对弈**：当前模型与自己对弈生成训练数据
2. **训练**：神经网络在生成的数据上进行训练
3. **重复**：改进后的模型用于更多的自我对弈，形成学习循环

#### 关键损失函数

训练过程使用两个损失函数的组合：

**策略损失函数**：
$$\mathcal{L}_{\text{policy}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{a} \pi_i(a) \log(p_\theta(a|s_i))$$

其中:
- $N$ 是批次大小
- $\pi_i(a)$ 是从MCTS得到的策略分布
- $p_\theta(a|s_i)$ 是神经网络预测的策略分布
- $s_i$ 是游戏状态
- $a$ 表示可能的动作

**价值损失函数**：
$$\mathcal{L}_{\text{value}} = \frac{1}{N} \sum_{i=1}^{N} (v_i - V_\theta(s_i))^2$$

其中:
- $v_i$ 是游戏结果的真实值
- $V_\theta(s_i)$ 是神经网络预测的价值

**总损失函数**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$$

## 未来改进

- 实现多线程以加速自我对弈
- 添加图形用户界面以便与模型对战
- 扩展到其他棋盘游戏
- 优化超参数
- 实现渐进式课程学习训练

## 许可证

MIT