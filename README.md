# AlphaZero for Othello (Reversi)

This project implements AlphaZero for the game of Othello (also known as Reversi) using PyTorch. AlphaZero is a deep reinforcement learning algorithm developed by DeepMind that combines Monte Carlo Tree Search (MCTS) with deep neural networks to achieve superhuman performance in board games.

## Project Structure

```
AlphaZero/
├── env/
│   └── othello.py         # Othello game environment
├── models/
│   └── neural_network.py  # Neural network architecture
├── mcts/
│   └── mcts.py            # Monte Carlo Tree Search implementation
├── utils/                 # Utility functions
├── train.py               # Training module
├── play.py                # Play against the trained model
├── evaluate.py            # Evaluate model performance
├── main.py                # Main entry point
└── requirements.txt       # Project dependencies
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

## How It Works

AlphaZero consists of three main components:

1. **Neural Network**: A deep residual network that takes the game state as input and outputs:
   - A policy (probability distribution over all possible moves)
   - A value (estimated outcome of the game)

2. **Monte Carlo Tree Search (MCTS)**: Uses the neural network's predictions to guide the search for the best move.

3. **Self-Play Training**: The model improves by playing against itself and learning from the generated data.

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
conda run -n alphazero_env python main.py train --iterations 50 --self_play_games 100 --mcts_simulations 800 --use_mps
```

Resume training from a checkpoint:

```bash
conda run -n alphazero_env python main.py train --resume ./models/checkpoint_10.pt --use_mps
```

Set the MCTS exploration constant (`c_puct`) from the command line (default is 1.0). For example, to resume training from `checkpoint_7.pt` and use `c_puct=2.5`:

```bash
conda run -n alphazero_env python main.py train --resume ./models/checkpoint_7.pt --c_puct 2.5 --use_mps
```

You can also set other training hyperparameters when creating the trainer, for example:

```bash
conda run -n alphazero_env python main.py train --iterations 20 --self_play_games 150 --mcts_simulations 1000 --batch_size 256 --c_puct 2.5 --use_mps
```

### Playing Against the Model

Play against the trained model:

```bash
conda run -n alphazero_env python main.py play --model ./models/checkpoint_49.pt --player_color black --use_mps
```

### Evaluating the Model

Evaluate the model against a random player:

```bash
conda run -n alphazero_env python main.py evaluate --model ./models/checkpoint_7.pt --num_games 50 --mcts_simulations 800
```

### Comparing Models

Compare two different models:

```bash
conda run -n alphazero_env python main.py compare --model1 ./models/checkpoint_30.pt --model2 ./models/checkpoint_49.pt --num_games 50 --use_mps
```

## Rules of Othello

Othello is played on an 8x8 board. The game starts with four pieces in the center: two black and two white. Players take turns placing pieces of their color, with black going first.

A valid move must:
1. Be placed on an empty square
2. Capture at least one of the opponent's pieces

Capture occurs by "sandwiching" opponent pieces between the newly placed piece and another piece of the current player's color.

The game ends when:
1. Both players pass consecutively
2. The board is completely filled

The player with more pieces on the board wins.

## Implementation Details

### Neural Network Architecture

The neural network consists of:
- An input convolutional layer
- Multiple residual blocks
- Two heads:
  - Policy head: Outputs probabilities for each move
  - Value head: Estimates the game outcome

### MCTS Algorithm

The Monte Carlo Tree Search algorithm uses the neural network to guide the search:
1. **Selection**: Traverse the tree using UCB (Upper Confidence Bound) formula
2. **Expansion and Evaluation**: Use the neural network to evaluate new positions
3. **Backpropagation**: Update values in the tree
4. **Play**: Select the most promising move

### Training Process

1. **Self-play**: The current model plays against itself to generate training data
2. **Training**: The neural network is trained on the generated data
3. **Repeat**: The improved model is used for more self-play, creating a learning cycle

## Future Improvements

- Implement multi-threading for faster self-play
- Add a GUI for playing against the model
- Extend to other board games
- Optimize hyperparameters
- Implement a progressive curriculum for training

## License

MIT