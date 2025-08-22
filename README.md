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

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm

Install dependencies:

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

### Training

Train the AlphaZero model from scratch:

```bash
python main.py train --iterations 50 --self_play_games 100 --mcts_simulations 800
```

Resume training from a checkpoint:

```bash
python main.py train --resume ./models/checkpoint_10.pt
```

### Playing Against the Model

Play against the trained model:

```bash
python main.py play --model ./models/checkpoint_49.pt --player_color black
```

### Evaluating the Model

Evaluate the model against a random player:

```bash
python main.py evaluate --model ./models/checkpoint_49.pt --num_games 50
```

### Comparing Models

Compare two different models:

```bash
python main.py compare --model1 ./models/checkpoint_30.pt --model2 ./models/checkpoint_49.pt --num_games 50
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