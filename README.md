# StarChess.AI 2025

Welcome to ACM AI's Spring 2025 competition, StarChess.AI!

## Rule

Our chess rules are slightly different from the classical chess. Please check [`rules.md`](rules.md).

## Setup

To start, clone the repository:

```bash
git clone https://github.com/acmucsd/ai-chessbot.git
```

To setup the virtual environment:

```bash
python -m venv venv
# For MacOS/Linux:
source venv/bin/activate  
# On Windows (use command line if powershell gives any issues): 
venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

You are welcome to look over the repository to get better ideas on how you should implement your bot. We suggest that you start with a simple non-ML bot. We also provide three example bots, which are stored in the `data/classes/bots`.

- Random Bot: The bot chooses random move.
- Single Step Optimized Bot: The bot chooses the best move among the all possible moves at the moment. It does not consider the consequences of later moves.
- Minimax Bot: The bot is implemented with minimax algorithm (check [Resources](#resources)). However, we cannot guarentee the bot will work as expected, as it serves as an example for how your bot can be optimized.

The main function that you will be writing is the `move` function, which takes in a `side` parameter which represents if you're currently playing black or white, and a `board` parameter, which represents the current state of the board. This function should return a ((int, int), (int, int)) tuple, where the first element are the indices of the piece you wish to move, and the second element are the indices of the square you want to move to.

Some functions you may find useful in the `Board.py` file are:
-`get_board_state`, which returns the board setup as a 6x6 array. Each element in this array is either empty (which means it is not occupied by a piece), or has a two-character string in the format `{color}{Piece}`. For example `wB` would be white bishop, and `bK` would be black king. 
-`get_all_valid_moves`, which returns an array containing all legal moves
-`handle_move`, which attempts to make a move on the board, returning True if the move is valid and false otherwise 


⚠️ Warning: Please do not call API for other chess engines (such as Stockfish) because our chess rule and the implementation is different.

## Submission and Evaluation

You should only submit your [`bot.py`](data/classes/bots/bot.py), which should include class `Bot` with function `move(self, side, board)`.

Your bot will be matched against every other submitted bot in a round robin style tournament, where a win is worth 3 points, a draw is worth 1 point, and a loss is worth no points. 

Your bot will have 0.1 seconds to make a move. If your bot exceeds this time, a random move will be made on your bot's behalf. Likewise, if your bot returns an illegal move, a random move will also be made. **If your bot fails to compile, your bot will not be entered into the round robin tournamet, and score 0 points by default.**


## Packages

Please check [`requirements.py`](requirements.txt). Beside built-in modules, the modules on that list are the only 3rd party libraries we allow.

## Resources

Here is some suggested algorithms to try and implement:

1. Minimax Algorithm:
    - [Wikipedia](https://en.wikipedia.org/wiki/Minimax)
    - [Datacamp](https://www.datacamp.com/tutorial/minimax-algorithm-for-ai-in-python)
2. Alpha-beta pruning:
    - [Wikipedia](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
    - [Chess Programming Wiki](https://www.chessprogramming.org/Alpha-Beta)
3. Monte-Carlo Tree Search:
    - [Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
    - [Chess Programming Wiki](https://www.chessprogramming.org/Monte-Carlo_Tree_Search)
    - [Medium Article](https://medium.com/@ishaan.gupta0401/monte-carlo-tree-search-application-on-chess-5573fc0efb75) by Ishaan Gupta
4. Reinforcement Learning Algorithms:
    - [Chess Programming Wiki](https://www.chessprogramming.org/Reinforcement_Learning)
    - [Medium Article](https://medium.com/@samgill1256/reinforcement-learning-in-chess-73d97fad96b3) by @Aditya
    - [Policy Gradients](https://en.wikipedia.org/wiki/Policy_gradient_method)
