# StarChess.AI 2025

Welcome to ACM AI's Spring 2025 competition, StarChess.AI!

## Rule

Our chess rules are slightly different from the classical chess. Please check [`rules.md`](rules.md).

## Usage

You are welcome to look over the repository to get better ideas on how you should implement your bot. We suggest that you start with simple, without ML bot. We also provide three example bots, which are stored in the `data/classes/bots`.

- Random Bot: The bot chooses random move.
- Single Step Optimized Bot: The bot chooses the best move among the all possible moves at the moment. It does not consider the consequences of later moves.
- Minimax Bot: The bot is implemented with minimax algorithm (check [Resources](#resources)). However, we cannot guarentee the bot will work as expected, as it serves as an example for how your bot can be optimized.

⚠️ Warning: Please do not call API for other chess engines (such as Stockfish) because our chess rule and the implementation is different.

## Submission

You should only submit your [`bot.py`](data/classes/bots/bot.py), which should include class `Bot` with function `move()`. Check [`bot.py`](data/classes/bots/bot.py) to see the submission format.

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
