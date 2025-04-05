# StarChess.AI 2025

Welcome to ACM AI's Spring 2025 competition, StarChess.AI!

## Rule

Our chess rules are slightly different from the classical chess. Please check [`rules.md`](rules.md).

## Usage

You are welcome to look over the repository to get better ideas on how you should implement your bot. We suggest that you start with simple, without ML bot. We also provide three example bots, which are stored in the `data/classes/bots`.

- Random Bot: The bot chooses random move.
- Single Step Optimized Bot: The bot chooses the best move among the all possible moves at the moment. It does not consider the consequences of later moves.
- Minimax Bot: The bot is implemented with minimax algorithm (check [wiki](https://en.wikipedia.org/wiki/Minimax) and [datacamp](https://www.datacamp.com/tutorial/minimax-algorithm-for-ai-in-python)). However, we cannot guarentee the bot will work as expected, as it serves as an example for how your bot can be optimized.

⚠️ Warning: Please do not call API for other chess engines (such as Stockfish) because our chess rule and the implementation is different.

## Submission

You should only submit your [`bot.py`](data/classes/bots/bot.py), which should include class `Bot` with function `move()`.
