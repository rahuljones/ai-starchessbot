# StarChess.AI 2025

Welcome to ACM AI's Spring 2025 competition, StarChess.AI!

## Rule

Our chess rules are slightly different from the classical chess.

## Usage

Bots template are store in the `data/classes/bots`. We include three example bots:

- Random Bot: The bot chooses random move.
- Single Step Optimized Bot: The bot chooses the best move among the all possible moves at the moment. It does not consider the consequences of later moves.
- Minimax Bot: The bot is implemented with minimax algorithm (check [wiki](https://en.wikipedia.org/wiki/Minimax) and [datacamp](https://www.datacamp.com/tutorial/minimax-algorithm-for-ai-in-python)). However, we cannot guarentee the bot will work as expected, as it serves as an example for how your bot can be optimized.
