# ACM Chess Rules

## Overview

ACM Chess is a variant of traditional chess played on a **6x6 board** with modified rules and piece compositions. Each player starts with:

- **6 Pawns**
- **1 Castle** (Rook) on the far-left side
- **1 Knight** next to the Castle
- **1 Queen** and **1 King** in the center (Queen takes its color)
- **1 Bishop** to the right of the King
- **1 Star** (special piece) on the far-right side

![Rook](https://github.com/user-attachments/assets/2fcb40f5-efcc-42e1-9eb7-2fadadca4333)

## Objective

The objective of ACM Chess is to **Capture** your opponent’s King while following the modified board and piece movements.

## Piece Movements

### **Pawn**

_Pawns move as follows:_
- Forward movement: Can move one or two squares forward, but cannot capture directly in front of it.
- Capturing: Can capture **diagonally** forward one square to the left or right.
- Special moves: If this piece reaches the end of the board, it **promotes** into a Joker.

### **Castle (Rook)**

- Moves any number of squares **horizontally** or **vertically**.
- Cannot jump over other pieces.

### **Knight**

- Moves in an **L-shape**: Two squares in one direction, then one square perpendicular.
- Can **jump** over other pieces.

### **Bishop**

- Moves any number of squares **diagonally**.
- Cannot jump over other pieces.

### **Queen**

- Moves any number of squares **horizontally, vertically, or diagonally**.
- Cannot jump over other pieces.

### **King**

- Moves **one square** in any direction.

### **Star (Special Piece)**

- Moves: one square in any **diagonal** direction, or jump two squares **horizontally** or **vertically**.
- Can **jump** over other pieces.

### **Joker (Special Piece)**

- This piece is only obtained if you manage to **promote** a pawn into it by reaching the opposite side of the board.
- Moves: One square in any **diagonal**,  **horizontal** or **vertical** direction. Or jump two squares in any **diagonal**,  **horizontal** or **vertical** direction.
- Can **jump** over other pieces.

## Additional Rules

- **Board Setup:** The board is **6x6**, reducing the number of total pieces compared to standard chess.
- **Promotion Rules:** When a pawn reaches the last rank it promotes into the power Joker piece.

## Winning Conditions

- **King Capture:** If a player’s King is in captured the game ends, with the last standing king winning the game.
- **Draw conditions:** If the game does not end within 50 moves (each side) the game will be considered a draw.

## Evaluation Rules

- This tournament follows a **round-robin format**, where each team plays against every other team twice (one game as White and the other as Black)
- **Scores**:
  - **Win**: 3 points
  - **Draw**: 1 point
  - **Loss**: 0 point
- The ranking will based on the total score of each team after all matches are finished.
- **Tie-breaking methods**: If two or more teams have the same scores, tie-breaking rules will be applied in following order:
    1. **Sonneborn-Berger**: This is calculated by adding up the tournament score of each opponent you defeated, and half the tournament score of each drawn opponent.
    2. **Direct encounter**: If any players are still tied at this point, and all tied players have played against each other in the tournament, then the player with the most points out of those games is the winner.
    3. **Number of wins**: The player with the highest total number of wins breaks the tie.
    4. **Random Score**: Each tied player will be assigned a random score to break the tie.

**⚠️Notice**: We set a hard time limit **0.1 second** for each move. If your bot cannot respond by 0.1 second, the evaluation server will make a **random move** for your bot.
