import pygame

from data.classes.Board import Board
from data.classes.bots.random_bot import Bot as RandomBot
from data.classes.bots.single_step_bot import Bot as SingleStepBot
from data.classes.bots.minimax_bot import Bot as MinimaxBot
from data.classes.bots.multiThreadedminimaxbot import Bot as MultiThreadedMinimaxBot
from data.classes.bots.montecarlo_bot import Bot as MonteCarloBot
from data.classes.bots.iterative import Bot as IterativeBot

pygame.init()

WINDOW_SIZE = (600, 600)
# screen = pygame.display.set_mode(WINDOW_SIZE)

board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1])


def draw(display):
    display.fill("white")
    board.draw(display)
    pygame.display.update()


SCORES_DICT = {
    " ": 1,   # pawn
    "N": 3,   # knight
    "B": 3,   # bishop
    "S": 6,   # star
    "R": 8,   # rook
    "J": 9,   # joker
    "Q": 11,   # queen
    "K": 1000 # king (Increased value for safety)
}

if __name__ == "__main__":
    running = True
    bot1 = IterativeBot()
    bot2 = IterativeBot(SCORES_DICT=SCORES_DICT)

    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            # Quit the game if the user presses the close button
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If the mouse is clicked
                if event.button == 1:
                    board.handle_click(mx, my)
                    print(board.last_captured)
        # Comment out the next 6 lines if you want to disable bot moves
        if board.turn == "black":
            m = bot1.move("black", board)
            board.handle_move(m[0], m[1])
        else:
            m = bot2.move("white", board)
            board.handle_move(m[0], m[1])


        if board.is_in_checkmate("black"):  # If black is in checkmate
            print("White wins!")
            running = False
        elif board.is_in_checkmate("white"):  # If white is in checkmate
            print("Black wins!")
            running = False
        elif board.is_in_draw():
            print("Draw!")
            running = False
            
        # Draw the board
        # draw(screen)
