import pygame

from data.classes.Board import Board
from data.classes.bots.random_bot import Bot as RandomBot
from data.classes.bots.single_step_bot import Bot as SingleStepBot
from data.classes.bots.minimax_bot import Bot as MinimaxBot
from data.classes.bots.multiThreadedminimaxbot import Bot as MultiThreadedMinimaxBot
from data.classes.bots.montecarlo_bot import Bot as MonteCarloBot
from data.classes.bots.iterative import Bot as IterativeBot
from data.classes.bots.iterativeH import Bot as IterativeBotH
import argparse

pygame.init()

WINDOW_SIZE = (600, 600)
# screen = pygame.display.set_mode(WINDOW_SIZE)

board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1])


def draw(display):
    display.fill("white")
    board.draw(display)
    pygame.display.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Chess Bot Versus Mode")
    parser.add_argument("--bot1", type=str, default="IterativeBotH", help="Choose bot1 (e.g., RandomBot, SingleStepBot, MinimaxBot, MultiThreadedMinimaxBot, MonteCarloBot, IterativeBot)")
    parser.add_argument("--bot2", type=str, default="IterativeBot", help="Choose bot2 (e.g., RandomBot, SingleStepBot, MinimaxBot, MultiThreadedMinimaxBot, MonteCarloBot, IterativeBot)")
    args = parser.parse_args()

    # Dynamically load the bots based on CLI arguments
    bot1 = globals()[args.bot1]()
    bot2 = globals()[args.bot2]()

    running = True

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
