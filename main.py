import pygame
import time

from data.classes.Board import Board
from data.classes.Bot import Bot

pygame.init()

WINDOW_SIZE = (600, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)

board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1])


def draw(display):
    display.fill("white")
    board.draw(display)
    pygame.display.update()


if __name__ == "__main__":
    running = True
    bot1 = Bot("black", board)
    bot2 = Bot("white", board)
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

        # if board.turn == "black":
        #     bot1.move()
        #     time.sleep(1)
        # else:
        #     bot2.move()
        #     time.sleep(1)


        if board.is_in_checkmate("black"):  # If black is in checkmate
            print("White wins!")
            running = False
        elif board.is_in_checkmate("white"):  # If white is in checkmate
            print("Black wins!")
            running = False
        elif board.is_in_draw():
            print("Draw!")
            running = False

        #board.promote_pawn()
            
        # Draw the board
        draw(screen)
