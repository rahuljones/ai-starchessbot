import pygame
import random


class Bot:
    def __init__(self, side, board):
        self.side = side
        self.board = board

    def get_possible_moves(self):
        return self.board.get_all_valid_moves(self.side)

    def evaluate_move(self):
        """
        Evaluate each move for the bot
        """
        pass

    def move(self):
        # pick a random move for now
        moves = self.get_possible_moves()
        move = random.choice(moves)
        start_pos = move[0]
        end_pos = move[1]
        self.board.handle_move(start_pos, end_pos)
