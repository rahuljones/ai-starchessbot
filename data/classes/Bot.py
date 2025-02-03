import pygame
import random

random.seed(0) # fixed for testing
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
        SCORES_DICT = {
            " ": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "S": 5, # IDK what score to give
            "Q": 9,
            "K": 100
        }
        moves = self.get_possible_moves()
        best_move = []
        best_score = -1
        
        for init_pos, end_pos in moves:
            end_piece = self.board.get_piece_from_pos(end_pos)
            if end_piece:
                print(end_piece)
                score = SCORES_DICT[end_piece.notation]
            else:
                score = 0
            if best_score < score:
                best_score = score
                best_move = [(init_pos, end_pos)]
            elif best_score == score:
                best_move.append((init_pos, end_pos))
        return best_move[0] if len(best_move) == 1 else random.choice(best_move)

    def move(self):
        # pick a random move for now
        # moves = self.get_possible_moves()
        # move = random.choice(moves)

        # second version: evaluate the best move(only one layer)
        move = self.evaluate_move()

        start_pos = move[0]
        end_pos = move[1]
        self.board.handle_move(start_pos, end_pos)
