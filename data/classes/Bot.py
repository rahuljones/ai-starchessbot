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

    def evaluate_board(self, board):
        SCORES_DICT = {
            " ": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "S": 5, # IDK what score to give
            "Q": 9,
            "K": 100
        }
        evaluation = 0
        board_state = board.get_board_state()
        for x in board_state:
            for y in x:
                if y != "":
                    piece = board_state[x][y]
                    piece_value = SCORES_DICT[piece[1]]
                    if piece[0] == 'b' and self.side == 'black':
                        evaluation += piece_value
                    elif piece[0] == 'w' and self.side == 'white':
                        evaluation += piece_value
                    else:
                        evaluation -= piece_value
        return evaluation

    def alpha_beta(self, board ,depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_in_checkmate(self.side):
            return self.evaluate_board(board)

        moves = board.get_all_valid_moves(self.side)
        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                board = self.simulate_move(init_pos, end_pos)
                eval = self.alpha_beta(depth - 1, board, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                board = self.simulate_move(init_pos, end_pos)
                eval = self.alpha_beta(depth - 1, board, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return 
        
    def simulate_move(self, start_pos, end_pos):
        new_board = self.board.copy()  
        new_board.handle_move(start_pos, end_pos)
        return new_board
        
    def move(self):
        # pick a random move for now
        # moves = self.get_possible_moves()
        # move = random.choice(moves)

        # second version: evaluate the best move(only one layer)
        move = self.evaluate_move()

        start_pos = move[0]
        end_pos = move[1]
        self.board.handle_move(start_pos, end_pos)
