#### This is a bot that choose optimal step with only one depth ####
import random

class SingleStepBot:
    """
    A bot that chooses the best move based on a simple evaluation function.
    It evaluates the possible moves and selects the one with the highest score.
    This is a basic implementation and may not be optimal for all scenarios.
    You are responsible for testing and improving the bot's performance.
    Warning: we have set a hard time limit of 0.1 second for the bot to make a move.
    If your bot takes longer than that, it will be terminated and our evaluation server will choose random moves.
    """
    def __init__(self):
        pass

    def get_possible_moves(self, side, board):
        return board.get_all_valid_moves(side)
    
    def evaluate_move(self, side, board):
        """
        Evaluate each move for the bot
        """
        SCORES_DICT = {
            " ": 1, # pawn
            "N": 3, # knight
            "B": 3, # bishop
            "R": 5, # rook
            "S": 5, # star
            "Q": 9, # queen
            "J": 9, # joker
            "K": 100 # king
        }
        moves = self.get_possible_moves(side, board)
        best_move = []
        best_score = -1
        
        for init_pos, end_pos in moves:
            end_piece = board.get_piece_from_pos(end_pos)
            if end_piece:
                score = SCORES_DICT[end_piece.notation]
            else:
                score = 0
            if best_score < score:
                best_score = score
                best_move = [(init_pos, end_pos)]
            elif best_score == score:
                best_move.append((init_pos, end_pos))
        return best_move[0] if len(best_move) == 1 else random.choice(best_move)
    
    def move(self, side, board):
        best_move = self.evaluate_move(side, board)
        return best_move
