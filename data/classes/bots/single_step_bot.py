#### This is a bot that choose optimal step with only one depth ####
import random

class Bot:
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
            "K": 100, # king
            "J": 9 # joker

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

    def evaluate_board(self, side, board):
        SCORES_DICT = {
            " ": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "S": 5, 
            "Q": 9,
            "K": 100,
            "J": 9
        }
        evaluation = 0
        board_state = board.get_board_state()
        for x in board_state:
            for y in x:
                if y != "":
                    piece = board_state[x][y]
                    piece_value = SCORES_DICT[piece[1]]
                    if piece[0] == 'b' and side == 'black':
                        evaluation += piece_value
                    elif piece[0] == 'w' and side == 'white':
                        evaluation += piece_value
                    else:
                        evaluation -= piece_value
        return evaluation

    
    def move(self, side, board):
        best_move = self.evaluate_move(side, board)
        return best_move
