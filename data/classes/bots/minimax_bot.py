
import copy
import random

class MinimaxBot:
    """
    This is a sample minimax bot that uses the minimax algorithm to choose the best move.
    It evaluates the board state and simulates moves to find the optimal one.
    This is a basic implementation and may not be optimal for all scenarios. 
    You are responsible for testing and improving the bot's performance. We recommend using depth 1 at first.
    We also recommend using a more advanced evaluation function for better performance.
    Warning: we have set a hard time limit of 0.1 second for the bot to make a move. If your bot takes 
    longer than that, it will be terminated and our evaluation server will choose random moves. We have tested
    the bot with different depths and it nearly always exceeds the time limit if the depth is greater than 2.
    """
    def __init__(self):
        self.depth = 1 ## Please set the depth <= 2 unless you are sure your bot runs within the time limit.

    def get_possible_moves(self, side, board):
        return board.get_all_valid_moves(side)
    
    def evaluate_board(self, side, board):
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
        evaluation = 0
        board_state = board.get_board_state()
        for x in board_state:
            for y in x:
                if y != "":
                    piece = y
                    piece_value = SCORES_DICT[piece[1]]
                    if piece[0] == 'b' and side == 'black':
                        evaluation += piece_value
                    elif piece[0] == 'w' and side == 'white':
                        evaluation += piece_value
                    else:
                        evaluation -= piece_value
        return evaluation
    
    def simulate_move(self, board, start_pos, end_pos):
        new_board = copy.deepcopy(board)
        new_board.handle_move(start_pos, end_pos)
        return new_board
    
    def minimax(self, board, side, depth, maximizing_player):
        if depth == 0 or board.is_in_checkmate(side):
            return self.evaluate_board(side, board)

        moves = board.get_all_valid_moves(side)
        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                eval = self.minimax(simulated_board, side, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                eval = self.minimax(simulated_board, side, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def get_best_move_minimax(self, board, side, depth):
        best_move = []
        best_value = float('-inf')
        moves = board.get_all_valid_moves(side)
        for init_pos, end_pos in moves:
            simulated_board = self.simulate_move(board, init_pos, end_pos)
            move_value = self.minimax(simulated_board, side, depth - 1, False)
            if move_value > best_value:
                best_value = move_value
                best_move = [(init_pos, end_pos)]
            elif move_value == best_value:
                best_move.append((init_pos, end_pos))
        return best_move[0] if len(best_move) == 1 else random.choice(best_move)
        
    def move(self, side, board):
        best_move = self.get_best_move_minimax(board, side, self.depth)
        return best_move